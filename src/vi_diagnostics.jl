# Did the variational approximation work?
#
# ADVI reports an ELBO, which is a lower bound on the log evidence and says
# nothing on its own about how close `q` is to the posterior. The bound is
# checked against known evidence in this package's tests; a user fitting their
# own model has no such reference, and the ELBO going flat only means the
# optimiser stopped.
#
# Yao, Vehtari, Simpson and Gelman ("Yes, but did it work?", 2018) give two
# checks that need no reference. The first is the Pareto shape of the importance
# weights from `q` to the posterior, which is cheap because draws from `q` are
# free and the ratio is available in closed form. The second is calibration
# across replications, which is expensive and catches a different failure: an
# approximation can have well behaved weights and still be systematically
# narrow, and mean-field variational inference is narrow essentially always.

"""
    log_variational_density(family, params, z, d)

Log density of `q` at the point `transform_sample(family, params, z, d)`.

Computed from `z` rather than from the point, which is both exact and cheaper:
the transform is affine, so the density is the standard normal density of `z`
divided by the determinant of the scale, and that determinant is already what
[`entropy`](@ref) is built from.
"""
function log_variational_density(family::VariationalFamily, params::AbstractVector,
                                 z::AbstractVector, d::Int)
    log_det_scale = entropy(family, params, d) - d / 2 * (1 + LOG2PI)
    return -sum(abs2, z) / 2 - d / 2 * LOG2PI - log_det_scale
end

"""
    psis_check(r::VIResult; n_samples = 4000, rng)

Pareto shape of the importance weights from the fitted `q` to the posterior,
returned as a [`PSISResult`](@ref).

This is the cheap half of "did it work". `k` below 0.5 says the approximation is
close enough that importance sampling can correct what remains; above 0.7 the
weights have no usable variance and the approximation should not be corrected,
reported, or believed.

It is a check on the ratio, not on the shape. An approximation can have a small
`k` and still be narrow in a way that matters for an interval, which is what
[`vsbc`](@ref) is for.
"""
function psis_check(r::VIResult; n_samples::Int = 4000,
                    rng::AbstractRNG = Random.default_rng())
    d = dimension(r.model)
    ratios = Vector{Float64}(undef, n_samples)
    for i in 1:n_samples
        z = randn(rng, d)
        y = transform_sample(r.family, r.params, z, d)
        ratios[i] = logdensity(r.model, y) - log_variational_density(r.family, r.params, z, d)
    end
    return psis(ratios)
end

"""
    VSBCResult

Variational simulation based calibration: for each replication, the probability
the fitted `q` assigns to being below the parameter that generated the data.

Under a perfect approximation those probabilities are uniform. Two things are
reported about them, because they fail differently:

  * `uniform` is whether the whole distribution stayed inside the simultaneous
    band, the same test [`sbc`](@ref) uses.
  * `bias` is the mean of the probabilities minus a half, with its standard
    error. A symmetric departure from uniformity is the signature of an
    approximation that is the right shape and the wrong width, which mean-field
    variational inference is by construction; an asymmetric one means the
    location is wrong, which is a different and worse problem.

Separating them matters because a mean-field fit will usually fail the
uniformity test and should: the question is whether it is also biased.
"""
struct VSBCResult
    names::Vector{Symbol}
    p::Matrix{Float64}              # replications x parameters
    uniform::Vector{Bool}
    max_deviation::Vector{Float64}
    bias::Vector{Float64}
    bias_se::Vector{Float64}
    n_sims::Int
end

"""
    unbiased(r::VSBCResult; nse = 3)

Whether every parameter's calibration probabilities are centred, within `nse`
standard errors. This is the half of [`VSBCResult`](@ref) a mean-field fit can
be expected to pass.
"""
unbiased(r::VSBCResult; nse::Real = 3) = all(abs.(r.bias) .<= nse .* r.bias_se)

calibrated(r::VSBCResult) = all(r.uniform)

function Base.show(io::IO, r::VSBCResult)
    @printf(io, "VSBCResult(%d replications)\n", r.n_sims)
    @printf(io, "%-12s %10s %10s %10s %12s\n", "parameter", "bias", "std err", "max dev",
            "uniform")
    for i in eachindex(r.names)
        @printf(io, "%-12s %10.4f %10.4f %10.4f %12s\n", String(r.names[i]), r.bias[i],
                r.bias_se[i], r.max_deviation[i], r.uniform[i] ? "yes" : "no")
    end
    return print(io, "bias is the mean probability minus a half: a width problem is ",
                 "symmetric, a location problem is not")
end

"""
    vsbc(rng, problem, sampler = ADVI(); n_sims = 200, confidence = 0.95)

Variational simulation based calibration over a [`CalibrationProblem`](@ref).

Each replication draws a parameter from the prior, simulates data, fits the
approximation, and records the probability `q` puts below the parameter that
generated the data. Unlike [`sbc`](@ref) there is no rank and no thinning,
because draws from `q` are independent and the probability is available in
closed form, which is why this is affordable at all.
"""
function vsbc(rng::AbstractRNG, problem::CalibrationProblem, sampler::ADVI = ADVI();
              n_sims::Int = 200, confidence::Real = 0.95)
    n_sims >= 10 || throw(ArgumentError("need at least 10 replications, got $n_sims"))
    theta0 = problem.prior_rand(rng)
    model0 = problem.build(problem.simulate(theta0, rng))
    all_names = parameter_names(model0)
    names = problem.names === nothing ? all_names : problem.names
    idx = [findfirst(==(n), all_names) for n in names]
    any(isnothing, idx) &&
        throw(ArgumentError("names $(names) are not all parameters of the model"))

    p = Matrix{Float64}(undef, n_sims, length(names))
    seeds = rand(rng, UInt64, n_sims)
    Threads.@threads for s in 1:n_sims
        r = Random.Xoshiro(seeds[s])
        theta = problem.prior_rand(r)
        model = problem.build(problem.simulate(theta, r))
        fit = sample(model, sampler; rng = r)
        truth = flatten_draw(model, unconstrain(model, theta))
        # The probability q puts below the truth, from draws rather than from a
        # normal cdf: q is Gaussian in the unconstrained space and the reported
        # parameters are not, so the transform has to be applied first.
        draws = posterior_samples(fit, 2000; rng = r)
        for (j, k) in pairs(idx)
            p[s, j] = count(<=(truth[k]), view(draws.value, :, k, 1)) / size(draws.value, 1)
        end
    end

    uniform = Bool[]
    maxdev = Float64[]
    for j in eachindex(names)
        u = uniformity_ecdf(view(p, :, j); confidence = confidence)
        push!(uniform, u.inside)
        push!(maxdev, u.max_deviation)
    end
    bias = [Statistics.mean(view(p, :, j)) - 0.5 for j in eachindex(names)]
    bias_se = [Statistics.std(view(p, :, j)) / sqrt(n_sims) for j in eachindex(names)]
    return VSBCResult(collect(names), p, uniform, maxdev, bias, bias_se, n_sims)
end

vsbc(problem::CalibrationProblem, sampler::ADVI = ADVI(); kwargs...) =
    vsbc(Random.default_rng(), problem, sampler; kwargs...)
