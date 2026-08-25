# Calibration checks: simulation based calibration and the Geweke joint
# distribution test.
#
# These are the two tests that can fail even when every conjugate comparison
# passes, because they interrogate the sampler against the *joint* distribution
# rather than against one posterior. They live in the library rather than the
# test directory because they are part of what this repository is demonstrating,
# and because a user should be able to run them on their own model.

"""
    CalibrationProblem(build, prior_rand, simulate; names = nothing)

Everything needed to sample from the joint distribution `p(theta, y)` two
different ways:

  * `prior_rand(rng)`     -> a `NamedTuple` drawn from the prior
  * `simulate(theta, rng)` -> a data set drawn from `p(y | theta)`
  * `build(data)`         -> the [`Model`](@ref) for that data set

`names` selects which flattened parameters to track; the default is all of them.
"""
struct CalibrationProblem{B,P,S}
    build::B
    prior_rand::P
    simulate::S
    names::Union{Nothing,Vector{Symbol}}
end
CalibrationProblem(build, prior_rand, simulate; names = nothing) =
    CalibrationProblem(build, prior_rand, simulate, names)

"""
    conjugate_problem(builder, n; kwargs...)

Turn one of the conjugate reference constructors into a
[`CalibrationProblem`](@ref): `conjugate_problem(gamma_poisson, 30; a = 2.0, b = 1.0)`.
The data set size is fixed at `n`; the data themselves are redrawn from the
prior predictive at every replication, which is what makes the check a
calibration check rather than a single-posterior check.
"""
function conjugate_problem(builder, n::Int; kwargs...)
    template = builder(zeros(Int, n); kwargs...)
    prior_names = keys(template.prior)
    prior_rand = function (rng)
        return NamedTuple{prior_names}(map(d -> rand(rng, d), values(template.prior)))
    end
    simulate = function (theta, rng)
        ref = builder(zeros(Int, n); kwargs...)
        return ref.simulate(theta, rng)
    end
    build = data -> builder(data; kwargs...).model
    return CalibrationProblem(build, prior_rand, simulate)
end

# --------------------------------------------------------------------------
# Simulation based calibration
# --------------------------------------------------------------------------

"""
    SBCResult

Rank statistics from simulation based calibration, one row per replication and
one column per tracked parameter, together with the chi-square uniformity test
for each parameter.
"""
struct SBCResult
    ranks::Matrix{Int}
    names::Vector{Symbol}
    n_draws::Int
    n_bins::Int
    chisq::Vector{Float64}
    pvalue::Vector{Float64}
    ecdf_inside::Vector{Bool}
    max_deviation::Vector{Float64}
    time_seconds::Float64
end

"""
    calibrated(r::SBCResult; alpha = 0.01)

Whether every tracked parameter passed both uniformity tests: the binned
chi-square at level `alpha`, and the distribution function against its
simultaneous band.
"""
calibrated(r::SBCResult; alpha::Real = 0.01) = all(r.pvalue .> alpha) && all(r.ecdf_inside)

function Base.show(io::IO, r::SBCResult)
    @printf(io, "SBCResult(%d replications, %d posterior draws per replication)\n",
            size(r.ranks, 1), r.n_draws)
    @printf(io, "%-12s %12s %10s %14s %12s\n", "parameter", "chi-square", "p",
            "max deviation", "in the band")
    for i in eachindex(r.names)
        @printf(io, "%-12s %12.3f %10.4f %14.4f %12s\n", String(r.names[i]), r.chisq[i],
                r.pvalue[i], r.max_deviation[i], r.ecdf_inside[i] ? "yes" : "NO")
    end
end

"""
    sbc(rng, problem, sampler; n_sims = 200, n_draws = 64, thin = 8,
        n_warmup = 500, n_bins = 8)

Simulation based calibration (Talts, Betancourt, Simpson, Vehtari and Gelman,
2018).

For each replication, draw `theta0` from the prior, simulate data from
`p(y | theta0)`, sample the posterior, and record the rank of `theta0` among the
posterior draws. If the sampler targets the correct posterior, those ranks are
uniform on `0:n_draws` - and any bias, any missing Jacobian, any broken
acceptance step shows up as a departure from uniformity, usually a characteristic
shape: a U for over-dispersion, an inverted U for under-dispersion, a slope for a
location bias.

The posterior draws are thinned by `thin`, because the rank statistic assumes
independent draws and autocorrelated ones make the histogram bathtub-shaped even
for a correct sampler.
"""
function sbc(rng::AbstractRNG, problem::CalibrationProblem, sampler::AbstractSampler;
             n_sims::Int = 200, n_draws::Int = 64, thin::Int = 8, n_warmup::Int = 500,
             n_bins::Int = 8)
    t0 = time()
    theta0 = problem.prior_rand(rng)
    data0 = problem.simulate(theta0, rng)
    model0 = problem.build(data0)
    all_names = parameter_names(model0)
    names = problem.names === nothing ? all_names : problem.names
    idx = [findfirst(==(n), all_names) for n in names]
    any(isnothing, idx) && throw(ArgumentError("unknown parameter name in the problem"))

    ranks = Matrix{Int}(undef, n_sims, length(names))
    seeds = rand(rng, UInt64, n_sims)
    Threads.@threads for s in 1:n_sims
        srng = Random.Xoshiro(seeds[s])
        theta = problem.prior_rand(srng)
        data = problem.simulate(theta, srng)
        model = problem.build(data)
        truth = flatten_draw(model, unconstrain(model, theta))
        chn = sample(model, sampler, n_draws * thin; n_warmup = n_warmup, n_chains = 1,
                     rng = srng, thin = thin)
        for (j, k) in enumerate(idx)
            draws = view(chn.value, :, k, 1)
            ranks[s, j] = count(<(truth[k]), draws)
        end
    end

    chisq = Vector{Float64}(undef, length(names))
    pvalue = similar(chisq)
    inside = Vector{Bool}(undef, length(names))
    deviation = similar(chisq)
    for j in eachindex(names)
        chisq[j], pvalue[j] = rank_uniformity_test(view(ranks, :, j), n_draws, n_bins)
        # the distribution function test as well: it uses the ordering of the
        # ranks, which binning discards, and needs no bin count to be chosen
        e = rank_uniformity_ecdf(view(ranks, :, j), n_draws)
        inside[j] = e.inside
        deviation[j] = e.max_deviation
    end
    return SBCResult(ranks, collect(names), n_draws, n_bins, chisq, pvalue, inside,
                     deviation, time() - t0)
end

"""
    rank_uniformity_test(ranks, n_draws, n_bins)

Pearson chi-square test that ranks in `0:n_draws` are uniform, returning
`(statistic, p_value)`. Bins are equal width in rank space; the p value uses the
chi-square distribution with `n_bins - 1` degrees of freedom.
"""
function rank_uniformity_test(ranks, n_draws::Int, n_bins::Int)
    counts = zeros(Int, n_bins)
    for r in ranks
        b = min(floor(Int, r * n_bins / (n_draws + 1)) + 1, n_bins)
        counts[b] += 1
    end
    expected = length(ranks) / n_bins
    stat = sum((counts .- expected) .^ 2) ./ expected
    p = 1 - cdf(Gamma((n_bins - 1) / 2, 0.5), stat)      # chi-square with n_bins-1 dof
    return stat, p
end

"""
    rank_histogram(result, j = 1; n_bins = result.n_bins)

Binned rank counts for one parameter, for plotting.
"""
function rank_histogram(r::SBCResult, j::Int = 1; n_bins::Int = r.n_bins)
    counts = zeros(Int, n_bins)
    for rank in view(r.ranks, :, j)
        b = min(floor(Int, rank * n_bins / (r.n_draws + 1)) + 1, n_bins)
        counts[b] += 1
    end
    return counts
end

# --------------------------------------------------------------------------
# Geweke joint distribution test
# --------------------------------------------------------------------------

"""
    GewekeResult

Moments of the tracked parameters under the two simulators, and the z statistic
comparing them.
"""
struct GewekeResult
    names::Vector{Symbol}
    marginal_mean::Vector{Float64}
    successive_mean::Vector{Float64}
    z_mean::Vector{Float64}
    marginal_second::Vector{Float64}
    successive_second::Vector{Float64}
    z_second::Vector{Float64}
    n_marginal::Int
    n_successive::Int
    time_seconds::Float64
end

function Base.show(io::IO, r::GewekeResult)
    @printf(io, "GewekeResult(%d marginal draws, %d successive draws)\n", r.n_marginal, r.n_successive)
    @printf(io, "%-12s %12s %12s %8s %12s %12s %8s\n", "parameter", "mean (mc)", "mean (sc)",
            "z", "E[x^2] (mc)", "E[x^2] (sc)", "z")
    for i in eachindex(r.names)
        @printf(io, "%-12s %12.4f %12.4f %8.2f %12.4f %12.4f %8.2f\n", String(r.names[i]),
                r.marginal_mean[i], r.successive_mean[i], r.z_mean[i],
                r.marginal_second[i], r.successive_second[i], r.z_second[i])
    end
end

"""
    geweke(rng, problem, sampler; n_marginal = 20_000, n_successive = 20_000,
           n_steps = 5, n_warmup = 200)

Geweke's (2004) joint distribution test.

The *marginal-conditional* simulator draws `theta` from the prior and `y` from
`p(y | theta)` independently each time. The *successive-conditional* simulator
alternates one (or `n_steps`) transitions of the sampler's kernel targeting
`p(theta | y)` with a fresh draw of `y` from `p(y | theta)`. Both leave the
joint `p(theta, y)` invariant, so the `theta` marginals must agree. They only
agree if the transition kernel really targets the posterior, which is why this
catches errors that a single-posterior comparison cannot: a wrong normalising
constant, a missing Jacobian, an acceptance ratio that is off by a factor that
happens to cancel in one particular posterior.

The reported `z` uses the Monte Carlo standard error of both samples, with the
successive-conditional error inflated by its own autocorrelation.

`n_steps` matters more than it looks. With a single kernel step per sweep the
successive-conditional chain stays autocorrelated enough that its standard error
is optimistic, and the z statistic occasionally reaches 7 on a *correct*
sampler; at five steps per sweep the same six seeds all report |z| < 1.7.
"""
function geweke(rng::AbstractRNG, problem::CalibrationProblem, sampler::AbstractSampler;
                n_marginal::Int = 20_000, n_successive::Int = 20_000, n_steps::Int = 5,
                n_warmup::Int = 200)
    t0 = time()
    theta0 = problem.prior_rand(rng)
    data0 = problem.simulate(theta0, rng)
    model0 = problem.build(data0)
    all_names = parameter_names(model0)
    names = problem.names === nothing ? all_names : problem.names
    idx = [findfirst(==(n), all_names) for n in names]
    P = length(idx)

    # Marginal-conditional: independent draws from the prior.
    A = Matrix{Float64}(undef, n_marginal, P)
    for i in 1:n_marginal
        theta = problem.prior_rand(rng)
        flat = flatten_draw(model0, unconstrain(model0, theta))
        for (j, k) in enumerate(idx)
            A[i, j] = flat[k]
        end
    end

    # Successive-conditional: alternate data given theta, theta given data.
    # The kernel is adapted once, on one data set, and then held fixed. A kernel
    # whose step size is re-selected from the current position at every sweep is
    # not reversible, and the test correctly reports that as a failure - which is
    # how this code came to be written this way.
    B = Matrix{Float64}(undef, n_successive, P)
    theta = problem.prior_rand(rng)
    y = unconstrain(model0, theta)
    state = init_state(rng, model0, sampler, y; n_warmup = n_warmup)
    for _ in 1:n_warmup
        y, _ = step!(rng, model0, sampler, state, true)
    end
    state = finish_warmup!(rng, model0, sampler, state)
    for i in 1:n_successive
        data = problem.simulate(theta, rng)
        model = problem.build(data)
        refresh!(model, sampler, state)
        local ynew = y
        for _ in 1:max(n_steps, 1)
            ynew, _ = step!(rng, model, sampler, state, false)
        end
        y = ynew
        theta, _ = constrain(model, y)
        flat = flatten_draw(model, y)
        for (j, k) in enumerate(idx)
            B[i, j] = flat[k]
        end
    end

    mm = [Statistics.mean(view(A, :, j)) for j in 1:P]
    sm = [Statistics.mean(view(B, :, j)) for j in 1:P]
    ms = [Statistics.mean(view(A, :, j) .^ 2) for j in 1:P]
    ss = [Statistics.mean(view(B, :, j) .^ 2) for j in 1:P]
    zmean = Vector{Float64}(undef, P)
    zsecond = Vector{Float64}(undef, P)
    for j in 1:P
        a = view(A, :, j)
        b = view(B, :, j)
        se_a = Statistics.std(a) / sqrt(length(a))
        se_b = mcse_mean(reshape(collect(b), :, 1))
        zmean[j] = (sm[j] - mm[j]) / sqrt(se_a^2 + se_b^2)
        a2 = a .^ 2
        b2 = b .^ 2
        se_a2 = Statistics.std(a2) / sqrt(length(a2))
        se_b2 = mcse_mean(reshape(b2, :, 1))
        zsecond[j] = (ss[j] - ms[j]) / sqrt(se_a2^2 + se_b2^2)
    end
    return GewekeResult(collect(names), mm, sm, zmean, ms, ss, zsecond,
                        n_marginal, n_successive, time() - t0)
end
