# Checks on the prior, which the rest of this package does not make.
#
# Everything else here interrogates the posterior, and a prior that puts its mass
# somewhere absurd is invisible in that view until it has already distorted the
# answer, at which point attributing the distortion is hard.
#
# Two checks. The prior predictive one asks what data the model believes in
# before seeing any, which catches a prior whose implications nobody worked out;
# priors are usually written down one parameter at a time and their joint
# consequence for observable quantities is what goes wrong. The sensitivity one
# asks how much the posterior moves when the prior is sharpened or flattened,
# which separates a prior the data overwhelms from one it does not.

"""
    prior_predictive(problem, n; rng)

Draw `n` data sets from the prior predictive: a parameter from the prior, then
data given that parameter.

Takes a [`CalibrationProblem`](@ref), which already carries both halves, so a
model set up for calibration needs nothing new to be checked this way. Returns
`(data, theta)`.
"""
function prior_predictive(problem::CalibrationProblem, n::Int;
                          rng::AbstractRNG = Random.default_rng())
    n > 0 || throw(ArgumentError("n must be positive, got $n"))
    thetas = [problem.prior_rand(rng) for _ in 1:n]
    data = [problem.simulate(theta, rng) for theta in thetas]
    return (data = identity.(data), theta = identity.(thetas))
end

"""
    prior_predictive_check(problem, observed, statistic, n; rng)

Compare a statistic of the observed data against its distribution under the
prior predictive, returning a [`PredictiveCheck`](@ref).

This is the posterior predictive check run before the data are used, and it is
answering a different question. A posterior predictive p value near zero says
the fitted model cannot reproduce the data; a prior predictive one says the
model did not consider data like this plausible in the first place. The second
is the one that catches a prior nobody sanity checked, and it is checkable
before any sampling has been done at all.
"""
function prior_predictive_check(problem::CalibrationProblem, observed, statistic, n::Int;
                                rng::AbstractRNG = Random.default_rng())
    reps = prior_predictive(problem, n; rng = rng).data
    return predictive_check(observed, reps, statistic)
end

"""
    SensitivityResult

How far the posterior moves when the prior is raised to a power.

`alphas` are the powers, `means` and `stds` the posterior summaries at each, and
`sensitivity` the slope of the posterior mean against `log(alpha)` measured in
posterior standard deviations, which is what makes the number comparable across
parameters on different scales.

`sensitivity_se` is the Monte Carlo standard error of that slope, by batch
means over contiguous blocks of the draw sequence. A slope smaller than its own
uncertainty is not a small slope, it is an unmeasured one, and
[`sensitive`](@ref) refuses to report it as either. On the conjugate models here
the error is small, because every power reuses the same draws and what varies
between them is a smooth function of those draws rather than fresh noise; that
is the property being relied on, so it is worth reporting rather than assuming.

`khat` is the largest Pareto shape over the powers. The summaries at each power
come from reweighting one set of draws rather than from refitting, so `khat`
says whether that reweighting was possible: above 0.7 the numbers at that power
are not to be believed and the fit has to be redone with
[`power_scale`](@ref).
"""
struct SensitivityResult
    names::Vector{Symbol}
    alphas::Vector{Float64}
    means::Matrix{Float64}          # alphas x parameters
    stds::Matrix{Float64}
    sensitivity::Vector{Float64}
    sensitivity_se::Vector{Float64}
    khat::Vector{Float64}
    base_index::Int
end

"""
    sensitive(r::SensitivityResult; threshold = 0.05)

Which parameters moved more than `threshold` posterior standard deviations per
unit change in `log(alpha)`.

A parameter has to clear two bars: the slope exceeds `threshold`, and it is more
than two of its own standard errors from zero. Without the second, a slope that
is entirely Monte Carlo error gets reported as a finding whenever the noise
happens to be large, which is the failure mode of every diagnostic that reports
a point estimate on its own.

The threshold is a convention. What the number means is not: 0.05 says that
raising the weight of the prior by a factor of e moves the posterior mean by
about a twentieth of its own spread, which is small enough that no conclusion
should turn on it.
"""
sensitive(r::SensitivityResult; threshold::Real = 0.05) =
    r.names[findall(j -> abs(r.sensitivity[j]) > threshold &&
                        abs(r.sensitivity[j]) > 2 * r.sensitivity_se[j],
                    eachindex(r.names))]

"""
    reliable(r::SensitivityResult; threshold = 0.7)

Whether the reweighting behind every power was trustworthy.
"""
reliable(r::SensitivityResult; threshold::Real = 0.7) = maximum(r.khat) < threshold

function Base.show(io::IO, r::SensitivityResult)
    flagged = Set(sensitive(r))
    @printf(io, "%-12s %12s %10s %8s %16s\n", "parameter", "sensitivity", "std error", "k",
            "verdict")
    for i in eachindex(r.names)
        verdict = r.names[i] in flagged ? "prior matters" :
                  (abs(r.sensitivity[i]) <= 2 * r.sensitivity_se[i] ? "not measurable" :
                   "data dominates")
        @printf(io, "%-12s %12.4f %10.4f %8.3f %16s\n", String(r.names[i]), r.sensitivity[i],
                r.sensitivity_se[i], maximum(r.khat), verdict)
    end
    return print(io, "posterior mean shift per unit log(alpha), in posterior standard deviations")
end

"""
    power_scale(tm::TemperedModel, alpha)

The `Model` whose log density is `alpha * log prior + log likelihood`.

The mirror of [`at`](@ref), which scales the likelihood instead. `alpha` above
one sharpens the prior and below one flattens it; at zero the prior is gone
entirely, which is a proper posterior only if the likelihood makes it one.

The Jacobian of the transform is not scaled, because it belongs to the change of
variables rather than to the prior. That is also why the importance ratio
[`prior_sensitivity`](@ref) uses is exactly `(alpha - 1)` times the prior: the
Jacobian cancels.
"""
function power_scale(tm::TemperedModel{<:Model}, alpha::Real)
    return Model(NamedTuple{tm.prior.names}(tm.prior.transforms),
                 theta -> alpha * tm.prior.logjoint(theta) + tm.loglik(theta))
end

"""
    prior_sensitivity(tm::TemperedModel, chains; alphas = [0.8, 1.0, 1.25], thin = 1)

Report how far the posterior moves when the prior is raised to a power, by
reweighting the draws in `chains` rather than by refitting.

Power scaling is the cheap prior sensitivity analysis: rather than asking a user
to write several priors, it takes the one they wrote and varies how much of it
there is. A posterior that barely notices was determined by the data; one that
tracks the power was determined by the prior, and that is worth knowing before
the answer is reported rather than after someone asks.

Reweighting rather than refitting is cheaper, and it is also more accurate by a
margin that matters. Refitting draws each posterior with its own random numbers,
so the difference between two of them carries two independent Monte Carlo
errors, while the shift being measured is a small fraction of a posterior
standard deviation. On a Gamma-Poisson model with 200 observations, where the
answer is available in closed form as -0.1104, refitting at 4000 draws per power
gave -0.142 and reweighting the same draws gave -0.1103. The reweighting has an
error of its own, the reliability of the importance sampling, and that is
reported as `khat` rather than assumed away.
"""
function prior_sensitivity(tm::TemperedModel{<:Model}, c::Chains;
                           alphas = [0.8, 1.0, 1.25], thin::Int = 1, batches::Int = 10)
    as = collect(float.(alphas))
    all(>(0), as) || throw(DomainError(as, "powers must be positive"))
    base = findfirst(≈(1.0), as)
    base === nothing && throw(ArgumentError("alphas must include 1.0, the prior as written"))

    model = at(tm, 1.0)
    thetas = collect(parameter_draws(model, c; thin = thin))
    logprior = [tm.prior.logjoint(theta) for theta in thetas]
    draws = reduce(vcat, (reshape(flatten_draw(model, unconstrain(model, theta)), 1, :)
                          for theta in thetas))
    names = parameter_names(model)
    P = length(names)

    means = Matrix{Float64}(undef, length(as), P)
    stds = similar(means)
    khat = Vector{Float64}(undef, length(as))
    for (k, a) in pairs(as)
        r = psis((a - 1) .* logprior)
        w = exp.(r.log_weights)
        khat[k] = r.k
        for j in 1:P
            m = sum(w .* view(draws, :, j))
            means[k, j] = m
            stds[k, j] = sqrt(max(sum(w .* (view(draws, :, j) .- m) .^ 2), 0.0))
        end
    end

    slope = _sensitivity_slope(as, means, stds, base, P)

    # Batch means for the uncertainty on that slope. Contiguous blocks of the
    # draw sequence rather than independent refits: the same reweighting is done
    # inside each block, so what this measures is the sampling error of the
    # slope and not the difference between two fits.
    B = max(min(batches, size(draws, 1) ÷ 50), 2)
    edges = round.(Int, range(0, size(draws, 1); length = B + 1))
    per_batch = Matrix{Float64}(undef, B, P)
    for b in 1:B
        idx = (edges[b] + 1):edges[b + 1]
        bm = Matrix{Float64}(undef, length(as), P)
        bs = similar(bm)
        for (k, a) in pairs(as)
            lr = (a - 1) .* view(logprior, idx)
            w = exp.(lr .- logsumexp(lr))
            for j in 1:P
                col = view(draws, idx, j)
                m = sum(w .* col)
                bm[k, j] = m
                bs[k, j] = sqrt(max(sum(w .* (col .- m) .^ 2), 0.0))
            end
        end
        per_batch[b, :] = _sensitivity_slope(as, bm, bs, base, P)
    end
    se = [Statistics.std(view(per_batch, :, j)) / sqrt(B) for j in 1:P]

    return SensitivityResult(collect(names), as, means, stds, slope, se, khat, base)
end

# Least squares slope against log(alpha), so that more than two powers
# contribute, scaled by the posterior spread at the prior as written.
function _sensitivity_slope(as, means, stds, base::Int, P::Int)
    x = log.(as)
    xbar = Statistics.mean(x)
    denom = sum((x .- xbar) .^ 2)
    return [sum((x .- xbar) .* (means[:, j] .- Statistics.mean(means[:, j]))) / denom /
            max(stds[base, j], eps()) for j in 1:P]
end
