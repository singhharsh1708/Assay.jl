# Pareto smoothed importance sampling, and leave-one-out cross validation
# built on it.
#
# Importance sampling estimates are only as good as their weights, and the
# weights are frequently terrible: a few draws carry almost all the mass and
# the estimator has infinite variance without ever saying so. Pareto smoothing
# (Vehtari, Simpson, Gelman, Yao and Gabry, 2024) fits a generalised Pareto
# distribution to the largest weights and replaces them with the fitted
# quantiles, which stabilises the estimate. The shape parameter of that fit,
# `k`, is the diagnostic that matters:
#
#     k < 0.5        the variance of the weights is finite, estimates converge fast
#     0.5 <= k < 0.7 finite mean, infinite variance, usable with more draws
#     k >= 0.7       the estimate is not to be trusted, whatever it says
#
# That last line is the reason this file exists. Every other importance
# sampling implementation returns a number; this one returns a number and a
# statement about whether the number means anything.

"""
    gpd_fit(x)

Fit a generalised Pareto distribution to positive, sorted-ascending `x` by the
empirical Bayes method of Zhang and Stephens (2009), returning `(k, sigma)`.

The profile likelihood is evaluated on a grid of the single parameter `theta`
and averaged over it, which is both faster and more stable than maximising a
two parameter likelihood. `k` is then shrunk towards `1/2` with weight
equivalent to ten observations, which keeps the estimate from being wild on
short tails without moving it materially on long ones.
"""
function gpd_fit(x::AbstractVector{<:Real}; weakly_informative::Bool = true,
                 min_grid_points::Int = 30)
    n = length(x)
    n >= 5 || throw(ArgumentError("need at least 5 tail points to fit, got $n"))
    xs = issorted(x) ? x : sort(x)
    xs[1] > 0 || throw(DomainError(xs[1], "generalised Pareto fit needs positive values"))

    prior = 3.0
    m = min_grid_points + floor(Int, sqrt(n))
    quarter = xs[max(1, round(Int, n / 4 + 0.5))]           # first quartile
    theta = [1 / xs[n] + (1 - sqrt(m / (j - 0.5))) / prior / quarter for j in 1:m]

    # profile log likelihood at each grid point, then a weighted average
    lprofile = similar(theta)
    @inbounds for j in eachindex(theta)
        a = -theta[j]
        kj = Statistics.mean(log1p.(a .* xs))
        lprofile[j] = n * (log(a / kj) - kj - 1)
    end
    lse = logsumexp(lprofile)
    w = exp.(lprofile .- lse)
    theta_hat = sum(theta .* w)

    k = Statistics.mean(log1p.(-theta_hat .* xs))
    sigma = -k / theta_hat
    if weakly_informative
        a = 10.0
        k = k * n / (n + a) + a * 0.5 / (n + a)
    end
    return k, sigma
end

"""
    gpd_quantile(p, k, sigma)

Quantile function of the generalised Pareto distribution with location zero.
"""
function gpd_quantile(p::Real, k::Real, sigma::Real)
    return abs(k) < 1e-10 ? -sigma * log1p(-p) : sigma * expm1(-k * log1p(-p)) / k
end

"""
    PSISResult

Smoothed log weights, the Pareto shape `k`, and the number of tail draws that
were smoothed. `k` is the diagnostic: see [`psis`](@ref).
"""
struct PSISResult
    log_weights::Vector{Float64}
    k::Float64
    tail_length::Int
end

"""
    reliable(r::PSISResult; threshold = 0.7)

Whether the Pareto shape is small enough for the estimate to be trusted.
"""
reliable(r::PSISResult; threshold::Real = 0.7) = r.k < threshold

function Base.show(io::IO, r::PSISResult)
    verdict = r.k < 0.5 ? "good" : (r.k < 0.7 ? "usable, high variance" : "not to be trusted")
    @printf(io, "PSISResult(k = %.3f, %s, %d of %d draws smoothed)",
            r.k, verdict, r.tail_length, length(r.log_weights))
end

"""
    psis(log_ratios; tail_length = nothing)

Pareto smoothed importance sampling of a vector of log importance ratios.

The largest `tail_length` ratios are replaced by the quantiles of a generalised
Pareto distribution fitted to them, all weights are truncated at the largest raw
ratio, and the result is normalised. Returns a [`PSISResult`](@ref) whose `k` is
the shape of that fit.

The default tail length is `min(S/5, 3 sqrt(S))`, which is the choice made in
the reference implementation: enough points to fit a two parameter tail, few
enough that the fit only describes the tail.
"""
function psis(log_ratios::AbstractVector{<:Real}; tail_length::Union{Nothing,Int} = nothing)
    S = length(log_ratios)
    lw = collect(float.(log_ratios))
    M = tail_length === nothing ? min(S ÷ 5, ceil(Int, 3 * sqrt(S))) : tail_length
    if M < 5 || S < 10
        # too few draws to fit a tail: normalise and report an honest k of Inf
        lw .-= logsumexp(lw)
        return PSISResult(lw, Inf, 0)
    end

    perm = sortperm(lw)
    tail_idx = perm[(S - M + 1):S]
    cutoff = lw[perm[S - M]]                       # largest weight left unsmoothed

    tail = lw[tail_idx]
    x = exp.(tail) .- exp(cutoff)                  # exceedances over the threshold

    if any(!isfinite, x)
        # ratios that are not finite cannot be smoothed or trusted
        lw .-= logsumexp(lw)
        return PSISResult(lw, Inf, M)
    end
    if maximum(x) <= 1e-12 * max(exp(cutoff), 1.0)
        # the weights are effectively constant, which is the ideal case rather
        # than a failure: proposal and target agree, there is no tail to fit,
        # and reporting Inf here would flag the one situation that never needs
        # flagging.
        lw .-= logsumexp(lw)
        return PSISResult(lw, -Inf, M)
    end
    if !(minimum(x) > 0)
        # ties at the threshold: fit what is strictly above it
        keep = x .> 0
        count(keep) >= 5 || (lw .-= logsumexp(lw); return PSISResult(lw, Inf, M))
        tail_idx = tail_idx[keep]
        tail = tail[keep]
        x = x[keep]
        M = length(x)
    end

    order = sortperm(x)
    k, sigma = gpd_fit(x[order])
    if isfinite(k) && k < 1
        # replace the ordered tail by the fitted quantiles at (z - 0.5) / M
        smoothed = similar(tail)
        for (rank, pos) in enumerate(order)
            p = (rank - 0.5) / M
            smoothed[pos] = log(gpd_quantile(p, k, sigma) + exp(cutoff))
        end
        lw[tail_idx] = smoothed
    end

    # truncate at the largest raw ratio, then normalise
    lw .= min.(lw, maximum(log_ratios))
    lw .-= logsumexp(lw)
    return PSISResult(lw, k, M)
end

# --------------------------------------------------------------------------
# Leave-one-out cross validation
# --------------------------------------------------------------------------

"""
    LOOResult

Expected log pointwise predictive density estimated by leave-one-out cross
validation, its standard error, the effective number of parameters, and the
per-observation Pareto shapes.

`khat` is not decoration. An observation with `khat >= 0.7` is one the
importance sampling approximation could not handle, which usually means that
observation is highly influential, and its contribution to `elpd` should not be
believed.
"""
struct LOOResult
    elpd::Float64
    se::Float64
    p_loo::Float64
    pointwise::Vector{Float64}
    khat::Vector{Float64}
    n_draws::Int
end

"""
    problematic(r::LOOResult; threshold = 0.7)

Indices of observations whose Pareto shape exceeds `threshold`.
"""
problematic(r::LOOResult; threshold::Real = 0.7) = findall(>=(threshold), r.khat)

function Base.show(io::IO, r::LOOResult)
    bad = length(problematic(r))
    @printf(io, "LOOResult(elpd = %.2f ± %.2f, p_loo = %.2f, %d draws", r.elpd, r.se, r.p_loo, r.n_draws)
    bad == 0 ? print(io, ", all k below 0.7)") : @printf(io, ", %d observations with k >= 0.7)", bad)
end

"""
    loo(log_likelihood)

Leave-one-out cross validation by Pareto smoothed importance sampling, from a
`draws x observations` matrix of pointwise log likelihoods.

For each observation the importance ratios are `-log p(y_i | theta)`, because
the leave-one-out posterior is the full posterior reweighted by the reciprocal
of that observation's likelihood. Smoothing those ratios is what makes the
estimate usable; the returned `khat` says for which observations it worked.

    ll = [logpdf(Normal(draw.mu, sigma), y[i]) for draw in draws, i in eachindex(y)]
    result = loo(ll)
"""
function loo(log_likelihood::AbstractMatrix{<:Real})
    S, n = size(log_likelihood)
    S >= 10 || throw(ArgumentError("need at least 10 draws, got $S"))
    elpd_i = Vector{Float64}(undef, n)
    lpd_i = Vector{Float64}(undef, n)
    khat = Vector{Float64}(undef, n)
    logS = log(S)
    for i in 1:n
        ll = view(log_likelihood, :, i)
        r = psis(-ll)
        elpd_i[i] = logsumexp(r.log_weights .+ ll)      # weights already normalised
        lpd_i[i] = logsumexp(ll) - logS
        khat[i] = r.k
    end
    elpd = sum(elpd_i)
    se = sqrt(n * Statistics.var(elpd_i))
    p_loo = sum(lpd_i) - elpd
    return LOOResult(elpd, se, p_loo, elpd_i, khat, S)
end

"""
    waic(log_likelihood)

Widely applicable information criterion, on the same `elpd` scale as
[`loo`](@ref), returned as `(elpd, se, p_waic)`.

Kept because it is often asked for and costs nothing, but leave-one-out is the
better estimator: WAIC has no diagnostic that tells you when it has failed,
which is the property this package exists to object to.
"""
function waic(log_likelihood::AbstractMatrix{<:Real})
    S, n = size(log_likelihood)
    logS = log(S)
    lpd_i = [logsumexp(view(log_likelihood, :, i)) - logS for i in 1:n]
    pen_i = [Statistics.var(view(log_likelihood, :, i)) for i in 1:n]
    elpd_i = lpd_i .- pen_i
    return (elpd = sum(elpd_i), se = sqrt(n * Statistics.var(elpd_i)), p_waic = sum(pen_i))
end

"""
    loo_compare(results...)

Compare models on `elpd`, best first, returning a named tuple per model with the
difference from the best and the standard error of that difference.

The standard error of a difference is computed from the paired pointwise
differences rather than from the two separate standard errors, because the two
estimates are computed on the same observations and are strongly correlated.
"""
function loo_compare(results::LOOResult...)
    length(results) >= 2 || throw(ArgumentError("need at least two models to compare"))
    n = length(first(results).pointwise)
    all(r -> length(r.pointwise) == n, results) ||
        throw(DimensionMismatch("models must be compared on the same observations"))
    order = sortperm(collect(r.elpd for r in results); rev = true)
    best = results[order[1]]
    return [(model = i, elpd = results[i].elpd, delta = results[i].elpd - best.elpd,
             se_delta = i == order[1] ? 0.0 :
                        sqrt(n * Statistics.var(results[i].pointwise .- best.pointwise)),
             max_khat = maximum(results[i].khat)) for i in order]
end
