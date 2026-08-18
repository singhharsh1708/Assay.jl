# Convergence diagnostics, implemented here rather than imported.
#
# The estimators follow Vehtari, Gelman, Simpson, Carpenter and Bürkner (2021),
# "Rank-normalization, folding, and localization: an improved R-hat":
#
#   * split the chains in half before computing R-hat, so that a single chain
#     that has not mixed with itself is still caught;
#   * rank-normalise, so the diagnostics do not assume finite variance;
#   * report bulk and tail effective sample size separately, because a chain can
#     have excellent central mixing and useless tails.
#
# Autocovariances are accumulated directly rather than through an FFT. That is
# `O(n * lag)` instead of `O(n log n)`, but the Geyer truncation almost always
# fires within a few hundred lags, and it keeps the package free of an FFT
# dependency.

"""
    autocov(x, maxlag)

Biased (1/n normalised) autocovariance of a single chain at lags `0:maxlag`.
"""
function autocov(x::AbstractVector{<:Real}, maxlag::Int)
    n = length(x)
    xbar = Statistics.mean(x)
    out = Vector{Float64}(undef, maxlag + 1)
    @inbounds for t in 0:maxlag
        s = 0.0
        for i in 1:(n - t)
            s += (x[i] - xbar) * (x[i + t] - xbar)
        end
        out[t + 1] = s / n
    end
    return out
end

"""
    split_chains(x)

Split each column of a `draws x chains` matrix in half, returning a
`floor(draws/2) x 2*chains` matrix. Split R-hat is the diagnostic; unsplit R-hat
misses a chain that drifts monotonically.
"""
function split_chains(x::AbstractMatrix{<:Real})
    n, m = size(x)
    h = n ÷ 2
    out = Matrix{Float64}(undef, h, 2m)
    for j in 1:m
        out[:, 2j - 1] = x[1:h, j]
        out[:, 2j] = x[(n - h + 1):n, j]
    end
    return out
end

"""
    rank_normalize(x)

Rank-normalise a `draws x chains` matrix: average ranks across all draws, mapped
through the inverse normal cdf with the Blom offset 3/8. Makes R-hat and ESS
well defined for heavy-tailed targets such as the funnel.
"""
function rank_normalize(x::AbstractMatrix{<:Real})
    v = vec(x)
    N = length(v)
    perm = sortperm(v)
    r = Vector{Float64}(undef, N)
    i = 1
    while i <= N
        j = i
        while j < N && v[perm[j + 1]] == v[perm[i]]
            j += 1
        end
        avg = (i + j) / 2                     # average rank for ties
        for k in i:j
            r[perm[k]] = avg
        end
        i = j + 1
    end
    z = @. norminvcdf((r - 3 / 8) / (N + 1 / 4))
    return reshape(z, size(x))
end

"""
    norminvcdf(p)

Inverse standard normal cdf via Acklam's rational approximation plus one
Halley refinement, accurate to about 1e-15 over the range used here.
"""
function norminvcdf(p::Real)
    p <= 0 && return -Inf
    p >= 1 && return Inf
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow, phigh = 0.02425, 1 - 0.02425
    local x
    if p < plow
        q = sqrt(-2 * log(p))
        x = (((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]) /
            ((((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1)
    elseif p <= phigh
        q = p - 0.5
        r = q * q
        x = (((((a[1] * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * r + a[6]) * q /
            (((((b[1] * r + b[2]) * r + b[3]) * r + b[4]) * r + b[5]) * r + 1)
    else
        q = sqrt(-2 * log1p(-p))
        x = -(((((c[1] * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) * q + c[6]) /
             ((((d[1] * q + d[2]) * q + d[3]) * q + d[4]) * q + 1)
    end
    e = 0.5 * erfc(-x / sqrt(2)) - p          # Halley refinement
    u = e * sqrt(2pi) * exp(x * x / 2)
    return x - u / (1 + x * u / 2)
end

"""
    rhat_plain(x)

Split R-hat on the raw scale for a `draws x chains` matrix. Exposed mainly so
tests can compare it against the rank-normalised version.
"""
function rhat_plain(x::AbstractMatrix{<:Real})
    s = split_chains(x)
    n, m = size(s)
    (n < 2 || m < 2) && return NaN
    means = [Statistics.mean(view(s, :, j)) for j in 1:m]
    vars = [Statistics.var(view(s, :, j)) for j in 1:m]
    W = Statistics.mean(vars)
    B = n * Statistics.var(means)
    W <= 0 && return NaN
    varplus = ((n - 1) * W + B) / n
    return sqrt(varplus / W)
end

"""
    rhat(x)

Rank-normalised split R-hat: the maximum of R-hat on rank-normalised draws and
on folded (absolute deviation from the median) rank-normalised draws. Values
above 1.01 are the usual signal to keep sampling.
"""
function rhat(x::AbstractMatrix{<:Real})
    r1 = rhat_plain(rank_normalize(x))
    folded = abs.(x .- Statistics.median(vec(x)))
    r2 = rhat_plain(rank_normalize(folded))
    return max(r1, r2)
end
rhat(c::Chains, name::Symbol) = rhat(c[name])

"""
    ess(x)

Effective sample size of a `draws x chains` matrix, using the multi-chain
autocorrelation estimate combined with Geyer's initial monotone positive
sequence truncation.
"""
function ess(x::AbstractMatrix{<:Real})
    s = split_chains(x)
    n, m = size(s)
    (n < 4) && return NaN
    means = [Statistics.mean(view(s, :, j)) for j in 1:m]
    vars = [Statistics.var(view(s, :, j)) for j in 1:m]
    W = Statistics.mean(vars)
    W <= 0 && return float(n * m)                 # constant parameter
    B = m > 1 ? n * Statistics.var(means) : 0.0
    varplus = m > 1 ? ((n - 1) * W + B) / n : W
    maxlag = min(n - 1, 1000)
    acovs = [autocov(view(s, :, j), maxlag) for j in 1:m]
    rho = Vector{Float64}(undef, maxlag + 1)
    @inbounds for t in 0:maxlag
        meanacov = Statistics.mean(a[t + 1] for a in acovs)
        rho[t + 1] = 1 - (W - meanacov) / varplus
    end
    # Geyer's initial positive sequence: pair consecutive autocorrelations,
    # truncate at the first non-positive pair, then force the sequence to be
    # monotone decreasing. Pairing is what makes the truncation reliable - the
    # sum of two consecutive autocorrelations of a reversible chain is positive.
    pairs = Float64[]
    t = 0
    while 2t + 1 <= maxlag
        p = rho[2t + 1] + rho[2t + 2]
        p <= 0 && break
        push!(pairs, p)
        t += 1
    end
    for k in 2:length(pairs)
        pairs[k] = min(pairs[k], pairs[k - 1])    # initial monotone sequence
    end
    tau = -1 + 2 * sum(pairs; init = 0.0)
    tau = max(tau, 1 / log10(max(n * m, 11)))     # Stan's cap on the reported ESS
    return n * m / tau
end
ess(c::Chains, name::Symbol) = ess(c[name])

"""
    ess_bulk(x)

ESS of the rank-normalised draws: the number that governs the accuracy of
posterior means.
"""
ess_bulk(x::AbstractMatrix{<:Real}) = ess(rank_normalize(x))
ess_bulk(c::Chains, name::Symbol) = ess_bulk(c[name])

"""
    ess_tail(x)

The smaller of the ESS of the 5% and 95% tail indicator sequences: the number
that governs the accuracy of extreme quantiles.
"""
function ess_tail(x::AbstractMatrix{<:Real})
    v = vec(x)
    q05 = Statistics.quantile(v, 0.05)
    q95 = Statistics.quantile(v, 0.95)
    lo = ess(rank_normalize(Float64.(x .<= q05)))
    hi = ess(rank_normalize(Float64.(x .<= q95)))
    return min(lo, hi)
end
ess_tail(c::Chains, name::Symbol) = ess_tail(c[name])

"""
    mcse_mean(x)

Monte Carlo standard error of the posterior mean, `sd / sqrt(ess_bulk)`.
"""
mcse_mean(x::AbstractMatrix{<:Real}) = Statistics.std(vec(x)) / sqrt(max(ess_bulk(x), 1.0))
mcse_mean(c::Chains, name::Symbol) = mcse_mean(c[name])

"""
    bfmi(chains)

Bayesian fraction of missing information per chain, from the energy statistic of
a Hamiltonian sampler. Values below about 0.3 mean the momentum resampling is
not refreshing the energy fast enough, typically because of a badly scaled mass
matrix.
"""
function bfmi(c::Chains)
    haskey(c.stats, :energy) || return Float64[]
    E = c.stats[:energy]
    return [begin
                e = view(E, :, j)
                sum(diff(collect(e)) .^ 2) / (length(e) * Statistics.var(e))
            end for j in 1:size(E, 2)]
end
