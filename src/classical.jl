# The convergence diagnostics people reported before rank-normalised split
# R-hat, implemented because published analyses are full of them.
#
# None of these is the one to reach for. Rank-normalised split R-hat with bulk
# and tail effective sample size, already in `diagnostics.jl`, dominates all of
# them: it is invariant to reparameterisation, it notices a difference in
# variance and not only in location, and it does not need the chain to be
# approximately normal. These are here so that a reader holding a paper that
# reports a Raftery-Lewis dependence factor can compute the same number, and so
# that the comparison between the old diagnostics and the new one can be made in
# this repository rather than asserted.
#
# `discretediag` is deliberately not here. It tests chains whose values are
# categorical, and every sampler in this package works in a continuous
# unconstrained space, so no chain any of them produces is one it applies to.
# Discrete parameters are open as their own piece of work.
#
# A naming collision worth stating plainly, because it has caught people. There
# are two things called the Geweke diagnostic. The one here, [`geweke_z`](@ref),
# is the 1992 convergence check: it compares the start of one chain against its
# end. The one in `calibration.jl`, [`geweke`](@ref), is the 2004 joint
# distribution test, which checks a sampler against the model that generated its
# data and can fail on a perfectly converged chain. They share an author and
# nothing else.

"""
    gelman_rubin(x; alpha = 0.05)

The original potential scale reduction factor, with the degrees of freedom
correction of Brooks and Gelman (1998), from a `draws x chains` matrix.

Returns `(psrf, upper)`, the point estimate and the upper end of a `1 - alpha`
interval for it.

This is not [`rhat`](@ref). It splits nothing, it ranks nothing, and it assumes
the target is roughly normal, so it is blind to a chain that has the right mean
and the wrong variance, and it can be fooled by a heavy tail. It is reported
here because it is what a paper from before 2019 will have quoted.
"""
function gelman_rubin(x::AbstractMatrix{<:Real}; alpha::Real = 0.05)
    n, m = size(x)
    m >= 2 || throw(ArgumentError("need at least 2 chains, got $m"))
    n >= 2 || throw(ArgumentError("need at least 2 draws per chain, got $n"))

    means = [Statistics.mean(view(x, :, j)) for j in 1:m]
    vars = [Statistics.var(view(x, :, j)) for j in 1:m]
    W = Statistics.mean(vars)
    B = n * Statistics.var(means)
    W > 0 || return (psrf = 1.0, upper = 1.0)

    sigma2 = (n - 1) / n * W + B / n
    V = sigma2 + B / (m * n)

    # Variance of V-hat, from which the degrees of freedom come. The three terms
    # are the within-chain variance of the variances, the variance of B, and the
    # covariance between them, and dropping the third is a common error that
    # makes the interval too narrow.
    grand = Statistics.mean(means)
    var_W = Statistics.var(vars) / m
    var_B = 2 * B^2 / (m - 1)
    cov_term = (n / m) * (Statistics.cov(vars, means .^ 2) -
                          2 * grand * Statistics.cov(vars, means))
    var_V = ((n - 1) / n)^2 * var_W + ((m + 1) / (m * n))^2 * var_B +
            2 * (m + 1) * (n - 1) / (m * n^2) * cov_term
    df = var_V > 0 ? 2 * V^2 / var_V : Inf

    psrf = sqrt(V / W * (df + 3) / (df + 1))
    # Upper bound through the F approximation on the ratio of B to W
    df1, df2 = m - 1, 2 * W^2 * m / (Statistics.var(vars) > 0 ? Statistics.var(vars) : eps())
    fq = _f_quantile(df1, df2, 1 - alpha / 2)
    upper = sqrt(((n - 1) / n + (m + 1) / (m * n) * (B / n) / W * fq) * (df + 3) / (df + 1))
    return (psrf = psrf, upper = max(upper, psrf))
end

# Quantile of an F distribution by bisection on the incomplete beta function,
# which is the only place this package needs one.
function _f_quantile(d1::Real, d2::Real, p::Real)
    isfinite(d2) || return _chisq_quantile(d1, p) / d1
    f(x) = beta_inc(d1 / 2, d2 / 2, d1 * x / (d1 * x + d2))[1]
    lo, hi = 0.0, 1.0
    while f(hi) < p && hi < 1e12
        hi *= 2
    end
    for _ in 1:200
        mid = (lo + hi) / 2
        f(mid) < p ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end

_chisq_quantile(k::Real, p::Real) = 2 * quantile(Gamma(k / 2, 1.0), p)

"""
    geweke_z(x; first = 0.1, last = 0.5)

Geweke's 1992 convergence diagnostic: a z score comparing the mean of the first
`first` of a chain against the mean of its last `last`.

Large values say the chain had not settled by the point the first window covers.
The two windows are compared using each one's own Monte Carlo standard error, so
the score accounts for autocorrelation rather than treating the draws as
independent.

This is the 1992 diagnostic, not the 2004 joint distribution test in
[`geweke`](@ref). The names collide in the literature and the tests answer
different questions.
"""
function geweke_z(x::AbstractVector{<:Real}; first::Real = 0.1, last::Real = 0.5)
    0 < first < 1 && 0 < last < 1 && first + last <= 1 ||
        throw(ArgumentError("windows must be positive fractions summing to at most one, " *
                            "got $first and $last"))
    n = length(x)
    n1 = max(round(Int, first * n), 2)
    n2 = max(round(Int, last * n), 2)
    a = view(x, 1:n1)
    b = view(x, (n - n2 + 1):n)
    # Each window gets its own spectral estimate, which is the whole point: if
    # the chain is still moving, the two windows have different correlation
    # structure as well as different means.
    se = sqrt(spectrum0_ar(a) / n1 + spectrum0_ar(b) / n2)
    se > 0 || return 0.0
    return (Statistics.mean(a) - Statistics.mean(b)) / se
end

geweke_z(x::AbstractMatrix{<:Real}; kwargs...) =
    [geweke_z(view(x, :, j); kwargs...) for j in 1:size(x, 2)]

"""
    spectrum0_ar(x; max_order = nothing)

Spectral density of `x` at frequency zero, from an autoregressive fit whose
order is chosen by AIC.

Both this and `n * mcse_mean(x)^2` estimate the same quantity for a stationary
chain, and they behave differently on one that is not. A drifting chain has a
collapsed effective sample size, so a spectral density built from the Monte
Carlo error is enormous, and any statistic that divides by it stops being able
to see the drift. That is a real trap: it is why [`heidelberger_welch`](@ref)
estimates its denominator once from the second half of the chain rather than
from the window it is testing.
"""
function spectrum0_ar(x::AbstractVector{<:Real}; max_order::Union{Nothing,Int} = nothing)
    n = length(x)
    n >= 4 || return Statistics.var(x)
    pmax = max_order === nothing ? min(n - 1, floor(Int, 10 * log10(n))) : max_order
    pmax = max(min(pmax, n ÷ 2), 1)
    r = autocov(collect(float.(x)), pmax)
    r[1] > 0 || return 0.0

    best_aic, best_spec = Inf, r[1]
    for p in 0:pmax
        if p == 0
            aic = n * log(r[1])
            aic < best_aic && ((best_aic, best_spec) = (aic, r[1]))
            continue
        end
        R = [r[abs(i - j) + 1] for i in 1:p, j in 1:p]
        rhs = r[2:(p + 1)]
        phi = try
            Symmetric(R) \ rhs
        catch
            continue
        end
        all(isfinite, phi) || continue
        sigma2 = r[1] - sum(phi .* rhs)
        sigma2 > 0 || continue
        denom = (1 - sum(phi))^2
        denom > 1e-12 || continue
        aic = n * log(sigma2) + 2p
        if aic < best_aic
            best_aic = aic
            best_spec = sigma2 / denom
        end
    end
    return best_spec
end

"""
    HeidelbergerWelch

The result of the Heidelberger-Welch procedure: whether the chain looks
stationary after discarding some of its start, and whether the mean is estimated
precisely enough to report.

The two halves are separate questions. `stationary` says the chain has settled;
`halfwidth_passed` says it has run long enough for the mean to be worth quoting,
which a stationary chain can easily fail.
"""
struct HeidelbergerWelch
    stationary::Bool
    pvalue::Float64
    burn_in::Int
    mean::Float64
    halfwidth::Float64
    halfwidth_passed::Bool
end

function Base.show(io::IO, r::HeidelbergerWelch)
    @printf(io, "HeidelbergerWelch(%s after discarding %d, p = %.3f; mean %.4g ± %.4g, %s)",
            r.stationary ? "stationary" : "NOT stationary", r.burn_in, r.pvalue, r.mean,
            r.halfwidth, r.halfwidth_passed ? "precise enough" : "not precise enough")
end

"""
    heidelberger_welch(x; alpha = 0.05, eps = 0.1)

The Heidelberger-Welch stationarity and halfwidth tests on one chain.

Discards the first tenth of the chain, then the first fifth, and so on up to
half, and applies the Cramer-von Mises test for a Brownian bridge to what is
left, stopping at the first portion that passes. Then checks whether the
halfwidth of a `1 - alpha` interval for the mean is within `eps` of the mean
itself.

Both halves rest on an estimate of the spectral density at zero, and there is no
canonical choice for it. `spectrum0` is that choice, defaulting to
[`spectrum0_ar`](@ref). Implementations that use a different one report different
p values and can disagree about a borderline chain, which is a property of the
diagnostic rather than of any implementation of it.

The halfwidth test is the half worth keeping. It is the only diagnostic in this
file that asks the question a user actually has, which is whether the answer is
precise enough to report, and it is the same question [`mcse_mean`](@ref)
answers directly.
"""
function heidelberger_welch(x::AbstractVector{<:Real}; alpha::Real = 0.05, eps::Real = 0.1,
                            spectrum0 = spectrum0_ar)
    n = length(x)
    n >= 20 || throw(ArgumentError("need at least 20 draws, got $n"))

    # The denominator comes from the second half of the chain and is then held
    # fixed. Estimating it separately for each window would let a window that is
    # still drifting inflate its own denominator and pass, which is exactly the
    # case the test exists to catch.
    # `mcse_mean` is not the estimator to use here. It is built on bulk
    # effective sample size, which is rank-normalised and split, and on a chain
    # that is still drifting the split alone collapses it, so the denominator
    # becomes large enough to hide the drift. The autoregressive estimate works
    # on the raw series and does neither.
    S0 = spectrum0(view(x, (n ÷ 2):n))

    delta = max(trunc(Int, 0.1 * n), 1)
    i = 1
    pvalue = 1.0
    stationary = false
    y = view(x, 1:n)
    while i < n / 2
        y = view(x, i:n)
        pvalue = _cvm_pvalue(_cvm_statistic(y, S0))
        stationary = pvalue > alpha
        stationary && break
        i += delta
    end

    m = Statistics.mean(y)
    half = quantile(Normal(0.0, 1.0), 1 - alpha / 2) * sqrt(spectrum0(y) / length(y))
    return HeidelbergerWelch(stationary, pvalue, i - 1, m, half, half <= eps * abs(m))
end

# Cramer-von Mises statistic on the standardised cumulative sum, which under
# stationarity converges to a Brownian bridge.
function _cvm_statistic(y::AbstractVector{<:Real}, S0::Real)
    S0 > 0 || return 0.0
    n = length(y)
    csum = cumsum(y .- Statistics.mean(y))
    return sum(csum .^ 2) / (n^2 * S0)
end

# Asymptotic distribution of the Cramer-von Mises statistic, by the series of
# Anderson and Darling (1952). Converges in a handful of terms.
function _cvm_pvalue(w::Real)
    w > 0 || return 1.0
    acc = 0.0
    for j in 0:15
        a = (4j + 1)^2 / (16 * w)
        a > 700 && break                                  # the term has underflowed
        term = exp(loggamma(j + 0.5) - loggamma(0.5) - loggamma(j + 1.0)) *
               sqrt(4j + 1) * exp(-a) * besselk(0.25, a)
        isfinite(term) || break
        acc += term
    end
    return clamp(1 - acc / (pi * sqrt(w)), 0.0, 1.0)
end

"""
    RafteryLewis

How long a chain has to be to estimate one quantile to a stated accuracy.

`dependence` is the ratio of the required length to what independent draws would
need. Values above about 5 are the usual signal that the chain is badly mixed,
and are the reason this diagnostic gets quoted at all.
"""
struct RafteryLewis
    burn_in::Int
    total::Int
    minimum::Int
    dependence::Float64
    thin::Int
end

function Base.show(io::IO, r::RafteryLewis)
    @printf(io, "RafteryLewis(burn-in %d, total %d, %d if independent, dependence factor %.2f)",
            r.burn_in, r.total, r.minimum, r.dependence)
end

"""
    raftery_lewis(x; q = 0.025, r = 0.005, s = 0.95, eps = 0.001)

How many draws are needed to estimate the `q` quantile to within `r` with
probability `s`, following Raftery and Lewis (1992).

The chain is reduced to whether each draw is below the `q` quantile, thinned
until that binary chain is closer to first-order Markov than to second-order,
and the two-state transition probabilities are read off. Everything the answer
depends on comes from those two numbers, which is both what makes the method
cheap and why it addresses one quantile rather than the distribution.
"""
function raftery_lewis(x::AbstractVector{<:Real}; q::Real = 0.025, r::Real = 0.005,
                       s::Real = 0.95, eps::Real = 0.001)
    0 < q < 1 || throw(DomainError(q, "q must lie in (0, 1)"))
    n = length(x)
    phi = quantile(Normal(0.0, 1.0), (s + 1) / 2)
    nmin = ceil(Int, phi^2 * q * (1 - q) / r^2)
    n > nmin ÷ 10 ||
        throw(ArgumentError("chain of $n draws is too short to assess a target of $nmin"))

    cutoff = Statistics.quantile(collect(x), q)
    z = x .<= cutoff

    # Thin until the binary chain prefers a first-order Markov model to a
    # second-order one by BIC, which is the test the method is built on.
    k = 1
    while k < n ÷ 10
        thinned = z[1:k:end]
        _bic_prefers_first_order(thinned) && break
        k += 1
    end
    thinned = z[1:k:end]

    a, b = _two_state_transitions(thinned)
    # Both directions have to be observed. A chain that goes below the quantile
    # and never comes back gives a = 0, and the formula then reports needing
    # fewer draws than independent sampling would, which is not a small number,
    # it is a meaningless one.
    (a > 0 && b > 0) ||
        throw(ArgumentError("the indicator chain crosses the $(q) quantile in only one " *
                            "direction (P(0->1) = $a, P(1->0) = $b), so its transition " *
                            "probabilities cannot be estimated. The chain has not mixed " *
                            "enough for this diagnostic to say anything."))
    lambda = 1 - a - b
    burn = if abs(lambda) < 1e-12
        k
    else
        k * max(ceil(Int, log(eps * (a + b) / max(a, b)) / log(abs(lambda))), 1)
    end
    total_draws = k * ceil(Int, (2 - a - b) * a * b / (a + b)^3 * (phi / r)^2)
    total = burn + total_draws
    return RafteryLewis(burn, total, nmin, total / nmin, k)
end

function _two_state_transitions(z::AbstractVector{Bool})
    n01 = n00 = n10 = n11 = 0
    for t in 1:(length(z) - 1)
        if z[t]
            z[t + 1] ? (n11 += 1) : (n10 += 1)
        else
            z[t + 1] ? (n01 += 1) : (n00 += 1)
        end
    end
    a = (n01 + n00) > 0 ? n01 / (n01 + n00) : 0.0        # P(0 -> 1)
    b = (n10 + n11) > 0 ? n10 / (n10 + n11) : 0.0        # P(1 -> 0)
    return a, b
end

# Second-order against first-order by BIC on the three-way transition counts.
function _bic_prefers_first_order(z::AbstractVector{Bool})
    length(z) >= 20 || return true
    counts = zeros(Int, 2, 2, 2)
    for t in 1:(length(z) - 2)
        counts[z[t] + 1, z[t + 1] + 1, z[t + 2] + 1] += 1
    end
    g2 = 0.0
    total = sum(counts)
    for i in 1:2, j in 1:2, k in 1:2
        c = counts[i, j, k]
        c == 0 && continue
        expected = sum(counts[i, j, :]) * sum(counts[:, j, k]) / max(sum(counts[:, j, :]), 1)
        expected > 0 && (g2 += 2 * c * log(c / expected))
    end
    return g2 - 2 * log(total) < 0
end
