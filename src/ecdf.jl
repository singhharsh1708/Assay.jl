# Uniformity testing for calibration ranks, done properly.
#
# Binning ranks into a chi-square test throws away the ordering of the bins,
# which is exactly the information that says *how* a sampler is miscalibrated:
# a slope means a location bias, a U means over-dispersion, an inverted U means
# under-dispersion. It also makes the answer depend on a bin count nobody has a
# principled way to choose.
#
# The alternative, following Säilynoja, Bürkner and Vehtari (2022), is to
# compare the empirical distribution function of the ranks against the uniform
# one, and to judge it against a band that holds *simultaneously* across the
# whole curve rather than pointwise. A pointwise 95% band is crossed by a
# correct sampler almost every time, because it is being checked at a hundred
# places at once; the simultaneous band is the one whose 5% means 5%.
#
# The coverage of a candidate band is computed exactly rather than simulated.
# The empirical distribution function of N uniform draws, read at an increasing
# grid, is a Markov chain: given that k of them fall below z, the number falling
# below the next grid point is k plus a binomial draw from those remaining. That
# makes the probability of staying inside the band a dynamic programme over the
# grid, and the band itself a one dimensional search.

"""
    binomial_quantile(n, p, q)

Smallest `k` with `P(Binomial(n, p) <= k) >= q`, by summation. Exact, and fast
enough for the sample sizes calibration uses.
"""
function binomial_quantile(n::Int, p::Real, q::Real)
    q <= 0 && return 0
    q >= 1 && return n
    acc = 0.0
    for k in 0:n
        acc += exp(logpdf(Binomial(n, p), k))
        acc >= q - 1e-12 && return k
    end
    return n
end

"""
    ecdf_band_coverage(n_sims, grid, lower, upper)

Probability that the empirical distribution function of `n_sims` uniform draws
stays within `lower` and `upper` at every point of `grid`.

Computed by dynamic programming over the grid. The state is the number of draws
seen so far; the transition from one grid point to the next is binomial over the
draws not yet seen, with probability equal to the conditional width of the
interval.
"""
function ecdf_band_coverage(n_sims::Int, grid::AbstractVector, lower::AbstractVector,
                            upper::AbstractVector)
    K = length(grid)
    K == length(lower) == length(upper) ||
        throw(DimensionMismatch("grid and bounds must have the same length"))

    # probability of each count at the first grid point, restricted to the band
    prob = zeros(n_sims + 1)
    z1 = grid[1]
    for k in Int(max(lower[1], 0)):Int(min(upper[1], n_sims))
        prob[k + 1] = exp(logpdf(Binomial(n_sims, z1), k))
    end

    for j in 2:K
        remaining = 1 - grid[j - 1]
        step = remaining <= 0 ? 0.0 : (grid[j] - grid[j - 1]) / remaining
        step = clamp(step, 0.0, 1.0)
        nextprob = zeros(n_sims + 1)
        lo = Int(max(lower[j], 0))
        hi = Int(min(upper[j], n_sims))
        for k in 0:n_sims
            pk = prob[k + 1]
            pk == 0 && continue
            left = n_sims - k
            for extra in 0:left
                target = k + extra
                target < lo && continue
                target > hi && break
                nextprob[target + 1] += pk * exp(logpdf(Binomial(left, step), extra))
            end
        end
        prob = nextprob
    end
    return sum(prob)
end

"""
    ecdf_simultaneous_band(n_sims; n_grid = 100, confidence = 0.95)

A band that a correct sampler's rank distribution function stays inside with
probability `confidence`, across the whole curve at once.

Returns `(grid, lower, upper)` as proportions. The band is found by searching
the pointwise level whose simultaneous coverage equals the target, with
coverage evaluated exactly by [`ecdf_band_coverage`](@ref).
"""
function ecdf_simultaneous_band(n_sims::Int; n_grid::Int = 100, confidence::Real = 0.95)
    n_sims >= 10 || throw(ArgumentError("need at least 10 replications, got $n_sims"))
    0 < confidence < 1 || throw(DomainError(confidence, "confidence must lie in (0, 1)"))
    grid = collect(range(1 / (n_grid + 1), n_grid / (n_grid + 1); length = n_grid))

    bounds = function (gamma)
        lower = [binomial_quantile(n_sims, z, gamma / 2) - 1 for z in grid]
        upper = [binomial_quantile(n_sims, z, 1 - gamma / 2) for z in grid]
        return lower, upper
    end

    # the pointwise level is always tighter than the simultaneous one, so the
    # search runs from it downwards
    lo, hi = 1e-6, 1 - confidence
    local lower, upper
    for _ in 1:40
        gamma = (lo + hi) / 2
        lower, upper = bounds(gamma)
        if ecdf_band_coverage(n_sims, grid, lower, upper) >= confidence
            lo = gamma
        else
            hi = gamma
        end
        hi - lo < 1e-9 && break
    end
    lower, upper = bounds(lo)
    return grid, lower ./ n_sims, upper ./ n_sims
end

"""
    rank_ecdf(ranks, n_draws; n_grid = 100)

Empirical distribution function of calibration ranks, scaled to `(0, 1)` and
evaluated on the same grid [`ecdf_simultaneous_band`](@ref) uses.
"""
function rank_ecdf(ranks, n_draws::Int; n_grid::Int = 100)
    u = [(r + 0.5) / (n_draws + 1) for r in ranks]
    grid = collect(range(1 / (n_grid + 1), n_grid / (n_grid + 1); length = n_grid))
    return grid, [count(<=(z), u) / length(u) for z in grid]
end

"""
    uniformity_ecdf(u; n_grid = 100, confidence = 0.95)

Test values on `(0, 1)` for uniformity by their distribution function, returning
`(inside, max_deviation, grid, ecdf, lower, upper)`.

`inside` is whether the whole curve stayed within the simultaneous band, which
is the decision; the rest is what a plot needs. Unlike a binned chi-square this
uses the ordering, so it responds to a slope, and it needs no bin count to be
chosen.

Calibration ranks are one source of such values and leave-one-out probability
integral transforms are another, so the test lives here rather than inside
either.
"""
function uniformity_ecdf(u::AbstractVector{<:Real}; n_grid::Int = 100, confidence::Real = 0.95)
    grid = collect(range(1 / (n_grid + 1), n_grid / (n_grid + 1); length = n_grid))
    curve = [count(<=(z), u) / length(u) for z in grid]
    _, lower, upper = ecdf_simultaneous_band(length(u); n_grid = n_grid, confidence = confidence)
    inside = all(lower .<= curve .<= upper)
    deviation = maximum(abs.(curve .- grid))
    return (inside = inside, max_deviation = deviation, grid = grid, ecdf = curve,
            lower = lower, upper = upper)
end

"""
    rank_uniformity_ecdf(ranks, n_draws; n_grid = 100, confidence = 0.95)

[`uniformity_ecdf`](@ref) applied to calibration ranks, scaled to `(0, 1)` the
same way [`rank_ecdf`](@ref) does.
"""
function rank_uniformity_ecdf(ranks, n_draws::Int; n_grid::Int = 100, confidence::Real = 0.95)
    u = [(r + 0.5) / (n_draws + 1) for r in ranks]
    return uniformity_ecdf(u; n_grid = n_grid, confidence = confidence)
end
