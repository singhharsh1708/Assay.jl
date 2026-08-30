# Densities built out of other densities: mixtures, truncation and censoring.
#
# All three are ordinary in applied work and all three are places where the
# arithmetic is the whole difficulty. A mixture written the obvious way
# overflows. Truncation needs a normalising constant that a user has to look up
# and get right, and getting it wrong is silent: the posterior is biased and
# nothing raises. Censoring needs the tail probability of the observation, which
# is the one quantity a naive `log(1 - cdf(x))` computes worst.
#
# Those are exactly the details a library should own.

# --------------------------------------------------------------------------
# Tail probabilities
# --------------------------------------------------------------------------

"""
    ccdf(d, x)

Complement of the cdf, `P(X > x)`.

Written separately rather than as `1 - cdf(d, x)` because that subtraction is
where the accuracy goes. At four standard deviations into the upper tail of a
normal, `1 - cdf` has about eleven correct digits; by six it has eight, and it
reaches zero at eight while the true value is 6e-16. Censored data lives in that
tail by construction, so the complement is computed directly wherever the
special function allows it.
"""
ccdf(d::UnivariateDensity, x::Real) = 1 - cdf(d, x)
ccdf(d::Normal, x::Real) = 0.5 * erfc((x - d.mu) / (d.sigma * sqrt(2)))
ccdf(d::LogNormal, x::Real) = x <= 0 ? 1.0 : ccdf(Normal(d.mu, d.sigma), log(x))
ccdf(d::Exponential, x::Real) = x <= 0 ? 1.0 : exp(-d.rate * x)
ccdf(d::Gamma, x::Real) = x <= 0 ? 1.0 : gamma_inc(d.shape, d.rate * x, 0)[2]
ccdf(d::Uniform, x::Real) = clamp((d.b - x) / (d.b - d.a), 0, 1)

"""
    logcdf(d, x)

Log of the cdf. Falls back to `log(cdf(d, x))`.
"""
logcdf(d::UnivariateDensity, x::Real) = log(cdf(d, x))

"""
    logccdf(d, x)

Log of the complementary cdf, which is the log survival function: what a
right-censored observation contributes to a likelihood.
"""
logccdf(d::UnivariateDensity, x::Real) = log(ccdf(d, x))
logccdf(d::Exponential, x::Real) = x <= 0 ? 0.0 : -d.rate * x

# --------------------------------------------------------------------------
# Mixtures
# --------------------------------------------------------------------------

"""
    MixtureDensity(weights, components)

Mixture of univariate densities, evaluated through `logsumexp`.

The naive form, `log(sum(w .* pdf.(components, x)))`, is the reason this exists.
Two components a few hundred standard deviations apart make every term but one
underflow to zero, and the log of zero is the answer the user gets. Working in
log space throughout, the same evaluation is exact to the last digit.

`weights` must be non-negative and sum to one, which is what
[`simplex`](@ref) produces, so a mixture whose weights are being inferred
composes with the transform layer rather than needing its own.

    MixtureDensity([0.3, 0.7], [Normal(0.0, 1.0), Normal(500.0, 1.0)])
"""
struct MixtureDensity{C<:UnivariateDensity,W<:AbstractVector{<:Real}} <: UnivariateDensity
    weights::W
    components::Vector{C}

    function MixtureDensity(weights::W, components::AbstractVector{C}) where {C<:UnivariateDensity,
                                                                             W<:AbstractVector{<:Real}}
        length(weights) == length(components) ||
            throw(DimensionMismatch("got $(length(weights)) weights for " *
                                    "$(length(components)) components"))
        isempty(components) && throw(ArgumentError("a mixture needs at least one component"))
        all(>=(0), weights) || throw(DomainError(weights, "mixture weights must be non-negative"))
        s = sum(weights)
        isapprox(s, 1; atol = 1e-8) ||
            throw(DomainError(s, "mixture weights must sum to one, got $s"))
        return new{C,W}(weights, collect(components))
    end
end

n_components(d::MixtureDensity) = length(d.components)

function logpdf(d::MixtureDensity, x::Real)
    terms = [log(w) + logpdf(c, x) for (w, c) in zip(d.weights, d.components)]
    return logsumexp(terms)
end

function Base.rand(rng::AbstractRNG, d::MixtureDensity)
    u = rand(rng)
    acc = zero(eltype(d.weights))
    for (w, c) in zip(d.weights, d.components)
        acc += w
        u <= acc && return rand(rng, c)
    end
    return rand(rng, last(d.components))       # only reachable through rounding
end

mean(d::MixtureDensity) = sum(w * mean(c) for (w, c) in zip(d.weights, d.components))

# Law of total variance: the mixture is not the average of the component
# variances, it is that plus the spread of the component means. Getting this
# wrong understates the variance of a well separated mixture by orders of
# magnitude.
function var(d::MixtureDensity)
    m = mean(d)
    within = sum(w * var(c) for (w, c) in zip(d.weights, d.components))
    between = sum(w * (mean(c) - m)^2 for (w, c) in zip(d.weights, d.components))
    return within + between
end

cdf(d::MixtureDensity, x::Real) = sum(w * cdf(c, x) for (w, c) in zip(d.weights, d.components))
ccdf(d::MixtureDensity, x::Real) = sum(w * ccdf(c, x) for (w, c) in zip(d.weights, d.components))

function _support(d::MixtureDensity)
    los = [_support(c)[1] for c in d.components]
    his = [_support(c)[2] for c in d.components]
    return minimum(los), maximum(his)
end

# --------------------------------------------------------------------------
# Truncation
# --------------------------------------------------------------------------

"""
    Truncated(d, lower, upper)

`d` restricted to `[lower, upper]` and renormalised. Either bound may be
infinite.

The normalising constant is the mass `d` puts on the interval, and computing it
as `cdf(upper) - cdf(lower)` is only accurate when that difference is not a
difference of two nearly equal numbers. A one-sided truncation far into a tail
is exactly that case, so it is computed from the complementary cdf instead, and
the two-sided case is computed from whichever end is nearer.

A user who writes the constant by hand and gets it wrong sees nothing: the
posterior is biased and no diagnostic fires. That is the reason this is here
rather than in a docstring somewhere telling them how.
"""
struct Truncated{D<:UnivariateDensity,L<:Real,U<:Real} <: UnivariateDensity
    d::D
    lower::L
    upper::U
    logmass::Float64

    function Truncated(d::D, lower::L, upper::U) where {D<:UnivariateDensity,L<:Real,U<:Real}
        lower < upper ||
            throw(ArgumentError("need lower < upper, got lower = $lower and upper = $upper"))
        lm = _log_interval_mass(d, lower, upper)
        isfinite(lm) ||
            throw(ArgumentError("the truncation interval [$lower, $upper] carries no mass " *
                                "under $(typeof(d).name.name), so there is nothing to " *
                                "renormalise onto"))
        return new{D,L,U}(d, lower, upper, lm)
    end
end

Truncated(d::UnivariateDensity; lower::Real = -Inf, upper::Real = Inf) = Truncated(d, lower, upper)

"""
    truncated(d, lower, upper)

Convenience constructor for [`Truncated`](@ref).
"""
truncated(d::UnivariateDensity, lower::Real = -Inf, upper::Real = Inf) =
    Truncated(d, lower, upper)

# Which end to compute from is the whole numerical question. Working from the
# far tail of a cdf is what loses the digits.
function _log_interval_mass(d::UnivariateDensity, lower::Real, upper::Real)
    if !isfinite(lower) && !isfinite(upper)
        return 0.0
    elseif !isfinite(lower)
        return logcdf(d, upper)
    elseif !isfinite(upper)
        return logccdf(d, lower)
    end
    # Two-sided: subtract at whichever end the survivors are, so the subtraction
    # is between numbers that differ rather than between two numbers near one.
    if cdf(d, lower) < 0.5
        return log(cdf(d, upper) - cdf(d, lower))
    else
        return log(ccdf(d, lower) - ccdf(d, upper))
    end
end

function logpdf(t::Truncated, x::Real)
    (t.lower <= x <= t.upper) || return oftype(float(logpdf(t.d, x)), -Inf)
    return logpdf(t.d, x) - t.logmass
end

function cdf(t::Truncated, x::Real)
    x <= t.lower && return 0.0
    x >= t.upper && return 1.0
    lo = isfinite(t.lower) ? cdf(t.d, t.lower) : 0.0
    return (cdf(t.d, x) - lo) / exp(t.logmass)
end

ccdf(t::Truncated, x::Real) = 1 - cdf(t, x)
_support(t::Truncated) = (float(t.lower), float(t.upper))

# Inverse cdf sampling: exact, and unlike rejection it does not become slow as
# the interval gets narrow, which is when truncation is most often used.
function Base.rand(rng::AbstractRNG, t::Truncated)
    lo = isfinite(t.lower) ? cdf(t.d, t.lower) : 0.0
    hi = isfinite(t.upper) ? cdf(t.d, t.upper) : 1.0
    return quantile(t.d, lo + rand(rng) * (hi - lo))
end

# --------------------------------------------------------------------------
# Censoring
# --------------------------------------------------------------------------

"""
    Censored(d, lower, upper)

`d` with everything below `lower` reported as `lower` and everything above
`upper` reported as `upper`.

Truncation and censoring are different and the difference matters. A truncated
observation is one that could not have been recorded outside the interval; a
censored one was recorded, and all that is known is which side of the limit it
fell. A detection limit is censoring, and a subject still alive at the end of a
study is censoring. Treating either as truncation throws away the observation
and biases the answer.

So a value at the limit contributes the tail probability rather than a density:
`log P(X <= lower)` at the lower limit and `log P(X >= upper)` at the upper one.
Those are a probability, not a density, which is why the result is not a density
with respect to a single measure and integrates to one only in the mixed sense.
"""
struct Censored{D<:UnivariateDensity,L<:Real,U<:Real} <: UnivariateDensity
    d::D
    lower::L
    upper::U

    function Censored(d::D, lower::L, upper::U) where {D<:UnivariateDensity,L<:Real,U<:Real}
        lower < upper ||
            throw(ArgumentError("need lower < upper, got lower = $lower and upper = $upper"))
        return new{D,L,U}(d, lower, upper)
    end
end

Censored(d::UnivariateDensity; lower::Real = -Inf, upper::Real = Inf) = Censored(d, lower, upper)

"""
    censored(d, lower, upper)

Convenience constructor for [`Censored`](@ref).
"""
censored(d::UnivariateDensity, lower::Real = -Inf, upper::Real = Inf) = Censored(d, lower, upper)

function logpdf(c::Censored, x::Real)
    x < c.lower && return oftype(float(logpdf(c.d, x)), -Inf)
    x > c.upper && return oftype(float(logpdf(c.d, x)), -Inf)
    x == c.lower && isfinite(c.lower) && return logcdf(c.d, c.lower)
    x == c.upper && isfinite(c.upper) && return logccdf(c.d, c.upper)
    return logpdf(c.d, x)
end

Base.rand(rng::AbstractRNG, c::Censored) = clamp(rand(rng, c.d), c.lower, c.upper)
cdf(c::Censored, x::Real) = x < c.lower ? 0.0 : (x >= c.upper ? 1.0 : cdf(c.d, x))
_support(c::Censored) = (float(c.lower), float(c.upper))
