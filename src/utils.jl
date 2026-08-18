# Numerically careful scalar helpers.
#
# These are deliberately hand-written rather than pulled from LogExpFunctions: the
# whole library is meant to be readable end to end, and these four functions are
# where naive implementations lose precision in the tails.

"""
    logistic(y)

Inverse logit, `1 / (1 + exp(-y))`, evaluated in the numerically stable branch
for each sign of `y`.
"""
function logistic(y::Real)
    if y >= zero(y)
        z = exp(-y)
        return one(y) / (one(y) + z)
    else
        z = exp(y)
        return z / (one(y) + z)
    end
end

"""
    logit(x)

Inverse of [`logistic`](@ref); returns `log(x / (1 - x))`.
"""
logit(x::Real) = log(x) - log1p(-x)

"""
    log1pexp(y)

`log(1 + exp(y))` without overflow. Branch points follow Mächler (2012).
"""
function log1pexp(y::Real)
    if y < -37
        return exp(y)
    elseif y <= 18
        return log1p(exp(y))
    elseif y <= 33
        return y + exp(-y)
    else
        return float(y)
    end
end

"""
    loglogistic(y)

`log(logistic(y)) = -log1pexp(-y)`, kept separate because it is the term that
appears in every logit-transform Jacobian.
"""
loglogistic(y::Real) = -log1pexp(-y)

"""
    logsumexp(xs)

Stable `log(sum(exp, xs))`. Returns `-Inf` for an all `-Inf` input rather than
`NaN`, which matters for importance weights that have all collapsed.
"""
function logsumexp(xs)
    m = -Inf
    for x in xs
        x > m && (m = x)
    end
    isfinite(m) || return m == -Inf ? -Inf : m
    s = zero(float(m))
    for x in xs
        s += exp(x - m)
    end
    return m + log(s)
end

"""
    softmax!(w, logw)

Write `exp.(logw .- logsumexp(logw))` into `w` and return the normalising
constant `logsumexp(logw)`.
"""
function softmax!(w::AbstractVector, logw::AbstractVector)
    lse = logsumexp(logw)
    if isfinite(lse)
        @inbounds for i in eachindex(w, logw)
            w[i] = exp(logw[i] - lse)
        end
    else
        fill!(w, one(eltype(w)) / length(w))
    end
    return lse
end
