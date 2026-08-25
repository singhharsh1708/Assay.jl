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

# --------------------------------------------------------------------------
# A radix-2 fast Fourier transform
#
# Autocovariances are what effective sample size is built on, and computing
# them directly costs O(n * lag). Through the Wiener-Khinchin theorem an FFT
# turns that into O(n log n) and, more usefully, makes it affordable to use
# every lag rather than truncating at an arbitrary maximum.
#
# This is here rather than as a dependency for the same reason the densities
# are: the point of the package is that every number traces to code in it. It
# is checked against a direct discrete Fourier transform in the test suite.
# --------------------------------------------------------------------------

"""
    next_power_of_two(n)

Smallest power of two at least `n`.
"""
next_power_of_two(n::Integer) = n <= 1 ? 1 : 2^(ceil(Int, log2(n)))

"""
    fft!(x)

In-place radix-2 Cooley-Tukey transform of a complex vector whose length is a
power of two.

The twiddle factors are evaluated directly as `cis(theta * k)` rather than
accumulated by repeated multiplication. Accumulation is faster and drifts:
the error grows with the transform length, which is exactly the regime a long
chain puts it in.
"""
function fft!(x::Vector{ComplexF64})
    n = length(x)
    ispow2(n) || throw(ArgumentError("fft! needs a power-of-two length, got $n"))
    n == 1 && return x

    # bit-reversal permutation
    j = 0
    @inbounds for i in 0:(n - 2)
        if i < j
            x[i + 1], x[j + 1] = x[j + 1], x[i + 1]
        end
        m = n >> 1
        while m >= 1 && j >= m
            j -= m
            m >>= 1
        end
        j += m
    end

    len = 2
    while len <= n
        half = len >> 1
        theta = -2pi / len
        @inbounds for start in 1:len:n
            for k in 0:(half - 1)
                w = cis(theta * k)
                u = x[start + k]
                v = x[start + k + half] * w
                x[start + k] = u + v
                x[start + k + half] = u - v
            end
        end
        len <<= 1
    end
    return x
end

"""
    ifft!(x)

Inverse of [`fft!`](@ref), normalised by the transform length.
"""
function ifft!(x::Vector{ComplexF64})
    n = length(x)
    @inbounds for i in eachindex(x)
        x[i] = conj(x[i])
    end
    fft!(x)
    @inbounds for i in eachindex(x)
        x[i] = conj(x[i]) / n
    end
    return x
end
