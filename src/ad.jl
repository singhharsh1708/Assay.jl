"""
Gradient backends.

Every sampler that needs a gradient asks for it through
[`logdensity_and_gradient`](@ref), never by calling an AD package directly. That
indirection is what makes the backend a user-visible choice, and it is also the
seam the test suite uses to inject a deliberately wrong gradient
(`test/negative_controls.jl`) without touching library code.
"""
abstract type ADBackend end

"""
    ForwardDiffAD()

Forward-mode AD. The default: model dimensions here are small, and forward mode
has no tape, so it composes with the `Dual`-friendly densities without care.
"""
struct ForwardDiffAD <: ADBackend end

"""
    ReverseDiffAD(; compile = false)

Reverse-mode AD, provided by a package extension that loads only if the user has
ReverseDiff available. Worth it once the parameter dimension is in the hundreds.
"""
Base.@kwdef struct ReverseDiffAD <: ADBackend
    compile::Bool = false
end

"""
    FiniteDiffAD(; h = cbrt(eps()))

Central differences. Far too slow and too inaccurate for sampling, but it is the
independent check the gradient tests compare `ForwardDiffAD` against, so it is
part of the library rather than the test suite.
"""
Base.@kwdef struct FiniteDiffAD <: ADBackend
    h::Float64 = cbrt(eps(Float64))
end

"""
    logdensity_and_gradient(backend, f, y)

Return `(f(y), ∇f(y))` for a scalar function `f` on `R^n`.
"""
function logdensity_and_gradient end

function logdensity_and_gradient(::ForwardDiffAD, f, y::AbstractVector)
    result = DiffResults.GradientResult(y)
    result = ForwardDiff.gradient!(result, f, y)
    return DiffResults.value(result), DiffResults.gradient(result)
end

function logdensity_and_gradient(b::FiniteDiffAD, f, y::AbstractVector)
    v = f(y)
    g = similar(y, float(eltype(y)))
    yp = collect(float.(y))
    for i in eachindex(y)
        hi = b.h * max(abs(y[i]), 1.0)
        old = yp[i]
        yp[i] = old + hi
        fp = f(yp)
        yp[i] = old - hi
        fm = f(yp)
        yp[i] = old
        g[i] = (fp - fm) / (2 * hi)
    end
    return v, g
end

function logdensity_and_gradient(::ReverseDiffAD, f, y::AbstractVector)
    error("ReverseDiffAD requires ReverseDiff to be loaded: `using ReverseDiff`.")
end
