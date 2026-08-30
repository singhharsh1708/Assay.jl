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

Reverse-mode automatic differentiation, provided by a package extension that
loads only when the user has ReverseDiff available. Worth it once the parameter
dimension is in the hundreds: forward mode costs one pass per parameter, reverse
mode one pass in total.

With `compile = true` the tape is recorded once and reused, which removes most
of the per-call overhead. That is only valid for a log density whose control
flow does not depend on the parameter values - no branching on `theta`, no early
returns from a support check. Every density in this package returns `-Inf`
outside the support through arithmetic rather than a branch on the parameter,
but a user model that branches must leave `compile` off.

The backend is mutable because the compiled tape is cached in it, so a given
`ReverseDiffAD()` instance belongs to one model.
"""
Base.@kwdef mutable struct ReverseDiffAD <: ADBackend
    compile::Bool = false
    tape::Any = nothing
    tape_length::Int = -1
end

"""
    ZygoteAD()

Reverse-mode automatic differentiation through Zygote, supplied by a package
extension.

Zygote cannot differentiate array mutation, and three transforms here build
their output by writing into an array: `simplex`, `ordered` and
`corr_cholesky`. Models using those need a different backend, and this one says
so rather than failing with a stack trace about `setindex!`. Elementwise
transforms are broadcast and work.
"""
struct ZygoteAD <: ADBackend end

"""
    EnzymeAD()

Reverse-mode automatic differentiation through Enzyme, supplied by a package
extension. Handles every transform in this package, mutation included, and
matches forward mode to machine precision on the mixed model in the test suite.
"""
struct EnzymeAD <: ADBackend end

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

"""
    logdensity_and_gradient(backend::ADBackend, f, y)

Fallback for a backend with no method. `ReverseDiffAD` lands here until
ReverseDiff is loaded, at which point the package extension supplies the real
method. Defining the fallback on the abstract type rather than on
`ReverseDiffAD` matters: a concrete stub would be *overwritten* by the
extension, which Julia refuses to do during precompilation.
"""
function logdensity_and_gradient(b::ADBackend, f, y::AbstractVector)
    b isa ReverseDiffAD && throw(BackendUnavailableError(typeof(b), :ReverseDiff))
    throw(BackendUnavailableError(typeof(b), nothing))
end
