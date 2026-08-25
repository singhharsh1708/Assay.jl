module AssayLogDensityProblemsExt

# Both halves of the LogDensityProblems bridge.
#
#   * an Assay `Model` implements the interface, so any sampler in the Julia
#     ecosystem can run on a model written here;
#   * `LogDensityModel(problem)` wraps anything implementing the interface, so
#     every sampler here can run on a model written elsewhere.
#
# Neither direction requires the other package at load time: this is a package
# extension, and Assay does not depend on LogDensityProblems.

using Assay: Assay, AbstractModel, Model, LogDensityModel, ADBackend, ForwardDiffAD
using LogDensityProblems: LogDensityProblems

# --- Assay model, seen by the rest of the ecosystem ------------------------

LogDensityProblems.capabilities(::Type{<:Model}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(m::Model) = Assay.dimension(m)
LogDensityProblems.logdensity(m::Model, y::AbstractVector) = Assay.logdensity(m, y)

function LogDensityProblems.logdensity_and_gradient(m::Model, y::AbstractVector)
    return Assay.logdensity_and_gradient(m, y)
end

# --- someone else's model, seen by the samplers here -----------------------

"""
    LogDensityModel(problem; backend = ForwardDiffAD(), names = nothing)

Wrap any object implementing the LogDensityProblems interface. If the object
advertises first-order capability its own gradient is used, which matters when
that gradient is exact or compiled; otherwise gradients are taken with
`backend`.
"""
function Assay.LogDensityModel(problem; backend::ADBackend = ForwardDiffAD(), names = nothing)
    d = LogDensityProblems.dimension(problem)
    order = LogDensityProblems.capabilities(problem)
    order === nothing &&
        throw(ArgumentError("$(typeof(problem)) does not implement the LogDensityProblems interface"))
    f = y -> LogDensityProblems.logdensity(problem, y)
    model = Assay.LogDensityModel(f, d; names = names, backend = backend)
    if order isa LogDensityProblems.LogDensityOrder{0}
        return model
    end
    return Assay.LogDensityModel(f, d, model.names, ProblemGradient(problem, backend))
end

"""
    ProblemGradient(problem, fallback)

An AD backend that defers to the wrapped problem's own
`logdensity_and_gradient`. It exists so that a model arriving with an exact or
compiled gradient keeps it instead of being differentiated a second time.
"""
struct ProblemGradient{P,B<:ADBackend} <: ADBackend
    problem::P
    fallback::B
end

function Assay.logdensity_and_gradient(b::ProblemGradient, f, y::AbstractVector)
    v, g = LogDensityProblems.logdensity_and_gradient(b.problem, y)
    return v, collect(g)
end

end # module
