# Sampling a model this package did not define.
#
# The samplers only ever ask a model for its dimension, its log density and its
# gradient. Anything that can answer those three questions can be sampled here,
# whether or not it was written against `Model`. `LogDensityModel` is that
# adapter: it wraps a callable, or any object implementing the
# LogDensityProblems interface, and presents it as an `AbstractModel`.
#
# The direction matters for a user. It means a log density already written for
# Turing, DynamicPPL, Stan through BridgeStan, or by hand, can be handed to NUTS
# here without being rewritten, and conversely that an Assay `Model` can be
# handed to any sampler in the wider ecosystem (see the LogDensityProblems
# extension, which supplies that half).

"""
    LogDensityModel(logdensity, dimension; names = nothing, backend = ForwardDiffAD())

Adapt a plain log density function on `R^n` to the [`AbstractModel`](@ref)
interface, so that every sampler in this package can run on it.

`logdensity` is called as `logdensity(y)` and must return a real number, `-Inf`
being the way to express "outside the support". Gradients are taken with
`backend`, so the function has to accept the dual numbers of whichever backend
is chosen; the densities in this package all do.

There is no transform layer here and there cannot be: an arbitrary function
carries no declaration of which coordinates are constrained. Draws are therefore
reported exactly as the sampler sees them. If a parameter is bounded, either
transform it inside `logdensity` and add the log Jacobian determinant yourself,
or use [`Model`](@ref), which does both for you.

    model = LogDensityModel(y -> -0.5 * sum(abs2, y), 3)
    chain = sample(model, NUTS(), 1000)

Loading LogDensityProblems adds a method taking any object implementing that
interface, using the object's own gradient when it advertises one.
"""
struct LogDensityModel{F,B<:ADBackend} <: AbstractModel
    logdensity::F
    dimension::Int
    names::Vector{Symbol}
    backend::B
end

function LogDensityModel(f, dimension::Int; names = nothing, backend::ADBackend = ForwardDiffAD())
    dimension > 0 || throw(ArgumentError("dimension must be positive, got $dimension"))
    nms = if names === nothing
        dimension == 1 ? [:x] : [Symbol("x[$i]") for i in 1:dimension]
    else
        length(names) == dimension ||
            throw(DimensionMismatch("got $(length(names)) names for $dimension parameters"))
        collect(Symbol.(names))
    end
    return LogDensityModel(f, dimension, nms, backend)
end

dimension(m::LogDensityModel) = m.dimension
flat_dimension(m::LogDensityModel) = m.dimension
parameter_names(m::LogDensityModel) = copy(m.names)

function logdensity(m::LogDensityModel, y::AbstractVector; jacobian::Bool = true)
    length(y) == m.dimension ||
        throw(DimensionMismatch("model is on R^$(m.dimension), got a length-$(length(y)) vector"))
    return m.logdensity(y)
end

function logdensity_and_gradient(m::LogDensityModel, y::AbstractVector;
                                 backend::ADBackend = m.backend, jacobian::Bool = true)
    return logdensity_and_gradient(backend, m.logdensity, y)
end

"""
    constrain(m::LogDensityModel, y)

Returns the draw under its parameter names and a zero Jacobian term, because a
wrapped log density has no declared constraints to correct for.
"""
function constrain(m::LogDensityModel, y::AbstractVector)
    return (x = collect(float.(y)),), zero(float(eltype(y)))
end

unconstrain(::LogDensityModel, theta::NamedTuple) = collect(float.(theta.x))
flatten_draw(m::LogDensityModel, y::AbstractVector) = collect(float.(y))

Base.show(io::IO, m::LogDensityModel) =
    print(io, "LogDensityModel(dimension ", m.dimension, ", ", nameof(typeof(m.backend)), ")")

