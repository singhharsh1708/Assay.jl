# The model interface.
#
# A `Model` is two things and nothing else:
#
#   1. an ordered set of named parameters, each with a transform that says how it
#      is embedded in `R^n`;
#   2. a function of the *constrained* parameters returning the log joint density
#      `log p(θ) + log p(y | θ)`.
#
# The user writes the log joint in the space the mathematics is written in
# (`σ > 0`, `p ∈ (0,1)`, `w` on the simplex). The library owns the mapping to
# `R^n` and adds the log absolute Jacobian determinant. Samplers then only ever
# see an unconstrained vector and a gradient, which is why adding a sampler never
# requires touching this file.
#
#     model = Model((p = unit(),), θ -> logpdf(Beta(1, 1), θ.p) + loglikelihood(Bernoulli(θ.p), data))
#     logdensity(model, [0.3])

"""
    AbstractModel

The interface a sampler is written against. Anything that answers

    dimension(m)                             -> Int
    logdensity(m, y)                         -> Real
    logdensity_and_gradient(m, y; backend)   -> (Real, AbstractVector)
    parameter_names(m)                       -> Vector{Symbol}
    flat_dimension(m)                        -> Int
    flatten_draw(m, y)                       -> Vector{Float64}

can be sampled by every algorithm in this package. [`Model`](@ref) is the
implementation that owns constraints and Jacobians; [`LogDensityModel`](@ref)
wraps anything satisfying the LogDensityProblems interface, so a model written
for another package can be sampled here without being rewritten.
"""
abstract type AbstractModel end

"""
    Model(params::NamedTuple, logjoint)

A model as an ordered set of named, constrained parameters plus a log joint
density written in the constrained space.
"""
struct Model{N,T<:Tuple,F} <: AbstractModel
    names::NTuple{N,Symbol}
    transforms::T
    logjoint::F
    dimension::Int
end

"""
    Model(params::NamedTuple, logjoint)

`params` maps parameter names to transforms, in the order they are packed into
the unconstrained vector; `logjoint` takes a `NamedTuple` of constrained values.
"""
function Model(params::NamedTuple, logjoint)
    ts = values(params)
    all(t -> t isa AbstractTransform, ts) ||
        throw(ArgumentError("every entry of `params` must be an AbstractTransform"))
    return Model(keys(params), ts, logjoint, sum(udim, ts; init = 0))
end

"""
    dimension(model)

Dimension of the unconstrained space the samplers work in.
"""
dimension(m::Model) = m.dimension

Base.show(io::IO, m::Model) =
    print(io, "Model(", join(("$n::$(typeof(t).name.name)" for (n, t) in zip(m.names, m.transforms)), ", "),
          ") on R^", m.dimension)

@inline function _unpack(ts::Tuple, y::AbstractVector, off::Int)
    t = first(ts)
    n = udim(t)
    x, lj = to_constrained(t, view(y, (off + 1):(off + n)))
    rest, ljrest = _unpack(Base.tail(ts), y, off + n)
    return (x, rest...), lj + ljrest
end
@inline _unpack(::Tuple{}, y::AbstractVector, off::Int) = ((), zero(float(eltype(y))))

"""
    constrain(model, y)

Map an unconstrained vector to the constrained `NamedTuple` and the log absolute
Jacobian determinant of the map.
"""
function constrain(m::Model, y::AbstractVector)
    length(y) == m.dimension ||
        throw(DimensionMismatch("model is on R^$(m.dimension), got a length-$(length(y)) vector"))
    vals, lj = _unpack(m.transforms, y, 0)
    return NamedTuple{m.names}(vals), lj
end

"""
    unconstrain(model, θ::NamedTuple)

Inverse of [`constrain`](@ref); useful for placing a sampler at a known point.
"""
function unconstrain(m::Model, theta::NamedTuple)
    y = Float64[]
    for (name, t) in zip(m.names, m.transforms)
        append!(y, to_unconstrained(t, getproperty(theta, name)))
    end
    return y
end

"""
    logdensity(model, y; jacobian = true)

Log density on the unconstrained space:

    log p(θ(y)) + log |det dθ/dy|

Setting `jacobian = false` drops the correction. That is *not* a valid target
distribution; the option exists so the test suite can show, from the outside,
that dropping the term breaks the conjugate checks (`test/negative_controls.jl`).
"""
function logdensity(m::Model, y::AbstractVector; jacobian::Bool = true)
    theta, lj = constrain(m, y)
    lp = m.logjoint(theta)
    return jacobian ? lp + lj : lp
end

"""
    logdensity_and_gradient(model, y; backend = ForwardDiffAD(), jacobian = true)

Value and gradient of [`logdensity`](@ref) with respect to the unconstrained
vector.
"""
function logdensity_and_gradient(m::Model, y::AbstractVector;
                                 backend::ADBackend = ForwardDiffAD(), jacobian::Bool = true)
    return logdensity_and_gradient(backend, z -> logdensity(m, z; jacobian = jacobian), y)
end

"""
    parameter_names(model)

Flattened column names, e.g. `[:mu, :sigma, Symbol("w[1]"), ...]`.
"""
function parameter_names(m::Model)
    names = Symbol[]
    for (name, t) in zip(m.names, m.transforms)
        append!(names, flat_names(t, name))
    end
    return names
end

"""
    flat_dimension(model)

Number of scalar columns in the constrained space, which exceeds
[`dimension`](@ref) whenever a simplex parameter is present.
"""
flat_dimension(m::Model) = sum(cdim, m.transforms; init = 0)

"""
    flatten_draw(model, y)

Constrained parameter vector corresponding to unconstrained `y`, flattened in
the order of [`parameter_names`](@ref).
"""
function flatten_draw(m::Model, y::AbstractVector)
    theta, _ = constrain(m, y)
    out = Float64[]
    for name in m.names
        append!(out, flatten(getproperty(theta, name)))
    end
    return out
end
