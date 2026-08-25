"""
Bijections between an unconstrained real vector and a constrained parameter,
each carrying its own log absolute Jacobian determinant.

Every sampler in this package works exclusively on `R^n`. A transform is the
only place where a constraint is expressed, and the only place where the
Jacobian correction is applied, so a new constraint is a new `AbstractTransform`
and never a new branch inside a sampler.

The contract for a transform `t`:

  * `udim(t)`   : length of the unconstrained vector it consumes
  * `cdim(t)`   : number of scalars in the constrained value it produces
  * `to_constrained(t, y)`   -> `(x, logabsdetjac)`
  * `to_unconstrained(t, x)` -> `y` such that `to_constrained(t, y)[1] == x`
  * `flat_names(t, name)`    : column names for the chain object

`logabsdetjac` is `log |det dx/dy|`, i.e. the term that must be *added* to a log
density written in the constrained space to obtain the correct log density in
the unconstrained space.
"""
abstract type AbstractTransform end

"""
    udim(t)

Dimension of the unconstrained vector consumed by `t`.
"""
function udim end

"""
    to_constrained(t, y)

Map an unconstrained vector to the constrained value, returning
`(x, log|det dx/dy|)`. The Jacobian term is returned alongside the value because
the two share intermediate quantities for every non-elementwise transform.
"""
function to_constrained end

"""
    to_unconstrained(t, x)

Inverse of [`to_constrained`](@ref). Throws a `DomainError` if `x` is outside
the constrained set.
"""
function to_unconstrained end

"""
    cdim(t)

Number of scalars in the constrained value produced by `t`.
"""
function cdim end

# --------------------------------------------------------------------------
# Elementwise transforms
# --------------------------------------------------------------------------

"""
    IdentityT(n)

`x = y` on `R^n`. Jacobian term is zero.
"""
struct IdentityT <: AbstractTransform
    n::Int
end

"""
    PositiveT(n)

`x = exp(y)`, mapping `R^n` to the positive orthant. `log|dx/dy| = sum(y)`.
"""
struct PositiveT <: AbstractTransform
    n::Int
end

"""
    UnitT(n)

`x = logistic(y)`, mapping `R^n` to `(0, 1)^n`.
"""
struct UnitT <: AbstractTransform
    n::Int
end

"""
    IntervalT(lo, hi, n)

`x = lo + (hi - lo) * logistic(y)`, mapping `R^n` to `(lo, hi)^n`.
"""
struct IntervalT{T<:Real} <: AbstractTransform
    lo::T
    hi::T
    n::Int
    function IntervalT(lo::T, hi::T, n::Int) where {T<:Real}
        hi > lo || throw(ArgumentError("IntervalT requires hi > lo, got ($lo, $hi)"))
        new{T}(lo, hi, n)
    end
end
IntervalT(lo::Real, hi::Real, n::Int) = IntervalT(promote(lo, hi)..., n)

"""
    LowerT(lo, n)

`x = lo + exp(y)`, mapping `R^n` to `(lo, Inf)^n`.
"""
struct LowerT{T<:Real} <: AbstractTransform
    lo::T
    n::Int
end

"""
    UpperT(hi, n)

`x = hi - exp(y)`, mapping `R^n` to `(-Inf, hi)^n`.
"""
struct UpperT{T<:Real} <: AbstractTransform
    hi::T
    n::Int
end

const ElementwiseT = Union{IdentityT,PositiveT,UnitT,IntervalT,LowerT,UpperT}

udim(t::ElementwiseT) = t.n
cdim(t::ElementwiseT) = t.n

to_constrained(t::IdentityT, y::AbstractVector) = (collect(y), zero(eltype(y)))
to_unconstrained(::IdentityT, x::AbstractVector) = collect(float.(x))

function to_constrained(t::PositiveT, y::AbstractVector)
    x = exp.(y)
    return x, sum(y)
end
function to_unconstrained(::PositiveT, x::AbstractVector)
    all(>(0), x) || throw(DomainError(x, "PositiveT expects strictly positive values"))
    return log.(float.(x))
end

function to_constrained(t::UnitT, y::AbstractVector)
    x = logistic.(y)
    # log d/dy logistic(y) = log s + log(1 - s) = loglogistic(y) + loglogistic(-y)
    lj = sum(loglogistic(yi) + loglogistic(-yi) for yi in y; init = zero(eltype(y)))
    return x, lj
end
function to_unconstrained(::UnitT, x::AbstractVector)
    all(xi -> 0 < xi < 1, x) || throw(DomainError(x, "UnitT expects values in (0, 1)"))
    return logit.(float.(x))
end

function to_constrained(t::IntervalT, y::AbstractVector)
    w = t.hi - t.lo
    x = t.lo .+ w .* logistic.(y)
    lj = sum(log(w) + loglogistic(yi) + loglogistic(-yi) for yi in y; init = zero(eltype(y)))
    return x, lj
end
function to_unconstrained(t::IntervalT, x::AbstractVector)
    all(xi -> t.lo < xi < t.hi, x) || throw(DomainError(x, "IntervalT expects values in ($(t.lo), $(t.hi))"))
    return [logit((xi - t.lo) / (t.hi - t.lo)) for xi in float.(x)]
end

function to_constrained(t::LowerT, y::AbstractVector)
    x = t.lo .+ exp.(y)
    return x, sum(y)
end
function to_unconstrained(t::LowerT, x::AbstractVector)
    all(>(t.lo), x) || throw(DomainError(x, "LowerT expects values above $(t.lo)"))
    return log.(float.(x) .- t.lo)
end

function to_constrained(t::UpperT, y::AbstractVector)
    x = t.hi .- exp.(y)
    return x, sum(y)
end
function to_unconstrained(t::UpperT, x::AbstractVector)
    all(<(t.hi), x) || throw(DomainError(x, "UpperT expects values below $(t.hi)"))
    return log.(t.hi .- float.(x))
end

# --------------------------------------------------------------------------
# Multivariate transforms
# --------------------------------------------------------------------------

"""
    SimplexT(K)

Stick-breaking map from `R^(K-1)` onto the interior of the `K`-simplex.

As with the other bounded transforms, the image is the open simplex only up to
floating point: a component underflows to exactly zero for `y` beyond roughly
37 in magnitude. The log density at such a point is `-Inf` for any density with
an interior support, so a sampler rejects it, but the constrained value handed
to user code is on the boundary rather than strictly inside it.

With `z_k = logistic(y_k + log(1 / (K - k)))` and `r_k = 1 - sum(x_1..x_{k-1})`,

    x_k = r_k * z_k   for k < K,      x_K = r_K

and

    log|dx/dy| = sum_k [ log z_k + log(1 - z_k) + log r_k ].

The offset `log(1 / (K - k))` makes `y = 0` map to the uniform point `1/K`,
which is the same convention Stan uses; without it the map is still a bijection
but the origin is badly off-centre for large `K`.
"""
struct SimplexT <: AbstractTransform
    K::Int
    function SimplexT(K::Int)
        K >= 2 || throw(ArgumentError("SimplexT requires K >= 2, got $K"))
        new(K)
    end
end

udim(t::SimplexT) = t.K - 1
cdim(t::SimplexT) = t.K

function to_constrained(t::SimplexT, y::AbstractVector)
    K = t.K
    T = float(eltype(y))
    x = Vector{T}(undef, K)
    lj = zero(T)
    remaining = one(T)
    @inbounds for k in 1:(K - 1)
        zk = logistic(y[k] + log(one(T) / (K - k)))
        xk = remaining * zk
        lj += loglogistic(y[k] + log(one(T) / (K - k))) +
              loglogistic(-(y[k] + log(one(T) / (K - k)))) + log(remaining)
        x[k] = xk
        remaining -= xk
    end
    # The accumulated subtraction can leave `remaining` at a value like -1e-17
    # for extreme `y`; clamping keeps the contract that the output is a point of
    # the simplex, at a cost below the rounding error already present in the sum.
    x[K] = max(remaining, zero(T))
    return x, lj
end

function to_unconstrained(t::SimplexT, x::AbstractVector)
    K = t.K
    length(x) == K || throw(DimensionMismatch("SimplexT($K) expects a length-$K point"))
    all(>(0), x) || throw(DomainError(x, "simplex points must be strictly positive"))
    isapprox(sum(x), 1; atol = 1e-8) || throw(DomainError(sum(x), "simplex points must sum to 1"))
    T = float(eltype(x))
    y = Vector{T}(undef, K - 1)
    remaining = one(T)
    @inbounds for k in 1:(K - 1)
        zk = x[k] / remaining
        y[k] = logit(zk) - log(one(T) / (K - k))
        remaining -= x[k]
    end
    return y
end

"""
    OrderedT(n)

Maps `R^n` onto `{x : x_1 < x_2 < ... < x_n}` via `x_1 = y_1`,
`x_k = x_{k-1} + exp(y_k)`. `log|dx/dy| = sum(y_2..y_n)`.

Included because ordering is the constraint that makes mixture models
identifiable, and it exercises a non-diagonal Jacobian that still has a
closed-form determinant.
"""
struct OrderedT <: AbstractTransform
    n::Int
end

udim(t::OrderedT) = t.n
cdim(t::OrderedT) = t.n

function to_constrained(t::OrderedT, y::AbstractVector)
    n = t.n
    T = float(eltype(y))
    x = Vector{T}(undef, n)
    x[1] = y[1]
    lj = zero(T)
    @inbounds for k in 2:n
        x[k] = x[k - 1] + exp(y[k])
        lj += y[k]
    end
    return x, lj
end

function to_unconstrained(t::OrderedT, x::AbstractVector)
    issorted(x) && allunique(x) || throw(DomainError(x, "OrderedT expects a strictly increasing vector"))
    T = float(eltype(x))
    y = Vector{T}(undef, length(x))
    y[1] = x[1]
    @inbounds for k in 2:length(x)
        y[k] = log(x[k] - x[k - 1])
    end
    return y
end

"""
    CorrCholeskyT(K)

Maps `R^(K(K-1)/2)` onto the Cholesky factors of `K x K` correlation matrices:
lower triangular, positive diagonal, unit-length rows.

The route is the canonical partial correlations. Each unconstrained coordinate
becomes `z = tanh(y)` in `(-1, 1)`, and the rows are then built so that each one
has unit length:

    L[i,1] = z[i,1],   L[i,j] = z[i,j] * sqrt(1 - sum(L[i,1:j-1].^2)),
    L[i,i] = sqrt(1 - sum(L[i,1:i-1].^2))

Parameterising a correlation matrix directly does not work: the constraint is
positive definiteness, which is not a box, so no elementwise map onto it exists.
Partial correlations are the reparameterisation that turns it into one, and this
is the transform that hierarchical models with correlated effects need in order
to be written at all.
"""
struct CorrCholeskyT <: AbstractTransform
    K::Int
    function CorrCholeskyT(K::Int)
        K >= 2 || throw(ArgumentError("CorrCholeskyT requires K >= 2, got $K"))
        new(K)
    end
end

udim(t::CorrCholeskyT) = (t.K * (t.K - 1)) ÷ 2
cdim(t::CorrCholeskyT) = (t.K * (t.K - 1)) ÷ 2      # reported as correlations

function to_constrained(t::CorrCholeskyT, y::AbstractVector)
    K = t.K
    T = float(eltype(y))
    L = zeros(T, K, K)
    L[1, 1] = one(T)
    lj = zero(T)
    idx = 0
    @inbounds for i in 2:K
        remaining = one(T)
        for j in 1:(i - 1)
            idx += 1
            z = tanh(y[idx])
            # d tanh / dy = 1 - z^2, in a form that does not underflow
            lj += log(4) - 2 * abs(y[idx]) - 2 * log1pexp(-2 * abs(y[idx]))
            s = sqrt(remaining)
            L[i, j] = z * s
            # every strictly lower entry is a free coordinate and carries the
            # factor s; the entry determined by the unit-row constraint is the
            # diagonal L[i,i], not the last off-diagonal one
            lj += log(s)
            remaining -= L[i, j]^2
            remaining = max(remaining, zero(T))
        end
        L[i, i] = sqrt(remaining)
    end
    return LowerTriangular(L), lj
end

function to_unconstrained(t::CorrCholeskyT, L::AbstractMatrix)
    K = t.K
    size(L) == (K, K) || throw(DimensionMismatch("expected a $K x $K factor"))
    T = float(eltype(L))
    y = Vector{T}(undef, udim(t))
    idx = 0
    @inbounds for i in 2:K
        remaining = one(T)
        for j in 1:(i - 1)
            idx += 1
            s = sqrt(remaining)
            z = L[i, j] / s
            abs(z) < 1 || throw(DomainError(z, "not a valid correlation Cholesky factor"))
            y[idx] = atanh(z)
            remaining -= L[i, j]^2
        end
    end
    return y
end

"""
    correlation_matrix(L)

`L * L'`, the correlation matrix a Cholesky factor stands for.
"""
correlation_matrix(L::AbstractMatrix) = Matrix(L) * Matrix(L)'

# --------------------------------------------------------------------------
# Scalar wrapper
# --------------------------------------------------------------------------

"""
    ScalarT(inner)

Wraps a one-dimensional transform so the model sees a scalar rather than a
length-one vector. Purely an ergonomics layer: `θ.σ` reads better than `θ.σ[1]`
and keeps user log densities free of indexing noise.
"""
struct ScalarT{T<:AbstractTransform} <: AbstractTransform
    inner::T
    function ScalarT(inner::T) where {T<:AbstractTransform}
        udim(inner) == 1 && cdim(inner) == 1 ||
            throw(ArgumentError("ScalarT wraps 1-dimensional transforms only"))
        new{T}(inner)
    end
end

udim(::ScalarT) = 1
cdim(::ScalarT) = 1

function to_constrained(t::ScalarT, y::AbstractVector)
    x, lj = to_constrained(t.inner, y)
    return x[1], lj
end
to_unconstrained(t::ScalarT, x::Real) = to_unconstrained(t.inner, [x])
to_unconstrained(t::ScalarT, x::AbstractVector) = to_unconstrained(t.inner, x)

# --------------------------------------------------------------------------
# User-facing constructors
# --------------------------------------------------------------------------

"""
    unconstrained(), unconstrained(n)

Unconstrained scalar / length-`n` vector parameter. Named `unconstrained`
rather than `real` so it does not shadow `Base.real`.
"""
unconstrained() = ScalarT(IdentityT(1))
unconstrained(n::Int) = IdentityT(n)

"""
    positive(), positive(n)

Parameter constrained to `(0, Inf)` through `exp`.
"""
positive() = ScalarT(PositiveT(1))
positive(n::Int) = PositiveT(n)

"""
    unit(), unit(n)

Parameter constrained to `(0, 1)` through `logistic`.
"""
unit() = ScalarT(UnitT(1))
unit(n::Int) = UnitT(n)

"""
    interval(lo, hi), interval(lo, hi, n)

Parameter constrained to `(lo, hi)`.
"""
interval(lo::Real, hi::Real) = ScalarT(IntervalT(lo, hi, 1))
interval(lo::Real, hi::Real, n::Int) = IntervalT(lo, hi, n)

"""
    lower(lo), lower(lo, n)

Parameter constrained to `(lo, Inf)`.
"""
lower(lo::Real) = ScalarT(LowerT(lo, 1))
lower(lo::Real, n::Int) = LowerT(lo, n)

"""
    upper(hi), upper(hi, n)

Parameter constrained to `(-Inf, hi)`.
"""
upper(hi::Real) = ScalarT(UpperT(hi, 1))
upper(hi::Real, n::Int) = UpperT(hi, n)

"""
    simplex(K)

Parameter on the interior of the `K`-simplex; consumes `K - 1` unconstrained
coordinates.
"""
simplex(K::Int) = SimplexT(K)

"""
    ordered(n)

Strictly increasing length-`n` vector parameter.
"""
ordered(n::Int) = OrderedT(n)

"""
    corr_cholesky(K)

Cholesky factor of a `K x K` correlation matrix, consuming `K(K-1)/2`
unconstrained coordinates. The model sees a `LowerTriangular` factor `L`; draws
are reported as the correlations themselves, `R[2,1]`, `R[3,1]` and so on, since
that is what anyone reading a summary wants.
"""
corr_cholesky(K::Int) = CorrCholeskyT(K)

# --------------------------------------------------------------------------
# Naming, for chain columns
# --------------------------------------------------------------------------

function flat_names(t::CorrCholeskyT, name::Symbol)
    return [Symbol(name, "[", i, ",", j, "]") for i in 2:t.K for j in 1:(i - 1)]
end

flat_names(::ScalarT, name::Symbol) = [name]
flat_names(t::AbstractTransform, name::Symbol) =
    cdim(t) == 1 ? [name] : [Symbol(name, "[", i, "]") for i in 1:cdim(t)]

"""
    flatten(x)

Flatten a constrained parameter value (scalar or vector) into a vector of
scalars, matching the order produced by [`flat_names`](@ref).
"""
flatten(x::Real) = [float(x)]
flatten(x::AbstractVector) = collect(float.(x))

"""
    flatten(t, x)

Flatten a constrained value for storage, given the transform that produced it.
The default ignores the transform; a correlation factor reports the
correlations below the diagonal rather than the factor's own entries, because
`L[3,2]` means nothing to a reader and `R[3,2]` means everything.
"""
flatten(::AbstractTransform, x) = flatten(x)

function flatten(t::CorrCholeskyT, L::AbstractMatrix)
    R = correlation_matrix(L)
    return [R[i, j] for i in 2:t.K for j in 1:(i - 1)]
end
