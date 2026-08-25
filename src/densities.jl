"""
A small set of hand-written log densities and samplers.

Distributions.jl is not a dependency of the library: the point of the repository
is that every number can be traced to code in it. Each density here is written
so that its parameters can be `ForwardDiff.Dual`s (that is what makes gradients
of a user model work) while the observed value is plain data.

Parameterisation warning: `Gamma(shape, rate)` uses the *rate*, not the scale,
so `Gamma(a, b)` here equals `Distributions.Gamma(a, 1 / b)`. Rate is the
parameterisation in which Gamma-Poisson conjugacy is stated, which is what this
package is mostly used for.
"""
abstract type Density end

"""
    UnivariateDensity

Densities over a single scalar. These support `logpdf`, `rand`, and where the
verification suite needs them, `cdf`, `quantile`, `mean` and `var`.
"""
abstract type UnivariateDensity <: Density end

"""
    MultivariateDensity

Densities over a vector, such as `MvNormal` and `Dirichlet`.
"""
abstract type MultivariateDensity <: Density end

"""
    logpdf(d, x)

Log density of `d` at `x`, `-Inf` outside the support.
"""
function logpdf end

"""
    loglikelihood(d, xs)

Sum of `logpdf(d, x)` over an iterable of independent observations.
"""
loglikelihood(d::UnivariateDensity, xs) = sum(x -> logpdf(d, x), xs)

const LOG2PI = log(2 * pi)

# --------------------------------------------------------------------------
# Univariate continuous
# --------------------------------------------------------------------------

"""
    Normal(mu, sigma)

Normal density with standard deviation (not variance) `sigma`.
"""
struct Normal{T<:Real,S<:Real} <: UnivariateDensity
    mu::T
    sigma::S
end
Normal() = Normal(0.0, 1.0)

function logpdf(d::Normal, x::Real)
    d.sigma > 0 || return oftype(float(d.sigma * x), -Inf)
    z = (x - d.mu) / d.sigma
    return -0.5 * z * z - log(d.sigma) - 0.5 * LOG2PI
end
Base.rand(rng::AbstractRNG, d::Normal) = d.mu + d.sigma * randn(rng)
mean(d::Normal) = d.mu
var(d::Normal) = d.sigma^2

"""
    LogNormal(mu, sigma)

Density of `exp(z)` for `z ~ Normal(mu, sigma)`.
"""
struct LogNormal{T<:Real,S<:Real} <: UnivariateDensity
    mu::T
    sigma::S
end

function logpdf(d::LogNormal, x::Real)
    (d.sigma > 0 && x > 0) || return oftype(float(d.sigma * one(x)), -Inf)
    return logpdf(Normal(d.mu, d.sigma), log(x)) - log(x)
end
Base.rand(rng::AbstractRNG, d::LogNormal) = exp(d.mu + d.sigma * randn(rng))
mean(d::LogNormal) = exp(d.mu + d.sigma^2 / 2)
var(d::LogNormal) = (exp(d.sigma^2) - 1) * exp(2 * d.mu + d.sigma^2)

"""
    Cauchy(mu, sigma)

Standard heavy-tailed control case; used for the funnel and for weak priors.
"""
struct Cauchy{T<:Real,S<:Real} <: UnivariateDensity
    mu::T
    sigma::S
end

function logpdf(d::Cauchy, x::Real)
    d.sigma > 0 || return oftype(float(d.sigma * x), -Inf)
    z = (x - d.mu) / d.sigma
    return -log(pi) - log(d.sigma) - log1p(z * z)
end
Base.rand(rng::AbstractRNG, d::Cauchy) = d.mu + d.sigma * tan(pi * (rand(rng) - 0.5))

"""
    StudentT(nu, mu, sigma)

Location-scale Student t with `nu` degrees of freedom.
"""
struct StudentT{N<:Real,T<:Real,S<:Real} <: UnivariateDensity
    nu::N
    mu::T
    sigma::S
end
StudentT(nu::Real) = StudentT(nu, 0.0, 1.0)

function logpdf(d::StudentT, x::Real)
    (d.nu > 0 && d.sigma > 0) || return oftype(float(d.sigma * x), -Inf)
    z = (x - d.mu) / d.sigma
    return loggamma((d.nu + 1) / 2) - loggamma(d.nu / 2) - 0.5 * log(d.nu * pi) -
           log(d.sigma) - (d.nu + 1) / 2 * log1p(z * z / d.nu)
end
function Base.rand(rng::AbstractRNG, d::StudentT)
    g = rand(rng, Gamma(d.nu / 2, 0.5))    # chi-square with nu dof
    return d.mu + d.sigma * randn(rng) / sqrt(g / d.nu)
end

"""
    Uniform(a, b)
"""
struct Uniform{T<:Real,S<:Real} <: UnivariateDensity
    a::T
    b::S
end
Uniform() = Uniform(0.0, 1.0)

function logpdf(d::Uniform, x::Real)
    (d.b > d.a && d.a <= x <= d.b) || return oftype(float(d.b * one(x)), -Inf)
    return -log(d.b - d.a)
end
Base.rand(rng::AbstractRNG, d::Uniform) = d.a + (d.b - d.a) * rand(rng)
mean(d::Uniform) = (d.a + d.b) / 2
var(d::Uniform) = (d.b - d.a)^2 / 12

"""
    Exponential(rate)
"""
struct Exponential{T<:Real} <: UnivariateDensity
    rate::T
end

function logpdf(d::Exponential, x::Real)
    (d.rate > 0 && x >= 0) || return oftype(float(d.rate * one(x)), -Inf)
    return log(d.rate) - d.rate * x
end
Base.rand(rng::AbstractRNG, d::Exponential) = -log(rand(rng)) / d.rate
mean(d::Exponential) = 1 / d.rate
var(d::Exponential) = 1 / d.rate^2

"""
    Gamma(shape, rate)

`p(x) = rate^shape / Gamma(shape) * x^(shape-1) * exp(-rate * x)`.
Note this is the rate parameterisation (see the module docstring).
"""
struct Gamma{A<:Real,B<:Real} <: UnivariateDensity
    shape::A
    rate::B
end

function logpdf(d::Gamma, x::Real)
    (d.shape > 0 && d.rate > 0 && x > 0) || return oftype(float(d.shape * d.rate * one(x)), -Inf)
    return d.shape * log(d.rate) - loggamma(d.shape) + (d.shape - 1) * log(x) - d.rate * x
end
mean(d::Gamma) = d.shape / d.rate
var(d::Gamma) = d.shape / d.rate^2

"""
    rand(rng, d::Gamma)

Marsaglia and Tsang (2000) squeeze method for `shape >= 1`, with the
`shape < 1` case handled by the standard `x = G(shape + 1) * U^(1/shape)` boost.
"""
function Base.rand(rng::AbstractRNG, d::Gamma)
    a = float(d.shape)
    if a < 1
        return rand(rng, Gamma(a + 1, d.rate)) * rand(rng)^(1 / a)
    end
    dd = a - 1 / 3
    c = 1 / sqrt(9 * dd)
    while true
        x = randn(rng)
        v = (1 + c * x)^3
        v <= 0 && continue
        u = rand(rng)
        if log(u) < 0.5 * x^2 + dd - dd * v + dd * log(v)
            return dd * v / d.rate
        end
    end
end

"""
    InverseGamma(shape, scale)

Density of `1 / g` for `g ~ Gamma(shape, rate = scale)`. This is the conjugate
prior for a normal variance.
"""
struct InverseGamma{A<:Real,B<:Real} <: UnivariateDensity
    shape::A
    scale::B
end

function logpdf(d::InverseGamma, x::Real)
    (d.shape > 0 && d.scale > 0 && x > 0) || return oftype(float(d.shape * d.scale * one(x)), -Inf)
    return d.shape * log(d.scale) - loggamma(d.shape) - (d.shape + 1) * log(x) - d.scale / x
end
Base.rand(rng::AbstractRNG, d::InverseGamma) = 1 / rand(rng, Gamma(d.shape, d.scale))
mean(d::InverseGamma) = d.shape > 1 ? d.scale / (d.shape - 1) : Inf

"""
    Beta(a, b)
"""
struct Beta{A<:Real,B<:Real} <: UnivariateDensity
    a::A
    b::B
end

function logpdf(d::Beta, x::Real)
    (d.a > 0 && d.b > 0 && 0 < x < 1) || return oftype(float(d.a * d.b * one(x)), -Inf)
    return (d.a - 1) * log(x) + (d.b - 1) * log1p(-x) - logbeta(d.a, d.b)
end
function Base.rand(rng::AbstractRNG, d::Beta)
    g1 = rand(rng, Gamma(d.a, 1.0))
    g2 = rand(rng, Gamma(d.b, 1.0))
    return g1 / (g1 + g2)
end
mean(d::Beta) = d.a / (d.a + d.b)
var(d::Beta) = d.a * d.b / ((d.a + d.b)^2 * (d.a + d.b + 1))

# --------------------------------------------------------------------------
# Univariate discrete
# --------------------------------------------------------------------------

"""
    Bernoulli(p)
"""
struct Bernoulli{T<:Real} <: UnivariateDensity
    p::T
end

function logpdf(d::Bernoulli, x::Real)
    (0 <= d.p <= 1 && (x == 0 || x == 1)) || return oftype(float(d.p * one(x)), -Inf)
    return x == 1 ? log(d.p) : log1p(-d.p)
end
Base.rand(rng::AbstractRNG, d::Bernoulli) = rand(rng) < d.p ? 1 : 0
mean(d::Bernoulli) = d.p
var(d::Bernoulli) = d.p * (1 - d.p)

"""
    Binomial(n, p)
"""
struct Binomial{T<:Real} <: UnivariateDensity
    n::Int
    p::T
end

function logpdf(d::Binomial, x::Real)
    (0 <= d.p <= 1 && isinteger(x) && 0 <= x <= d.n) || return oftype(float(d.p * one(x)), -Inf)
    k = Int(x)
    lc = loggamma(d.n + 1) - loggamma(k + 1) - loggamma(d.n - k + 1)
    return lc + (d.p == 0 ? (k == 0 ? zero(float(d.p)) : -Inf) : k * log(d.p)) +
           (d.p == 1 ? (k == d.n ? zero(float(d.p)) : -Inf) : (d.n - k) * log1p(-d.p))
end
function Base.rand(rng::AbstractRNG, d::Binomial)
    s = 0
    for _ in 1:d.n
        s += rand(rng) < d.p ? 1 : 0
    end
    return s
end
mean(d::Binomial) = d.n * d.p
var(d::Binomial) = d.n * d.p * (1 - d.p)

"""
    Poisson(lambda)
"""
struct Poisson{T<:Real} <: UnivariateDensity
    lambda::T
end

function logpdf(d::Poisson, x::Real)
    (d.lambda >= 0 && isinteger(x) && x >= 0) || return oftype(float(d.lambda * one(x)), -Inf)
    d.lambda == 0 && return x == 0 ? zero(float(d.lambda)) : oftype(float(d.lambda), -Inf)
    return x * log(d.lambda) - d.lambda - loggamma(x + 1)
end

"""
    rand(rng, d::Poisson)

Knuth's product method below `lambda = 30`, and a gamma-mixture free inversion
with a running cdf above it. Exactness matters here: simulation based
calibration is only a valid test if the forward simulator is exact.
"""
function Base.rand(rng::AbstractRNG, d::Poisson)
    lam = float(d.lambda)
    lam == 0 && return 0
    if lam < 30
        L = exp(-lam)
        k = 0
        p = 1.0
        while true
            p *= rand(rng)
            p <= L && return k
            k += 1
        end
    else
        # Inversion from the mode outward is stable for the lambda values used
        # in this package; log-space recursion avoids underflow for large k.
        u = rand(rng)
        cdf = 0.0
        k = 0
        logpk = -lam
        while true
            cdf += exp(logpk)
            cdf >= u && return k
            k += 1
            logpk += log(lam) - log(k)
            k > 10 * lam + 1000 && return k
        end
    end
end
mean(d::Poisson) = d.lambda
var(d::Poisson) = d.lambda

# --------------------------------------------------------------------------
# Multivariate
# --------------------------------------------------------------------------

"""
    MvNormal(mu, Sigma)

Multivariate normal with a dense covariance. The Cholesky factor is computed
once at construction, so repeated `logpdf` calls cost one triangular solve.
"""
struct MvNormal{T<:Real,M<:AbstractMatrix,C,L<:Real} <: MultivariateDensity
    mu::Vector{T}
    Sigma::M
    chol::C
    logdetSigma::L
end

function MvNormal(mu::AbstractVector, Sigma::AbstractMatrix)
    C = cholesky(Symmetric(Sigma))
    return MvNormal(collect(float.(mu)), Sigma, C, 2 * sum(log, diag(C.U)))
end
MvNormal(mu::AbstractVector, sigma::Real) = MvNormal(mu, Matrix(sigma^2 * I, length(mu), length(mu)))

function logpdf(d::MvNormal, x::AbstractVector)
    z = x .- d.mu
    q = dot(z, d.chol \ z)
    return -0.5 * (q + d.logdetSigma + length(d.mu) * LOG2PI)
end
Base.rand(rng::AbstractRNG, d::MvNormal) = d.mu .+ d.chol.L * randn(rng, length(d.mu))
mean(d::MvNormal) = d.mu

"""
    MvNormalCholesky(mu, L)

Multivariate normal given the lower Cholesky factor of its covariance, so that
`Sigma = L L'`.

This is the form to use when the covariance is being estimated. Building
`Sigma` and factorising it again costs a factorisation per evaluation and can
lose positive definiteness to rounding at exactly the moment the sampler is
exploring a near-singular region; a factor is positive definite by
construction. It also composes directly with [`corr_cholesky`](@ref):
`Diagonal(sigma) * L` is the covariance factor for scales `sigma` and
correlation factor `L`.
"""
struct MvNormalCholesky{T<:Real,F<:AbstractMatrix} <: MultivariateDensity
    mu::Vector{T}
    L::F
end
MvNormalCholesky(mu::AbstractVector, L::AbstractMatrix) = MvNormalCholesky(collect(mu), L)

function logpdf(d::MvNormalCholesky, x::AbstractVector)
    n = length(d.mu)
    length(x) == n || return -Inf
    z = LowerTriangular(d.L) \ (x .- d.mu)          # one triangular solve
    logdet = zero(eltype(z))
    @inbounds for i in 1:n
        d.L[i, i] > 0 || return oftype(logdet, -Inf)
        logdet += log(d.L[i, i])
    end
    return -0.5 * sum(abs2, z) - logdet - n / 2 * LOG2PI
end

Base.rand(rng::AbstractRNG, d::MvNormalCholesky) = d.mu .+ d.L * randn(rng, length(d.mu))
mean(d::MvNormalCholesky) = d.mu

"""
    covariance(d::MvNormalCholesky)

`L L'`, the covariance the factor stands for.
"""
covariance(d::MvNormalCholesky) = Matrix(d.L) * Matrix(d.L)'

"""
    Dirichlet(alpha)
"""
struct Dirichlet{T<:Real} <: MultivariateDensity
    alpha::Vector{T}
end
Dirichlet(K::Int, a::Real) = Dirichlet(fill(float(a), K))

function logpdf(d::Dirichlet, x::AbstractVector)
    (all(>(0), d.alpha) && all(>(0), x)) || return -Inf
    isapprox(sum(x), 1; atol = 1e-8) || return -Inf
    s = sum((a - 1) * log(xi) for (a, xi) in zip(d.alpha, x))
    return s + loggamma(sum(d.alpha)) - sum(loggamma, d.alpha)
end
function Base.rand(rng::AbstractRNG, d::Dirichlet)
    g = [rand(rng, Gamma(a, 1.0)) for a in d.alpha]
    return g ./ sum(g)
end
mean(d::Dirichlet) = d.alpha ./ sum(d.alpha)

# Convenience: sampling n independent draws.
Base.rand(rng::AbstractRNG, d::UnivariateDensity, n::Int) = [rand(rng, d) for _ in 1:n]
Base.rand(d::Density) = rand(Random.default_rng(), d)
Base.rand(d::UnivariateDensity, n::Int) = rand(Random.default_rng(), d, n)

"""
    Categorical(p)

Distribution over `1:length(p)` with probabilities `p`.
"""
struct Categorical{T<:Real} <: UnivariateDensity
    p::Vector{T}
end

function logpdf(d::Categorical, x::Real)
    (isinteger(x) && 1 <= x <= length(d.p)) || return oftype(float(first(d.p)), -Inf)
    return log(d.p[Int(x)])
end
function Base.rand(rng::AbstractRNG, d::Categorical)
    u = rand(rng)
    c = zero(float(eltype(d.p)))
    for (i, pi) in enumerate(d.p)
        c += pi
        u <= c && return i
    end
    return length(d.p)
end
mean(d::Categorical) = sum(i * p for (i, p) in enumerate(d.p))

"""
    Multinomial(n, p)

Counts of `n` independent categorical trials. Used by the simplex dynamical
system example, where the state of the system is the probability vector.
"""
struct Multinomial{T<:Real} <: MultivariateDensity
    n::Int
    p::Vector{T}
end

function logpdf(d::Multinomial, x::AbstractVector)
    length(x) == length(d.p) || return -Inf
    (all(xi -> isinteger(xi) && xi >= 0, x) && sum(x) == d.n) || return -Inf
    all(>=(0), d.p) || return oftype(float(first(d.p)), -Inf)
    s = loggamma(d.n + 1)
    for (xi, pi) in zip(x, d.p)
        s -= loggamma(xi + 1)
        if xi > 0
            pi <= 0 && return oftype(float(first(d.p)), -Inf)
            s += xi * log(pi)
        end
    end
    return s
end

function Base.rand(rng::AbstractRNG, d::Multinomial)
    counts = zeros(Int, length(d.p))
    for _ in 1:d.n
        counts[rand(rng, Categorical(collect(d.p)))] += 1
    end
    return counts
end
mean(d::Multinomial) = d.n .* d.p

"""
    LKJCholesky(K, eta)

The Lewandowski, Kurowicka and Joe prior over `K x K` correlation matrices,
written over their Cholesky factors: `p(R) proportional to det(R)^(eta - 1)`.

`eta = 1` is uniform over correlation matrices. Above 1 concentrates towards the
identity, below 1 towards the boundary. Over factors the density picks up the
Jacobian of `R = L L'`, which is why the diagonal entries appear with
dimension-dependent powers:

    log p(L) = sum_{i=2}^{K} (K - i + 2 eta - 2) log L[i,i] + log c(K, eta)

The constant is included, so this can be used for evidence as well as for
sampling. It is checked against Distributions.jl in the test suite.
"""
struct LKJCholesky{T<:Real} <: MultivariateDensity
    K::Int
    eta::T
    function LKJCholesky(K::Int, eta::T) where {T<:Real}
        K >= 2 || throw(ArgumentError("LKJCholesky requires K >= 2, got $K"))
        eta > 0 || throw(DomainError(eta, "eta must be positive"))
        new{T}(K, eta)
    end
end

"""
    lkj_log_constant(K, eta)

Log normalising constant of [`LKJCholesky`](@ref).
"""
function lkj_log_constant(K::Int, eta::Real)
    return (K - 1) * loggamma(eta + (K - 1) / 2) -
           sum(0.5 * k * log(pi) + loggamma(eta + (K - 1 - k) / 2) for k in 1:(K - 1))
end

function logpdf(d::LKJCholesky, L::AbstractMatrix)
    size(L) == (d.K, d.K) || return -Inf
    core = zero(float(promote_type(eltype(L), typeof(d.eta))))
    @inbounds for i in 2:d.K
        L[i, i] > 0 || return oftype(core, -Inf)
        core += (d.K - i + 2 * d.eta - 2) * log(L[i, i])
    end
    return core + lkj_log_constant(d.K, d.eta)
end

"""
    rand(rng, d::LKJCholesky)

Draw a Cholesky factor by the onion method: build the matrix one dimension at a
time, drawing each new row's length from a Beta distribution and its direction
uniformly on the sphere. Exact, and the route through which the prior is
sampled for calibration checks.
"""
function Base.rand(rng::AbstractRNG, d::LKJCholesky)
    K, eta = d.K, float(d.eta)
    L = zeros(Float64, K, K)
    L[1, 1] = 1.0
    K == 1 && return LowerTriangular(L)
    beta = eta + (K - 2) / 2
    r2 = rand(rng, Beta(0.5, beta))
    L[2, 1] = sqrt(r2) * (rand(rng) < 0.5 ? -1.0 : 1.0)
    L[2, 2] = sqrt(1 - L[2, 1]^2)
    for i in 3:K
        beta -= 0.5
        r2 = rand(rng, Beta((i - 1) / 2, beta))
        # a uniform direction on the (i-1)-sphere
        u = randn(rng, i - 1)
        u ./= sqrt(sum(abs2, u))
        for j in 1:(i - 1)
            L[i, j] = sqrt(r2) * u[j]
        end
        L[i, i] = sqrt(max(1 - r2, 0.0))
    end
    return LowerTriangular(L)
end

mean(d::LKJCholesky) = Matrix{Float64}(I, d.K, d.K)

# --------------------------------------------------------------------------
# Cumulative distribution functions and quantiles
#
# Only implemented where the verification suite needs them: comparing sampled
# quantiles against analytic ones, and building calibration checks.
# --------------------------------------------------------------------------

"""
    cdf(d, x)

Cumulative distribution function.
"""
function cdf end

cdf(d::Normal, x::Real) = 0.5 * erfc(-(x - d.mu) / (d.sigma * sqrt(2)))
cdf(d::Beta, x::Real) = x <= 0 ? 0.0 : (x >= 1 ? 1.0 : beta_inc(d.a, d.b, x)[1])
cdf(d::Gamma, x::Real) = x <= 0 ? 0.0 : gamma_inc(d.shape, d.rate * x, 0)[1]
cdf(d::Exponential, x::Real) = x <= 0 ? 0.0 : -expm1(-d.rate * x)
cdf(d::Uniform, x::Real) = clamp((x - d.a) / (d.b - d.a), 0, 1)
cdf(d::LogNormal, x::Real) = x <= 0 ? 0.0 : cdf(Normal(d.mu, d.sigma), log(x))
cdf(d::Poisson, x::Real) = x < 0 ? 0.0 : gamma_inc(floor(x) + 1, d.lambda, 0)[2]

"""
    quantile(d, p)

Inverse cdf by bisection on [`cdf`](@ref), refined to `1e-12` in probability.
Bisection rather than a closed-form inverse keeps this correct for every
continuous density above at the cost of a few dozen cdf evaluations, which the
test suite can afford.
"""
function quantile(d::UnivariateDensity, p::Real)
    0 <= p <= 1 || throw(DomainError(p, "quantile requires p in [0, 1]"))
    lo, hi = _bracket(d, p)
    for _ in 1:200
        mid = (lo + hi) / 2
        if cdf(d, mid) < p
            lo = mid
        else
            hi = mid
        end
        hi - lo < 1e-12 * max(1, abs(lo)) && break
    end
    return (lo + hi) / 2
end

function _bracket(d::UnivariateDensity, p::Real)
    lo, hi = _support(d)
    if !isfinite(lo)
        lo = -1.0
        while cdf(d, lo) > p
            lo *= 2
        end
    end
    if !isfinite(hi)
        hi = 1.0
        while cdf(d, hi) < p
            hi *= 2
        end
    end
    return float(lo), float(hi)
end

_support(::Normal) = (-Inf, Inf)
_support(::LogNormal) = (0.0, Inf)
_support(::Beta) = (0.0, 1.0)
_support(::Gamma) = (0.0, Inf)
_support(::Exponential) = (0.0, Inf)
_support(d::Uniform) = (float(d.a), float(d.b))
