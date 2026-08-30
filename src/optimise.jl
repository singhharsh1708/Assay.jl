# The mode of a posterior, and the Gaussian that matches its curvature there.
#
# Not every question needs a posterior. A mode is useful for initialisation, for
# a quick answer, and as a check that sampling found the region it should have.
# The Laplace approximation turns the mode into a cheap posterior, and the
# importance weights of that approximation against the real thing say how cheap.
#
# One thing here is a trap rather than a detail. "The MAP estimate" names two
# different points, because a mode is not preserved by a change of variables.
# Optimising the log density this package hands a sampler finds the mode in the
# unconstrained space, Jacobian included, and mapping that point back is not the
# mode of the constrained posterior. For a Beta(a, b) posterior the two are
# exactly `(a - 1) / (a + b - 2)` and `a / (a + b)`, the second of which is the
# posterior mean. Neither is wrong; they answer different questions, and code
# that does not say which one it computed is the problem. So `space` is a
# required decision with a default rather than a hidden one.

"""
    ModeResult

A mode of the posterior: the point in the unconstrained space, the same point as
the model's own `NamedTuple`, the log density there, and whether the optimiser
converged.

`space` records which mode this is. `:unconstrained` includes the Jacobian of
the transform, which is the density a sampler targets and the one the Laplace
approximation needs. `:constrained` excludes it, which is what "the MAP
estimate" usually means when a reader says it out loud.
"""
struct ModeResult{T}
    y::Vector{Float64}
    theta::T
    log_density::Float64
    gradient_norm::Float64
    iterations::Int
    converged::Bool
    space::Symbol
end

function Base.show(io::IO, r::ModeResult)
    @printf(io, "ModeResult(%s space, log density %.4f, %d iterations%s, |g| = %.2e)",
            r.space, r.log_density, r.iterations,
            r.converged ? "" : ", NOT converged", r.gradient_norm)
end

# The objective without the Jacobian correction, which is the constrained-space
# log joint evaluated at the point the unconstrained coordinates map to.
_log_density_no_jacobian(model::AbstractModel, y::AbstractVector) =
    logdensity(model, y) - constrain(model, y)[2]

"""
    find_mode(model; space = :unconstrained, init, backend, max_iterations, g_tol)

Maximise the log density by BFGS, returning a [`ModeResult`](@ref).

`space` chooses which mode, and the two are different points: see
[`ModeResult`](@ref). The optimisation always runs in the unconstrained space,
because that is where the problem is unbounded and a step is always legal; only
the objective changes.

The line search is backtracking Armijo, and the inverse Hessian estimate is
reset to the identity whenever the proposed direction is not an ascent
direction, which is what keeps a bad curvature estimate from ending the run
rather than costing an iteration.
"""
function find_mode(model::AbstractModel; space::Symbol = :unconstrained, init = nothing,
                   backend::ADBackend = ForwardDiffAD(), max_iterations::Int = 1000,
                   g_tol::Real = 1e-8, rng::AbstractRNG = Random.default_rng())
    space in (:unconstrained, :constrained) ||
        throw(ArgumentError("space must be :unconstrained or :constrained, got $space"))
    y0 = init === nothing ? random_init(rng, model) : collect(float.(init))
    objective = space === :unconstrained ? (y -> logdensity(model, y)) :
                (y -> _log_density_no_jacobian(model, y))
    f_and_g = y -> logdensity_and_gradient(backend, objective, y)

    y, fy, g, iters, converged = _bfgs_ascent(f_and_g, y0, max_iterations, float(g_tol))
    theta, _ = constrain(model, y)
    return ModeResult(y, theta, fy, sqrt(sum(abs2, g)), iters, converged, space)
end

# BFGS on the minimisation of the negative log density, written in terms of the
# ascent so the signs stay readable. `H` is the inverse Hessian estimate of that
# minimisation problem, so `H * g` is the ascent direction.
function _bfgs_ascent(f_and_g, y0::AbstractVector, max_iterations::Int, g_tol::Float64)
    y = collect(float.(y0))
    fy, g = f_and_g(y)
    isfinite(fy) || throw(NonFiniteDensityError(y, fy, :logdensity))
    n = length(y)
    H = Matrix{Float64}(I, n, n)
    iters = 0
    for it in 1:max_iterations
        iters = it
        sqrt(sum(abs2, g)) < g_tol && return y, fy, g, iters, true

        d = H * g
        sum(d .* g) <= 0 && (H = Matrix{Float64}(I, n, n); d = copy(g))

        alpha = 1.0
        stepped = false
        ynew, fnew, gnew = y, fy, g
        for _ in 1:60
            ynew = y .+ alpha .* d
            fnew, gnew = f_and_g(ynew)
            if isfinite(fnew) && fnew >= fy + 1e-4 * alpha * sum(g .* d)
                stepped = true
                break
            end
            alpha /= 2
        end
        # A line search that cannot find an uphill step has either arrived or
        # run into a cliff. Both are the end of the run; only the first is
        # convergence.
        stepped || return y, fy, g, iters, sqrt(sum(abs2, g)) < sqrt(g_tol)

        s = ynew .- y
        yvec = -(gnew .- g)                 # gradient difference of the minimisation
        sy = sum(s .* yvec)
        if sy > 1e-12
            rho = 1 / sy
            V = Matrix{Float64}(I, n, n) .- rho .* (s * yvec')
            H = V * H * V' .+ rho .* (s * s')
        end
        y, fy, g = ynew, fnew, gnew
    end
    return y, fy, g, iters, false
end

"""
    LaplaceResult

The Gaussian that matches the posterior's curvature at its mode, in the
unconstrained space, together with a measurement of how well it does.

`khat` is the Pareto shape of the importance weights of this approximation
against the real posterior, computed from draws. Below 0.5 the approximation is
good enough to correct by importance sampling; above 0.7 it is not to be trusted
whatever the covariance says. An approximation that reports no diagnostic is
asking to be believed.

`log_evidence` is the Laplace estimate of the log normalising constant, exact for
a Gaussian posterior and an approximation everywhere else.
"""
struct LaplaceResult{M<:AbstractModel}
    model::M
    mode::ModeResult
    covariance::Matrix{Float64}
    factor::Matrix{Float64}
    log_evidence::Float64
    khat::Float64
    n_check::Int
end

function Base.show(io::IO, r::LaplaceResult)
    verdict = r.khat < 0.5 ? "good" : (r.khat < 0.7 ? "usable" : "not to be trusted")
    @printf(io, "LaplaceResult(%d dimensions, log evidence %.4f, k = %.3f, %s)",
            size(r.covariance, 1), r.log_evidence, r.khat, verdict)
end

"""
    laplace(model; mode = nothing, n_check = 2000, rng, kwargs...)

Laplace approximation to the posterior: a Gaussian centred at the unconstrained
mode with covariance the inverse of the negative Hessian there.

The Hessian is taken by forward mode differentiation of the gradient. If it is
not negative definite the point is not a maximum, and that is raised rather than
patched, because a covariance built from an indefinite Hessian is a number with
no meaning attached.

`n_check` draws are used to estimate the Pareto shape of the importance weights
against the true posterior, which is what turns the approximation into a
measured one. Set it to zero to skip that.

Keyword arguments other than `mode`, `n_check` and `rng` go to [`find_mode`](@ref).
"""
function laplace(model::AbstractModel; mode::Union{Nothing,ModeResult} = nothing,
                 n_check::Int = 2000, rng::AbstractRNG = Random.default_rng(), kwargs...)
    m = mode === nothing ? find_mode(model; rng = rng, kwargs...) : mode
    m.space === :unconstrained ||
        throw(ArgumentError("the Laplace approximation is built at the unconstrained mode, " *
                            "which is where the density it approximates lives; got a " *
                            "$(m.space) mode"))
    d = dimension(model)
    Hs = ForwardDiff.hessian(y -> logdensity(model, y), m.y)
    negH = Symmetric(-(Hs .+ Hs') ./ 2)
    chol = LinearAlgebra.cholesky(negH; check = false)
    issuccess(chol) ||
        throw(ArgumentError("the Hessian at this point is not negative definite, so the point " *
                            "is not a maximum and there is no Gaussian to build. Check that " *
                            "`find_mode` converged: it reported |g| = " *
                            "$(round(m.gradient_norm; sigdigits = 3))"))
    Sigma = Matrix(inv(negH))
    L = Matrix(LinearAlgebra.cholesky(Symmetric(Sigma)).L)
    log_evidence = m.log_density + d / 2 * LOG2PI - LinearAlgebra.logdet(chol) / 2

    khat = NaN
    if n_check > 0
        approx = MvNormalCholesky(m.y, L)
        ratios = Vector{Float64}(undef, n_check)
        for i in 1:n_check
            y = rand(rng, approx)
            ratios[i] = logdensity(model, y) - logpdf(approx, y)
        end
        khat = psis(ratios).k
    end
    return LaplaceResult(model, m, Sigma, L, log_evidence, khat, n_check)
end

"""
    posterior_samples(r::LaplaceResult, n; rng)

Draw from the Laplace approximation, returned as [`Chains`](@ref) so every
diagnostic and summary works on it unchanged.
"""
function posterior_samples(r::LaplaceResult, n::Int; rng::AbstractRNG = Random.default_rng())
    d = dimension(r.model)
    P = flat_dimension(r.model)
    value = Array{Float64,3}(undef, n, P, 1)
    raw = Array{Float64,3}(undef, n, d, 1)
    approx = MvNormalCholesky(r.mode.y, r.factor)
    for i in 1:n
        y = rand(rng, approx)
        value[i, :, 1] = flatten_draw(r.model, y)
        raw[i, :, 1] = y
    end
    info = Dict{Symbol,Any}(:sampler => r, :n_warmup => 0, :thin => 1,
                            :warmup_kept => false, :chain_info => Any[])
    return Chains(value, parameter_names(r.model), Dict{Symbol,Array{Float64,2}}(), info, raw)
end
