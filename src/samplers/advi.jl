# Automatic differentiation variational inference.
#
# The variational family lives on the *unconstrained* space, which is what makes
# a single Gaussian family sufficient for models with positive, bounded and
# simplex parameters: the transform layer carries the constraint, and the
# Jacobian term is already inside `logdensity`. This is the same reason the
# samplers need no per-constraint code, and it is the strongest argument for
# putting the transform layer under the model rather than inside each algorithm.
#
# The ELBO is maximised by plain stochastic gradient ascent with Adam, using the
# reparameterisation gradient: draw z ~ N(0, I), push it through the location
# scale map, and differentiate through both the map and the log density. The
# score-function estimator would work without differentiating the model, at
# perhaps a hundred times the variance.

"""
    VariationalFamily

The shape of `q`. Both members here are location-scale families, so both admit
the reparameterisation gradient.
"""
abstract type VariationalFamily end

"""
    MeanField()

`q(y) = N(mu, diag(exp(omega)^2))`. `2d` parameters. Cannot represent posterior
correlation, and therefore systematically understates the variance of correlated
targets - a fact the results document measures rather than glosses over.
"""
struct MeanField <: VariationalFamily end

"""
    FullRank()

`q(y) = N(mu, L L')` with `L` lower triangular, its diagonal parameterised in
logs. `d + d(d+1)/2` parameters; captures correlation at quadratic cost.
"""
struct FullRank <: VariationalFamily end

n_variational_params(::MeanField, d::Int) = 2d
n_variational_params(::FullRank, d::Int) = d + div(d * (d + 1), 2)

"""
    transform_sample(family, params, z, d)

Map a standard normal draw `z` to the variational sample, i.e. `mu + L z`.
"""
function transform_sample(::MeanField, params::AbstractVector, z::AbstractVector, d::Int)
    mu = view(params, 1:d)
    omega = view(params, (d + 1):(2d))
    return mu .+ exp.(omega) .* z
end

function transform_sample(::FullRank, params::AbstractVector, z::AbstractVector, d::Int)
    mu = view(params, 1:d)
    T = promote_type(eltype(params), eltype(z))
    out = Vector{T}(undef, d)
    k = d
    for i in 1:d
        acc = mu[i]
        for j in 1:i
            k += 1
            lij = i == j ? exp(params[k]) : params[k]     # log diagonal keeps L positive definite
            acc += lij * z[j]
        end
        out[i] = acc
    end
    return out
end

"""
    entropy(family, params, d)

Differential entropy of `q`, `sum(log scale) + d/2 (1 + log 2pi)`.
"""
function entropy(::MeanField, params::AbstractVector, d::Int)
    return sum(view(params, (d + 1):(2d))) + d / 2 * (1 + LOG2PI)
end

function entropy(::FullRank, params::AbstractVector, d::Int)
    s = zero(eltype(params))
    k = d
    for i in 1:d
        for j in 1:i
            k += 1
            i == j && (s += params[k])
        end
    end
    return s + d / 2 * (1 + LOG2PI)
end

"""
    ADVI(; family = MeanField(), n_samples = 1, n_iterations = 20_000,
           step_size = NaN, elbo_samples = 500, check_every = 100,
           rel_tol = 1e-3, patience = 20, backend = ForwardDiffAD())

Mean-field (or full-rank) ADVI with Adam.

The step size decays as `step_size / (1 + t / decay)`, and that matters more
than its starting value. With a constant step size there is no good default:
measured across four targets, the best of `0.5, 0.1, 0.05, 0.01, 0.001` was
`0.01` for a target whose scales differ by a factor of a million and `0.5` for a
model with five thousand observations, where `0.001` was wrong by 350 percent.

With decay the same four targets are all within 7 percent for any starting value
from 1.0 down to 0.1, and within 4 percent at the default of 0.2. Stan's
approach of trying several candidates on short runs was implemented, measured
and removed: a short trial rewards a step size that covers ground early and then
plateaus, and it lost to decay on every target.

Convergence follows Stan's rule rather than a simple "no improvement" counter:
the change in the ELBO relative to `max(|ELBO|, 1)` is kept in a circular
buffer of the last `patience` checks, and the run stops when either the mean or
the median of that buffer falls below `rel_tol`. A plain best-so-far counter stops far too early on
this objective, because the ELBO estimate is itself a noisy Monte Carlo
quantity - measured on a correlated bivariate normal, the naive rule stopped
with a covariance error of 0.14 where this rule reaches 0.02.
"""
Base.@kwdef struct ADVI{F<:VariationalFamily,B<:ADBackend}
    family::F = MeanField()
    n_samples::Int = 1
    n_iterations::Int = 20_000
    step_size::Float64 = 0.2
    decay::Float64 = 2000.0
    beta1::Float64 = 0.9
    beta2::Float64 = 0.999
    epsilon::Float64 = 1e-8
    elbo_samples::Int = 500
    check_every::Int = 100
    rel_tol::Float64 = 1e-3
    patience::Int = 20
    averaging::Float64 = 0.99
    init_scale::Float64 = 0.1
    backend::B = ForwardDiffAD()
end

"""
    VIResult

The fitted variational approximation: its parameters, the ELBO trace, and
whether the early stopping criterion fired. `posterior_samples` turns it into
the same [`Chains`](@ref) object the MCMC samplers produce, so every diagnostic
and every comparison table works unchanged.
"""
struct VIResult{F<:VariationalFamily,M<:AbstractModel}
    family::F
    model::M
    params::Vector{Float64}
    elbo_trace::Vector{Float64}
    elbo_iterations::Vector{Int}
    elbo_final::Float64
    converged::Bool
    iterations::Int
    time_seconds::Float64
end

"""
    variational_mean(result)

Mean of `q` in the unconstrained space.
"""
variational_mean(r::VIResult) = r.params[1:dimension(r.model)]

"""
    variational_scale(result)

Marginal standard deviations of `q` in the unconstrained space.
"""
function variational_scale(r::VIResult)
    d = dimension(r.model)
    if r.family isa MeanField
        return exp.(r.params[(d + 1):(2d)])
    else
        L = variational_factor(r)
        return [sqrt(sum(abs2, view(L, i, :))) for i in 1:d]
    end
end

"""
    variational_factor(result)

Lower triangular factor `L` of the full-rank family, with `Sigma = L L'`.
"""
function variational_factor(r::VIResult)
    r.family isa FullRank || throw(ArgumentError("only a full-rank family has a factor"))
    d = dimension(r.model)
    L = zeros(d, d)
    k = d
    for i in 1:d, j in 1:i
        k += 1
        L[i, j] = i == j ? exp(r.params[k]) : r.params[k]
    end
    return L
end

"""
    elbo(model, family, params, rng, n_samples)

Monte Carlo estimate of `E_q[log p] + H[q]` with fresh draws.
"""
function elbo(model::AbstractModel, family::VariationalFamily, params::AbstractVector,
              rng::AbstractRNG, n_samples::Int)
    d = dimension(model)
    acc = 0.0
    for _ in 1:n_samples
        z = randn(rng, d)
        y = transform_sample(family, params, z, d)
        acc += logdensity(model, y)
    end
    return acc / n_samples + entropy(family, params, d)
end

"""
    elbo_fixed(model, family, params, zs)

ELBO estimate on a *fixed* set of standard normal draws. Used for the
convergence check: with fresh draws each time, consecutive estimates differ by
their own Monte Carlo noise, which for a two-dimensional normal is around 0.05
nats and swamps any sensible tolerance. Common random numbers make the check
measure movement of the variational parameters instead.
"""
function elbo_fixed(model::AbstractModel, family::VariationalFamily, params::AbstractVector,
                    zs::Vector{Vector{Float64}})
    d = dimension(model)
    acc = 0.0
    for z in zs
        acc += logdensity(model, transform_sample(family, params, z, d))
    end
    return acc / length(zs) + entropy(family, params, d)
end

"""
    sample(model, advi::ADVI; rng, init)

Fit the variational approximation and return a [`VIResult`](@ref).
"""
function sample(model::AbstractModel, spl::ADVI; rng::AbstractRNG = Random.default_rng(),
                init = nothing, progress::Bool = false, progress_interval::Real = 10.0)
    t0 = time()
    reporter = ProgressReporter(spl.n_iterations; on = progress, interval = progress_interval,
                                what = "optimising the variational objective")
    d = dimension(model)
    np = n_variational_params(spl.family, d)
    params = zeros(np)
    if init !== nothing
        params[1:d] .= init
    else
        params[1:d] .= random_init(rng, model)
    end
    if spl.family isa MeanField
        params[(d + 1):(2d)] .= log(spl.init_scale)
    else
        k = d
        for i in 1:d, j in 1:i
            k += 1
            i == j && (params[k] = log(spl.init_scale))
        end
    end

    m = zeros(np)
    v = zeros(np)
    params_avg = copy(params)
    trace = Float64[]
    trace_iters = Int[]
    rel_changes = Float64[]
    check_zs = [randn(rng, d) for _ in 1:spl.elbo_samples]
    converged = false
    iter = 0

    for t in 1:spl.n_iterations
        iter = t
        tick!(reporter)
        # Reparameterisation gradient: the standard normal draws are fixed
        # inside the gradient evaluation, which is what makes this estimator
        # low variance compared to the score-function form.
        zs = [randn(rng, d) for _ in 1:spl.n_samples]
        objective = function (p)
            acc = zero(eltype(p))
            for z in zs
                acc += logdensity(model, transform_sample(spl.family, p, z, d))
            end
            return acc / length(zs) + entropy(spl.family, p, d)
        end
        _, g = logdensity_and_gradient(spl.backend, objective, params)
        all(isfinite, g) || continue

        @inbounds for i in 1:np
            m[i] = spl.beta1 * m[i] + (1 - spl.beta1) * g[i]
            v[i] = spl.beta2 * v[i] + (1 - spl.beta2) * g[i]^2
            mhat = m[i] / (1 - spl.beta1^t)
            vhat = v[i] / (1 - spl.beta2^t)
            eta_t = spl.decay > 0 ? spl.step_size / (1 + t / spl.decay) : spl.step_size
            params[i] += eta_t * mhat / (sqrt(vhat) + spl.epsilon)
        end
        w = min(spl.averaging, 1 - 1 / t)      # plain mean until the window fills
        @inbounds for i in 1:np
            params_avg[i] = w * params_avg[i] + (1 - w) * params[i]
        end

        if t % spl.check_every == 0
            e = elbo_fixed(model, spl.family, params_avg, check_zs)
            if !isempty(trace)
                prev = trace[end]
                # Scaled by max(|ELBO|, 1): a pure relative change is meaningless
                # when the ELBO passes near zero, which it does whenever the
                # target is close to standard normal.
                push!(rel_changes, abs(e - prev) / max(abs(prev), 1.0))
                length(rel_changes) > spl.patience && popfirst!(rel_changes)
            end
            push!(trace, e)
            push!(trace_iters, t)
            if length(rel_changes) == spl.patience &&
               (Statistics.mean(rel_changes) < spl.rel_tol ||
                Statistics.median(rel_changes) < spl.rel_tol)
                converged = true
                break
            end
        end
    end

    finish!(reporter)

    final = elbo(model, spl.family, params_avg, rng, 4 * spl.elbo_samples)
    return VIResult(spl.family, model, params_avg, trace, trace_iters, final, converged, iter,
                    time() - t0)
end

"""
    elbo_with_error(result; rng, n_samples = 20_000, batches = 20)

Estimate of the ELBO at the fitted parameters together with its Monte Carlo
standard error, computed from `batches` independent blocks. The single-number
ELBO carries enough noise (around 0.05 nats for a two-dimensional target at
2000 draws) to appear to violate its own bound, so anywhere the bound itself is
the claim being made, this is the function to use.
"""
function elbo_with_error(r::VIResult; rng::AbstractRNG = Random.default_rng(),
                         n_samples::Int = 20_000, batches::Int = 20)
    per = max(div(n_samples, batches), 1)
    est = [elbo(r.model, r.family, r.params, rng, per) for _ in 1:batches]
    return Statistics.mean(est), Statistics.std(est) / sqrt(batches)
end

"""
    posterior_samples(result, n; rng)

Draw `n` samples from `q`, mapped to the constrained space and returned as
[`Chains`](@ref) so the same summaries and comparisons apply as for MCMC output.
The draws are independent, so an effective sample size equal to `n` here is
expected and says nothing about how well `q` approximates the posterior.
"""
function posterior_samples(r::VIResult, n::Int; rng::AbstractRNG = Random.default_rng())
    d = dimension(r.model)
    P = flat_dimension(r.model)
    value = Array{Float64,3}(undef, n, P, 1)
    raw = Array{Float64,3}(undef, n, d, 1)
    for i in 1:n
        z = randn(rng, d)
        y = transform_sample(r.family, r.params, z, d)
        value[i, :, 1] = flatten_draw(r.model, y)
        raw[i, :, 1] = y
    end
    info = Dict{Symbol,Any}(:sampler => r, :time_seconds => r.time_seconds, :n_warmup => 0,
                            :thin => 1, :warmup_kept => false, :chain_info => Any[])
    return Chains(value, parameter_names(r.model), Dict{Symbol,Array{Float64,2}}(), info, raw)
end

function Base.show(io::IO, r::VIResult)
    fam = r.family isa MeanField ? "mean field" : "full rank"
    @printf(io, "VIResult(%s, %d iterations%s, ELBO = %.4f)\n", fam, r.iterations,
            r.converged ? ", converged" : "", r.elbo_final)
end
