# Sequential Monte Carlo over a tempered sequence of targets.
#
#     pi_t(theta) proportional to prior(theta) * likelihood(theta)^beta_t,
#     0 = beta_0 < beta_1 < ... < beta_T = 1
#
# Three design points worth defending:
#
#   1. Tempering needs the prior and the likelihood separately, which the
#      single-log-density `Model` deliberately does not provide. Rather than
#      widen `Model`, a `TemperedModel` pairs a prior model with a likelihood
#      and *produces* an ordinary `Model` at each inverse temperature. Every
#      MCMC sampler in the package then works as a rejuvenation kernel with no
#      changes at all.
#   2. The schedule is chosen adaptively by bisection on the effective sample
#      size, so the user picks a tolerance rather than a schedule.
#   3. The normalising constant falls out of the same weights. That is what
#      makes SMC verifiable: `log Z` is compared against the analytic evidence
#      of the conjugate models, which no amount of plausible-looking particle
#      cloud can fake.

"""
    TemperedModel(prior, loglikelihood; prior_rand = nothing)

A prior `Model` (whose log joint is the log prior alone) together with a
log likelihood function of the constrained parameters. `prior_rand(rng)` should
return a `NamedTuple` drawn from the prior; supply it whenever the prior can be
sampled directly, which is the case for every model in `conjugate.jl`.
"""
struct TemperedModel{M<:AbstractModel,F,R}
    prior::M
    loglik::F
    prior_rand::R
end
TemperedModel(prior::AbstractModel, loglik; prior_rand = nothing) =
    TemperedModel(prior, loglik, prior_rand)

"""
    at(tm::TemperedModel, beta)

The `Model` whose log density is `log prior + beta * log likelihood`.
"""
function at(tm::TemperedModel{<:Model}, beta::Real)
    return Model(NamedTuple{tm.prior.names}(tm.prior.transforms),
                 theta -> tm.prior.logjoint(theta) + beta * tm.loglik(theta))
end

"""
    at(tm::TemperedModel{<:LogDensityModel}, beta)

The wrapped-log-density case. There are no named transforms to rebuild, so the
tempered target is assembled as another wrapped log density.
"""
function at(tm::TemperedModel{<:LogDensityModel}, beta::Real)
    prior = tm.prior
    theta_of = y -> (x = collect(float.(y)),)
    return LogDensityModel(y -> logdensity(prior, y) + beta * tm.loglik(theta_of(y)),
                           dimension(prior); names = parameter_names(prior),
                           backend = prior.backend)
end

"""
    loglik_at(tm, y)

Log likelihood at an unconstrained point, used to reweight particles.
"""
function loglik_at(tm::TemperedModel, y::AbstractVector)
    theta, _ = constrain(tm.prior, y)
    return tm.loglik(theta)
end

dimension(tm::TemperedModel) = dimension(tm.prior)

# --------------------------------------------------------------------------
# Resampling
# --------------------------------------------------------------------------

"""
    AbstractResampler

Maps normalised weights to a vector of ancestor indices. Every scheme here is
unbiased, `E[count of i] = N * w_i`; they differ only in variance, which is
what `test/test_smc.jl` measures rather than assumes.
"""
abstract type AbstractResampler end

"""
    MultinomialResampling()

Independent draws from the weight distribution. The simplest and the noisiest.
"""
struct MultinomialResampling <: AbstractResampler end

"""
    StratifiedResampling()

One uniform draw per stratum of width `1/N`.
"""
struct StratifiedResampling <: AbstractResampler end

"""
    SystematicResampling()

A single uniform draw shared by every stratum. Lowest variance of the three for
smooth weight profiles, and the default.
"""
struct SystematicResampling <: AbstractResampler end

"""
    ResidualResampling()

Deterministic allocation of the integer parts of `N * w`, with the remainder
drawn multinomially.
"""
struct ResidualResampling <: AbstractResampler end

"""
    resample(rng, scheme, w)

Ancestor indices drawn from normalised weights `w`, one per particle.
"""
function resample end

function resample(rng::AbstractRNG, ::MultinomialResampling, w::AbstractVector)
    N = length(w)
    edges = cumsum(w)
    edges[end] = 1.0
    return [searchsortedfirst(edges, rand(rng)) for _ in 1:N]
end

function _inverse_cdf(edges::AbstractVector, us::AbstractVector)
    N = length(edges)
    idx = Vector{Int}(undef, length(us))
    j = 1
    for (k, u) in enumerate(us)
        while j < N && u > edges[j]
            j += 1
        end
        idx[k] = j
    end
    return idx
end

function resample(rng::AbstractRNG, ::StratifiedResampling, w::AbstractVector)
    N = length(w)
    edges = cumsum(w)
    edges[end] = 1.0
    us = [(k - 1 + rand(rng)) / N for k in 1:N]
    return _inverse_cdf(edges, us)
end

function resample(rng::AbstractRNG, ::SystematicResampling, w::AbstractVector)
    N = length(w)
    edges = cumsum(w)
    edges[end] = 1.0
    u0 = rand(rng) / N
    us = [u0 + (k - 1) / N for k in 1:N]
    return _inverse_cdf(edges, us)
end

function resample(rng::AbstractRNG, ::ResidualResampling, w::AbstractVector)
    N = length(w)
    counts = floor.(Int, N .* w)
    idx = Int[]
    sizehint!(idx, N)
    for i in 1:N, _ in 1:counts[i]
        push!(idx, i)
    end
    remaining = N - length(idx)
    if remaining > 0
        resid = N .* w .- counts
        resid ./= sum(resid)
        edges = cumsum(resid)
        edges[end] = 1.0
        for _ in 1:remaining
            push!(idx, searchsortedfirst(edges, rand(rng)))
        end
    end
    return idx
end

"""
    ess_weights(logw)

Effective sample size of a set of log weights, `1 / sum(w^2)` after
normalisation. This is the quantity the tempering schedule and the resampling
trigger are both defined against.
"""
function ess_weights(logw::AbstractVector)
    w = similar(logw, float(eltype(logw)))
    softmax!(w, logw)
    return 1 / sum(abs2, w)
end

# --------------------------------------------------------------------------
# The sampler
# --------------------------------------------------------------------------

"""
    AdaptiveRandomWalk(; n_steps = 3, scale = 2.38)

Rejuvenation kernel that rebuilds a random walk proposal from the current
particle covariance at every temperature. `scale / sqrt(d)` is the usual optimal
scaling for a Gaussian target.
"""
Base.@kwdef struct AdaptiveRandomWalk
    n_steps::Int = 3
    scale::Float64 = 2.38
end

"""
    SMC(; n_particles = 1000, resampler = SystematicResampling(),
          ess_threshold = 0.5, target_ess = 0.5, kernel = AdaptiveRandomWalk(),
          max_steps = 10_000)

Sequential Monte Carlo with adaptive tempering. `target_ess` is the fraction of
particles the next temperature is chosen to retain; `ess_threshold` is the
fraction below which the cloud is resampled.
"""
Base.@kwdef struct SMC{R<:AbstractResampler,K}
    n_particles::Int = 1000
    resampler::R = SystematicResampling()
    ess_threshold::Float64 = 0.5
    target_ess::Float64 = 0.5
    kernel::K = AdaptiveRandomWalk()
    max_steps::Int = 10_000
end

"""
    SMCResult

Particles (in constrained space), their normalised weights, the estimated log
normalising constant, and the diagnostics of the run: the temperature ladder,
the effective sample size before each resampling step, and the acceptance rate
of the rejuvenation kernel.
"""
struct SMCResult
    particles::Matrix{Float64}          # n_particles x n_flat_parameters
    unconstrained::Matrix{Float64}      # n_particles x dimension
    weights::Vector{Float64}
    names::Vector{Symbol}
    logZ::Float64
    betas::Vector{Float64}
    ess_trace::Vector{Float64}
    resampled::Vector{Bool}
    accept_trace::Vector{Float64}
    time_seconds::Float64
end

ndraws(r::SMCResult) = size(r.particles, 1)

function Base.show(io::IO, r::SMCResult)
    @printf(io, "SMCResult(%d particles, %d temperatures, log Z = %.4f)\n",
            size(r.particles, 1), length(r.betas), r.logZ)
    @printf(io, "%-12s %10s %10s\n", "parameter", "mean", "std")
    for (j, name) in enumerate(r.names)
        m = sum(r.weights .* r.particles[:, j])
        s = sqrt(max(sum(r.weights .* (r.particles[:, j] .- m) .^ 2), 0.0))
        @printf(io, "%-12s %10.4f %10.4f\n", String(name), m, s)
    end
end

"""
    weighted_mean(result, name)

Weighted posterior mean of one parameter.
"""
function weighted_mean(r::SMCResult, name::Symbol)
    j = findfirst(==(name), r.names)
    j === nothing && throw(KeyError(name))
    return sum(r.weights .* r.particles[:, j])
end

"""
    weighted_std(result, name)
"""
function weighted_std(r::SMCResult, name::Symbol)
    j = findfirst(==(name), r.names)
    j === nothing && throw(KeyError(name))
    m = weighted_mean(r, name)
    return sqrt(sum(r.weights .* (r.particles[:, j] .- m) .^ 2))
end

"""
    weighted_quantile(result, name, p)

Weighted quantile by inversion of the weighted empirical distribution.
"""
function weighted_quantile(r::SMCResult, name::Symbol, p::Real)
    j = findfirst(==(name), r.names)
    j === nothing && throw(KeyError(name))
    x = r.particles[:, j]
    perm = sortperm(x)
    cw = cumsum(r.weights[perm])
    k = searchsortedfirst(cw, p)
    return x[perm[clamp(k, 1, length(x))]]
end

"""
    next_beta(loglik, logw, beta, target)

Bisection for the next inverse temperature: the largest step that keeps the
effective sample size at `target * N`. Returns `1.0` once the full likelihood
can be taken in one step.
"""
function next_beta(loglik::AbstractVector, logw::AbstractVector, beta::Float64, target::Float64)
    N = length(loglik)
    goal = target * N
    ess_at = b -> ess_weights(logw .+ (b - beta) .* loglik)
    ess_at(1.0) >= goal && return 1.0
    lo, hi = beta, 1.0
    for _ in 1:100
        mid = (lo + hi) / 2
        if ess_at(mid) >= goal
            lo = mid
        else
            hi = mid
        end
        hi - lo < 1e-10 && break
    end
    # Never return the current temperature: that would stall the run.
    return max(lo, beta + 1e-8)
end

"""
    sample(tm::TemperedModel, smc::SMC; rng, init)

Run sequential Monte Carlo and return an [`SMCResult`](@ref).
"""
function sample(tm::TemperedModel, spl::SMC; rng::AbstractRNG = Random.default_rng(), init = nothing)
    t0 = time()
    N = spl.n_particles
    d = dimension(tm)
    ys = Matrix{Float64}(undef, N, d)
    if init !== nothing
        for i in 1:N
            ys[i, :] = init(rng)
        end
    elseif tm.prior_rand !== nothing
        for i in 1:N
            ys[i, :] = unconstrain(tm.prior, tm.prior_rand(rng))
        end
    else
        throw(ArgumentError("supply `init` or build the TemperedModel with `prior_rand`"))
    end

    logw = zeros(N)
    loglik = [loglik_at(tm, view(ys, i, :)) for i in 1:N]
    beta = 0.0
    logZ = 0.0
    betas = [0.0]
    ess_trace = Float64[]
    resampled = Bool[]
    accept_trace = Float64[]

    steps = 0
    while beta < 1.0 && steps < spl.max_steps
        steps += 1
        newbeta = next_beta(loglik, logw, beta, spl.target_ess)
        dbeta = newbeta - beta

        # Normalising constant increment, using the normalised previous weights.
        wprev = similar(logw)
        softmax!(wprev, logw)
        logZ += logsumexp(log.(wprev) .+ dbeta .* loglik)

        logw .+= dbeta .* loglik
        logw .-= logsumexp(logw)
        beta = newbeta
        push!(betas, beta)

        e = ess_weights(logw)
        push!(ess_trace, e)
        did_resample = e < spl.ess_threshold * N
        if did_resample
            w = similar(logw)
            softmax!(w, logw)
            idx = resample(rng, spl.resampler, w)
            ys = ys[idx, :]
            loglik = loglik[idx]
            fill!(logw, -log(N))
        end
        push!(resampled, did_resample)

        # Rejuvenate at the new temperature. Any AbstractSampler works here;
        # the default rebuilds a random walk from the particle covariance.
        acc = rejuvenate!(rng, ys, tm, beta, spl.kernel, logw)
        push!(accept_trace, acc)
        loglik = [loglik_at(tm, view(ys, i, :)) for i in 1:N]
    end

    w = similar(logw)
    softmax!(w, logw)
    parts = Matrix{Float64}(undef, N, flat_dimension(tm.prior))
    for i in 1:N
        parts[i, :] = flatten_draw(tm.prior, view(ys, i, :))
    end
    return SMCResult(parts, ys, w, parameter_names(tm.prior), logZ, betas, ess_trace,
                     resampled, accept_trace, time() - t0)
end

"""
    rejuvenate!(rng, ys, tm, beta, kernel, logw)

Move every particle with an MCMC kernel that leaves the current tempered target
invariant, and return the mean acceptance rate. Rejuvenation is what stops the
particle set from degenerating to a handful of distinct values.
"""
function rejuvenate!(rng::AbstractRNG, ys::Matrix{Float64}, tm::TemperedModel, beta::Float64,
                     k::AdaptiveRandomWalk, logw::AbstractVector)
    N, d = size(ys)
    model = at(tm, beta)
    w = similar(logw)
    softmax!(w, logw)
    mu = vec(sum(w .* ys; dims = 1))
    C = zeros(d, d)
    for i in 1:N
        z = ys[i, :] .- mu
        C .+= w[i] .* (z * z')
    end
    C .+= 1e-10 * Matrix(I, d, d)
    L = Matrix(cholesky(Symmetric(C)).L)
    scale = k.scale / sqrt(d)
    naccept = 0
    ntotal = 0
    for i in 1:N
        y = ys[i, :]
        lp = logdensity(model, y)
        for _ in 1:k.n_steps
            prop = y .+ scale .* (L * randn(rng, d))
            lpp = logdensity(model, prop)
            ntotal += 1
            if isfinite(lpp) && log(rand(rng)) < lpp - lp
                y, lp = prop, lpp
                naccept += 1
            end
        end
        ys[i, :] = y
    end
    return ntotal == 0 ? 0.0 : naccept / ntotal
end

function rejuvenate!(rng::AbstractRNG, ys::Matrix{Float64}, tm::TemperedModel, beta::Float64,
                     k::AbstractSampler, logw::AbstractVector)
    N, d = size(ys)
    model = at(tm, beta)
    accept = 0.0
    count = 0
    for i in 1:N
        state = init_state(rng, model, k, view(ys, i, :); n_warmup = 0)
        local y
        for _ in 1:3
            y, st = step!(rng, model, k, state, false)
            accept += get(st, :accept_prob, NaN)
            count += 1
        end
        ys[i, :] = y
    end
    return count == 0 ? 0.0 : accept / count
end
