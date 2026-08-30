# Proposals that use the gradient once.
#
# Between a random walk, which ignores the gradient entirely, and Hamiltonian
# methods, which integrate a trajectory through it, there is a gap: one gradient
# evaluation per proposal. MALA is the classical occupant. The Barker proposal
# (Livingstone and Zanella, 2022) is the interesting one, because it is robust
# to a badly chosen step size in a way MALA is not, and a badly chosen step size
# is what most of the failures measured in this repository come down to.
#
# The difference is in what each does with a gradient that is large. MALA adds
# `eps^2 / 2` times it to the mean of the proposal, so doubling the step size
# quadruples the drift and the proposal lands somewhere the density is
# negligible. Barker uses the gradient only to choose the sign of a step whose
# size was drawn independently, so the drift is bounded by the step itself
# however steep the density is, and the scheme degrades to a random walk instead
# of collapsing.

"""
    MALA(; step_size, adapt_step_size, adapt_precond, target_accept, backend)

Metropolis-adjusted Langevin: propose `y + eps^2/2 * C * grad + eps * sqrt(C) * z`
and accept with the Metropolis-Hastings ratio for that asymmetric proposal.

`target_accept` defaults to 0.574, the high-dimensional optimum for the Langevin
scheme (Roberts and Rosenthal, 1998), against 0.234 for a random walk. The
difference is the point: a proposal that uses the gradient can afford to be
accepted more often because it is aimed.

`adapt_precond` learns a diagonal preconditioner from the running variance
during warmup, which is what makes this usable on a target whose scales differ.
"""
Base.@kwdef struct MALA{B<:ADBackend} <: AbstractSampler
    step_size::Float64 = 0.5
    adapt_step_size::Bool = true
    adapt_precond::Bool = true
    target_accept::Float64 = 0.574
    backend::B = ForwardDiffAD()
end

"""
    Barker(; step_size, adapt_step_size, adapt_precond, target_accept, backend)

The Barker proposal: draw a step `z` from a symmetric distribution, then flip its
sign towards the gradient with probability `logistic(z * grad)`.

The gradient enters only through a probability, so the proposal never moves
further than `z` however steep the density is. That is what makes this robust to
a step size several times too large, where MALA's drift term grows with the
square of the step and throws the proposal into the tail.

`target_accept` defaults to 0.4. Unlike the MALA figure this is not a cited
optimum: the published optimal acceptance rate for the Barker scheme is lower
than the Langevin one, and 0.4 is a practical default for the dimensions this
package is used at rather than an asymptotic result.
"""
Base.@kwdef struct Barker{B<:ADBackend} <: AbstractSampler
    step_size::Float64 = 0.5
    adapt_step_size::Bool = true
    adapt_precond::Bool = true
    target_accept::Float64 = 0.4
    backend::B = ForwardDiffAD()
end

const GradientMH = Union{MALA,Barker}

mutable struct GradientMHState
    y::Vector{Float64}
    lp::Float64
    grad::Vector{Float64}
    logeps::Float64
    sd::Vector{Float64}          # diagonal preconditioner, as standard deviations
    iter::Int
    accepted::Int
    mean::Vector{Float64}
    M2::Vector{Float64}
end

function init_state(rng::AbstractRNG, model::AbstractModel, s::GradientMH, y0::AbstractVector;
                    n_warmup::Int = 0)
    d = dimension(model)
    lp, grad = logdensity_and_gradient(model, collect(float.(y0)); backend = s.backend)
    isfinite(lp) || throw(NonFiniteDensityError(y0, lp, :logdensity))
    if !all(isfinite, grad)
        i = findfirst(!isfinite, grad)
        throw(NonFiniteDensityError(collect(float.(y0)), grad[i], :gradient, i))
    end
    return GradientMHState(collect(float.(y0)), lp, collect(float.(grad)), log(s.step_size),
                           ones(d), 0, 0, zeros(d), zeros(d))
end

# The proposal and the Hastings correction, which is the whole difference
# between the two samplers. Returns the proposed point and
# `log q(y | prop) - log q(prop | y)` given the gradient at each end.
function _propose(rng::AbstractRNG, ::MALA, st::GradientMHState, eps::Float64)
    drift = (eps^2 / 2) .* st.sd .^ 2 .* st.grad
    noise = eps .* st.sd .* randn(rng, length(st.y))
    return st.y .+ drift .+ noise
end

function _log_hastings(::MALA, st::GradientMHState, prop::AbstractVector,
                       grad_prop::AbstractVector, eps::Float64)
    # Gaussian proposal densities, sharing eps and the preconditioner, so every
    # normalising constant cancels and only the exponents matter.
    fwd_mean = st.y .+ (eps^2 / 2) .* st.sd .^ 2 .* st.grad
    bwd_mean = prop .+ (eps^2 / 2) .* st.sd .^ 2 .* grad_prop
    scale = eps .* st.sd
    log_fwd = -sum(((prop .- fwd_mean) ./ scale) .^ 2) / 2
    log_bwd = -sum(((st.y .- bwd_mean) ./ scale) .^ 2) / 2
    return log_bwd - log_fwd
end

function _propose(rng::AbstractRNG, ::Barker, st::GradientMHState, eps::Float64)
    z = eps .* st.sd .* randn(rng, length(st.y))
    # sign chosen towards the gradient, with a probability rather than a rule
    signs = [rand(rng) < logistic(z[i] * st.grad[i]) ? 1.0 : -1.0 for i in eachindex(z)]
    return st.y .+ signs .* z
end

function _log_hastings(::Barker, st::GradientMHState, prop::AbstractVector,
                       grad_prop::AbstractVector, eps::Float64)
    # The symmetric part of the proposal cancels, leaving only the sign
    # probabilities at each end.
    delta = prop .- st.y
    return sum(log1pexp.(-st.grad .* delta) .- log1pexp.(grad_prop .* delta))
end

function step!(rng::AbstractRNG, model::AbstractModel, s::GradientMH, st::GradientMHState,
               warmup::Bool)
    d = length(st.y)
    st.iter += 1
    eps = exp(st.logeps)

    prop = _propose(rng, s, st, eps)
    lp_prop, grad_prop = logdensity_and_gradient(model, prop; backend = s.backend)
    alpha = if isfinite(lp_prop) && all(isfinite, grad_prop)
        min(1.0, exp(lp_prop - st.lp + _log_hastings(s, st, prop, grad_prop, eps)))
    else
        0.0
    end
    if rand(rng) < alpha
        st.y = prop
        st.lp = lp_prop
        st.grad = collect(float.(grad_prop))
        st.accepted += 1
    end

    if warmup
        if s.adapt_step_size
            gamma = 1 / st.iter^0.6                      # Robbins-Monro, summable squares
            st.logeps += gamma * (alpha - s.target_accept)
        end
        if s.adapt_precond
            _update_running_var!(st)
            if st.iter > 100 && st.iter % 50 == 0
                v = st.M2 ./ (st.iter - 1)
                st.sd = sqrt.(max.(v, 1e-12))
                st.sd ./= exp(Statistics.mean(log.(st.sd)))   # scale lives in eps, shape here
            end
        end
    end
    # one gradient per proposal, whether or not it was accepted, which is the
    # denominator a cost comparison against a trajectory sampler needs
    return copy(st.y), (accept_prob = alpha, log_density = st.lp, step_size = eps,
                        n_gradient = 1.0)
end

function _update_running_var!(st::GradientMHState)
    delta = st.y .- st.mean
    st.mean .+= delta ./ st.iter
    st.M2 .+= delta .* (st.y .- st.mean)
    return st
end

function refresh!(model::AbstractModel, s::GradientMH, st::GradientMHState)
    st.lp, g = logdensity_and_gradient(model, st.y; backend = s.backend)
    st.grad = collect(float.(g))
    return st
end

summary_of(st::GradientMHState) =
    (step_size = exp(st.logeps), acceptance = st.accepted / max(st.iter, 1))
