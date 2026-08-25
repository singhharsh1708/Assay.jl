# Random walk Metropolis-Hastings.
#
# The baseline. It needs no gradients, so it is the sampler that the conjugate
# tests are first proved against; everything gradient-based is then held to the
# same tests. It is also the honest control for the hard geometries, where its
# failure is the point being demonstrated.
#
# Two things here are deliberately abstract rather than inlined:
#
#   * the acceptance rule, so that Barker's rule is a two-line addition and so the
#     test suite can plug in an always-accept rule and watch the conjugate tests
#     fail (`test/negative_controls.jl`);
#   * the proposal covariance, which may be fixed, scale-adapted, or fully
#     adapted in the manner of Haario, Saksman and Tamminen (2001).

"""
    AcceptanceRule

Maps a log density ratio to an acceptance probability. Any rule that satisfies
detailed balance with a symmetric proposal may be used.
"""
abstract type AcceptanceRule end

"""
    accept_prob(rule, logratio)

Acceptance probability for a proposal whose log density ratio is `logratio`.
"""
function accept_prob end

"""
    MetropolisRule()

`α = min(1, exp(Δ))`. The standard choice, and the one that maximises the
acceptance probability among rules satisfying detailed balance (Peskun).
"""
struct MetropolisRule <: AcceptanceRule end
accept_prob(::MetropolisRule, logratio::Real) = logratio >= 0 ? 1.0 : exp(logratio)

"""
    BarkerRule()

`α = exp(Δ) / (1 + exp(Δ))`. Strictly less efficient than Metropolis, included
because having a second real rule is what keeps the abstraction honest.
"""
struct BarkerRule <: AcceptanceRule end
accept_prob(::BarkerRule, logratio::Real) = logistic(logratio)

"""
    RandomWalkMH(; scale = 1.0, cov = nothing, adapt_scale = true,
                   adapt_cov = false, target_accept = 0.234, rule = MetropolisRule())

Gaussian random walk proposal `y' = y + s * L * z`, `z ~ N(0, I)`, where `L L'`
is `cov` (identity by default).

During warmup, `log s` is moved by Robbins-Monro towards `target_accept`
(0.234 is the optimal acceptance rate for a random walk in high dimension,
Roberts, Gelman and Gilks 1997). With `adapt_cov = true` the proposal covariance
tracks the running sample covariance, scaled by `2.38^2 / d`.
"""
Base.@kwdef struct RandomWalkMH{R<:AcceptanceRule,C} <: AbstractSampler
    scale::Float64 = 1.0
    cov::C = nothing
    adapt_scale::Bool = true
    adapt_cov::Bool = false
    target_accept::Float64 = 0.234
    rule::R = MetropolisRule()
end

mutable struct MHState
    y::Vector{Float64}
    lp::Float64
    L::Matrix{Float64}
    logscale::Float64
    iter::Int
    accepted::Int
    mean::Vector{Float64}
    M2::Matrix{Float64}
end

function init_state(rng::AbstractRNG, model::AbstractModel, s::RandomWalkMH, y0::AbstractVector;
                    n_warmup::Int = 0)
    d = dimension(model)
    L = s.cov === nothing ? Matrix{Float64}(I, d, d) : Matrix(cholesky(Symmetric(s.cov)).L)
    lp = logdensity(model, y0)
    isfinite(lp) || error("initial point has non-finite log density")
    return MHState(collect(float.(y0)), lp, L, log(s.scale), 0, 0, zeros(d), zeros(d, d))
end

function step!(rng::AbstractRNG, model::AbstractModel, s::RandomWalkMH, st::MHState, warmup::Bool)
    d = length(st.y)
    st.iter += 1
    z = randn(rng, d)
    prop = st.y .+ exp(st.logscale) .* (st.L * z)
    lp_prop = logdensity(model, prop)
    logratio = lp_prop - st.lp
    alpha = isfinite(lp_prop) ? accept_prob(s.rule, logratio) : 0.0
    if rand(rng) < alpha
        st.y = prop
        st.lp = lp_prop
        st.accepted += 1
    end
    if warmup
        if s.adapt_scale
            gamma = 1 / st.iter^0.6                      # Robbins-Monro, summable squares
            st.logscale += gamma * (alpha - s.target_accept)
        end
        if s.adapt_cov
            _update_running_cov!(st)
            if st.iter > 100 && st.iter % 50 == 0
                C = st.M2 ./ (st.iter - 1) + 1e-10 * I    # ridge keeps the factorisation well posed
                st.L = Matrix(cholesky(Symmetric(C)).L) .* (2.38 / sqrt(d))
                st.logscale = 0.0
            end
        end
    end
    return copy(st.y), (accept_prob = alpha, log_density = st.lp, step_size = exp(st.logscale))
end

function _update_running_cov!(st::MHState)
    delta = st.y .- st.mean
    st.mean .+= delta ./ st.iter
    st.M2 .+= delta * (st.y .- st.mean)'
    return st
end

function refresh!(model::AbstractModel, ::RandomWalkMH, st::MHState)
    st.lp = logdensity(model, st.y)
    return st
end

summary_of(st::MHState) = (step_size = exp(st.logscale), accept_rate = st.accepted / max(st.iter, 1))
