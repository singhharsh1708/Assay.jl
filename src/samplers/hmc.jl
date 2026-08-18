# Hamiltonian Monte Carlo with a fixed-length trajectory.
#
# The trajectory length is the one thing HMC cannot adapt on its own, which is
# exactly the gap NUTS fills; keeping static HMC in the package makes that
# comparison concrete rather than asserted.

"""
    HMC(; n_leapfrog = 10, step_size = NaN, metric = :diag, adapt_step_size = true,
          adapt_metric = true, target_accept = 0.8, jitter = 0.0,
          divergence_threshold = 1000.0, backend = ForwardDiffAD())

Static Hamiltonian Monte Carlo. `step_size = NaN` means "find a reasonable one
and then dual-average it to `target_accept`". `metric` is `:unit`, `:diag` or
`:dense`; `jitter` randomises the trajectory length by that fraction, which
removes the resonance a fixed length can hit on near-periodic targets.
"""
Base.@kwdef struct HMC{B<:ADBackend} <: AbstractSampler
    n_leapfrog::Int = 10
    step_size::Float64 = NaN
    metric::Symbol = :diag
    adapt_step_size::Bool = true
    adapt_metric::Bool = true
    target_accept::Float64 = 0.8
    jitter::Float64 = 0.0
    divergence_threshold::Float64 = 1000.0
    init_buffer::Int = 75
    term_buffer::Int = 100
    base_window::Int = 25
    backend::B = ForwardDiffAD()
end

function init_state(rng::AbstractRNG, model::Model, s::HMC, y0::AbstractVector; n_warmup::Int = 0)
    return init_hamiltonian_state(rng, model, y0; metric_kind = s.metric, step_size = s.step_size,
                                  target_accept = s.target_accept, n_warmup = n_warmup,
                                  backend = s.backend, init_buffer = s.init_buffer,
                                  term_buffer = s.term_buffer, base_window = s.base_window)
end

function step!(rng::AbstractRNG, model::Model, s::HMC, st::HamiltonianState, warmup::Bool)
    p0 = rand_momentum(rng, st.metric)
    H0 = hamiltonian(st.lp, st.metric, p0)
    L = s.jitter > 0 ?
        max(1, round(Int, s.n_leapfrog * (1 + s.jitter * (2 * rand(rng) - 1)))) : s.n_leapfrog

    y, p, grad, lp = st.y, p0, st.grad, st.lp
    diverged = false
    for _ in 1:L
        y, p, grad, lp = leapfrog(model, st.metric, y, p, grad, st.step_size, s.backend)
        if !isfinite(lp) || any(!isfinite, grad)
            diverged = true
            break
        end
        if hamiltonian(lp, st.metric, p) - H0 > s.divergence_threshold
            diverged = true
            break
        end
    end

    H1 = diverged ? Inf : hamiltonian(lp, st.metric, p)
    dH = H0 - H1
    alpha = (isfinite(dH) && dH < 0) ? exp(dH) : (isfinite(dH) ? 1.0 : 0.0)
    energy = H0
    if !diverged && rand(rng) < alpha
        st.y, st.lp, st.grad = y, lp, grad
        energy = H1
    end
    diverged && (st.n_divergent += 1)

    adapt!(rng, model, st, alpha, warmup, s.adapt_step_size, s.adapt_metric, s.backend)
    return copy(st.y), (accept_prob = alpha, divergent = diverged, step_size = st.step_size,
                        n_leapfrog = L, energy = energy, log_density = st.lp)
end

finish_warmup!(rng, model, s::HMC, st::HamiltonianState) =
    finish_warmup_hamiltonian!(st, s.adapt_step_size)
