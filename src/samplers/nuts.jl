# The No-U-Turn Sampler: recursive doubling with multinomial sampling over the
# trajectory (Hoffman and Gelman 2014; Betancourt 2017).
#
# Three things are worth pointing at in this file, because they are where
# hand-rolled NUTS implementations usually go wrong:
#
#   1. The doubling is *biased* towards the newest subtree at the top level and
#      unbiased inside a subtree. Using the unbiased rule everywhere loses the
#      efficiency gain; using the biased rule everywhere breaks detailed balance.
#   2. A subtree that diverges or U-turns contributes no samples, but the
#      trajectory statistics it produced still feed step size adaptation.
#   3. The U-turn check has to be applied to the merged tree as well as to each
#      subtree, otherwise the sampler runs past the turn and the chain becomes
#      slightly, and very hard to detect, wrong. `test/negative_controls.jl`
#      installs a criterion that never fires and shows what that costs.

"""
    UTurnCriterion

Decides when a trajectory has doubled back on itself. Implementations return
`true` while the trajectory should keep growing.
"""
abstract type UTurnCriterion end

"""
    ClassicUTurn()

The original criterion: stop when the trajectory endpoints stop moving apart,
`(y+ - y-)' M^{-1} p± <= 0`.
"""
struct ClassicUTurn <: UTurnCriterion end

"""
    GeneralizedUTurn()

Betancourt's criterion, on the summed momentum `rho` of the subtree rather than
the endpoint separation, plus the two extra checks Stan applies across the
subtree boundary. More robust in regions of high curvature such as the neck of
the funnel.
"""
struct GeneralizedUTurn <: UTurnCriterion end

"""
    no_uturn(criterion, metric, y_minus, y_plus, p_minus, p_plus, rho)

`true` while the trajectory has not turned.
"""
function no_uturn(::ClassicUTurn, metric::AbstractMetric, y_minus, y_plus, p_minus, p_plus, rho)
    delta = y_plus .- y_minus
    return dot(velocity(metric, p_minus), delta) > 0 && dot(velocity(metric, p_plus), delta) > 0
end

function no_uturn(::GeneralizedUTurn, metric::AbstractMetric, y_minus, y_plus, p_minus, p_plus, rho)
    return dot(velocity(metric, p_minus), rho) > 0 && dot(velocity(metric, p_plus), rho) > 0
end

"""
    NUTS(; max_treedepth = 10, target_accept = 0.8, metric = :diag,
           adapt_step_size = true, adapt_metric = true, uturn = GeneralizedUTurn(),
           divergence_threshold = 1000.0, backend = ForwardDiffAD())

No-U-Turn sampler with dual-averaged step size and a windowed metric.
"""
Base.@kwdef struct NUTS{U<:UTurnCriterion,B<:ADBackend} <: AbstractSampler
    max_treedepth::Int = 10
    step_size::Float64 = NaN
    metric::Symbol = :diag
    adapt_step_size::Bool = true
    adapt_metric::Bool = true
    target_accept::Float64 = 0.8
    uturn::U = GeneralizedUTurn()
    divergence_threshold::Float64 = 1000.0
    init_buffer::Int = 75
    term_buffer::Int = 100
    base_window::Int = 25
    backend::B = ForwardDiffAD()
end

function init_state(rng::AbstractRNG, model::Model, s::NUTS, y0::AbstractVector; n_warmup::Int = 0)
    return init_hamiltonian_state(rng, model, y0; metric_kind = s.metric, step_size = s.step_size,
                                  target_accept = s.target_accept, n_warmup = n_warmup,
                                  backend = s.backend, init_buffer = s.init_buffer,
                                  term_buffer = s.term_buffer, base_window = s.base_window)
end

"""
    Tree

One subtree of a NUTS trajectory: its two endpoints, the summed momentum, the
draw selected from within it, and its log weight `log sum exp(H0 - H)`.
"""
struct Tree
    y_minus::Vector{Float64}
    p_minus::Vector{Float64}
    grad_minus::Vector{Float64}
    y_plus::Vector{Float64}
    p_plus::Vector{Float64}
    grad_plus::Vector{Float64}
    rho::Vector{Float64}
    y_sample::Vector{Float64}
    p_sample::Vector{Float64}
    grad_sample::Vector{Float64}
    lp_sample::Float64
    logw::Float64
    continued::Bool
    diverged::Bool
    alpha_sum::Float64
    n_alpha::Int
    n_leapfrog::Int
end

function build_tree(rng::AbstractRNG, model::Model, s::NUTS, st::HamiltonianState,
                    y, p, grad, v::Int, depth::Int, H0::Float64)
    if depth == 0
        ynew, pnew, gradnew, lp = leapfrog(model, st.metric, y, p, grad, v * st.step_size, s.backend)
        H = (isfinite(lp) && all(isfinite, gradnew)) ? hamiltonian(lp, st.metric, pnew) : Inf
        dH = H0 - H
        diverged = !isfinite(H) || -dH > s.divergence_threshold
        logw = isfinite(dH) ? dH : -Inf
        alpha = dH > 0 ? 1.0 : (isfinite(dH) ? exp(dH) : 0.0)
        return Tree(ynew, pnew, gradnew, ynew, pnew, gradnew, copy(pnew),
                    ynew, pnew, gradnew, lp, logw, !diverged, diverged, alpha, 1, 1)
    end

    left = build_tree(rng, model, s, st, y, p, grad, v, depth - 1, H0)
    (left.diverged || !left.continued) && return left

    right = if v == 1
        build_tree(rng, model, s, st, left.y_plus, left.p_plus, left.grad_plus, v, depth - 1, H0)
    else
        build_tree(rng, model, s, st, left.y_minus, left.p_minus, left.grad_minus, v, depth - 1, H0)
    end

    # A: subtree holding the backwards end, B: subtree holding the forwards end.
    A, B = v == 1 ? (left, right) : (right, left)
    rho = A.rho .+ B.rho
    logw = logsumexp((left.logw, right.logw))

    # Unbiased multinomial choice between the two halves.
    take_right = !right.diverged && isfinite(right.logw) &&
                 (right.logw >= logw || log(rand(rng)) < right.logw - logw)
    chosen = take_right ? right : left

    continued = left.continued && right.continued && !left.diverged && !right.diverged &&
                no_uturn(s.uturn, st.metric, A.y_minus, B.y_plus, A.p_minus, B.p_plus, rho)
    if continued && s.uturn isa GeneralizedUTurn
        # Stan's extra checks across the subtree boundary: they catch turns that
        # the endpoint check misses when curvature is high.
        rho1 = A.rho .+ B.p_minus
        rho2 = A.p_plus .+ B.rho
        continued = no_uturn(s.uturn, st.metric, A.y_minus, B.y_minus, A.p_minus, B.p_minus, rho1) &&
                    no_uturn(s.uturn, st.metric, A.y_plus, B.y_plus, A.p_plus, B.p_plus, rho2)
    end

    return Tree(A.y_minus, A.p_minus, A.grad_minus, B.y_plus, B.p_plus, B.grad_plus, rho,
                chosen.y_sample, chosen.p_sample, chosen.grad_sample, chosen.lp_sample,
                logw, continued, left.diverged || right.diverged,
                left.alpha_sum + right.alpha_sum, left.n_alpha + right.n_alpha,
                left.n_leapfrog + right.n_leapfrog)
end

function step!(rng::AbstractRNG, model::Model, s::NUTS, st::HamiltonianState, warmup::Bool)
    p0 = rand_momentum(rng, st.metric)
    H0 = hamiltonian(st.lp, st.metric, p0)

    y_minus = copy(st.y); p_minus = copy(p0); grad_minus = copy(st.grad)
    y_plus = copy(st.y);  p_plus = copy(p0);  grad_plus = copy(st.grad)
    rho = copy(p0)
    y_sample, lp_sample, grad_sample, p_sample = copy(st.y), st.lp, copy(st.grad), copy(p0)

    logw = 0.0                       # log weight of the initial point: exp(H0 - H0)
    depth = 0
    diverged = false
    alpha_sum = 0.0
    n_alpha = 0
    n_leapfrog = 0
    keep_going = true

    while keep_going && depth < s.max_treedepth
        v = rand(rng, Bool) ? 1 : -1
        tree = if v == 1
            build_tree(rng, model, s, st, y_plus, p_plus, grad_plus, v, depth, H0)
        else
            build_tree(rng, model, s, st, y_minus, p_minus, grad_minus, v, depth, H0)
        end
        if v == 1
            y_plus, p_plus, grad_plus = tree.y_plus, tree.p_plus, tree.grad_plus
        else
            y_minus, p_minus, grad_minus = tree.y_minus, tree.p_minus, tree.grad_minus
        end
        alpha_sum += tree.alpha_sum
        n_alpha += tree.n_alpha
        n_leapfrog += tree.n_leapfrog

        if tree.diverged
            diverged = true
            keep_going = false
        elseif tree.continued
            # Biased progressive sampling: prefer the new subtree, which is what
            # makes NUTS move further than a plain multinomial draw would.
            if isfinite(tree.logw) && (tree.logw >= logw || log(rand(rng)) < tree.logw - logw)
                y_sample, lp_sample, grad_sample, p_sample =
                    tree.y_sample, tree.lp_sample, tree.grad_sample, tree.p_sample
            end
        else
            keep_going = false
        end

        if !tree.diverged
            logw = logsumexp((logw, tree.logw))
            rho .+= tree.rho
            keep_going = keep_going &&
                         no_uturn(s.uturn, st.metric, y_minus, y_plus, p_minus, p_plus, rho)
        end
        depth += 1
    end

    st.y, st.lp, st.grad = y_sample, lp_sample, grad_sample
    diverged && (st.n_divergent += 1)
    alpha = n_alpha > 0 ? alpha_sum / n_alpha : 0.0
    adapt!(rng, model, st, alpha, warmup, s.adapt_step_size, s.adapt_metric, s.backend)

    return copy(st.y), (accept_prob = alpha, divergent = diverged, treedepth = depth,
                        n_leapfrog = n_leapfrog, step_size = st.step_size,
                        energy = hamiltonian(lp_sample, st.metric, p_sample),
                        log_density = st.lp)
end

finish_warmup!(rng, model, s::NUTS, st::HamiltonianState) =
    finish_warmup_hamiltonian!(st, s.adapt_step_size)
