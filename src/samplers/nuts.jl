# The No-U-Turn Sampler: recursive doubling with multinomial sampling over the
# trajectory (Hoffman and Gelman 2014; Betancourt 2017).
#
# Three things are worth pointing at, because they are where hand-rolled NUTS
# implementations go wrong, and two of them went wrong here first:
#
#   1. The doubling is *biased* towards the newest subtree at the top level and
#      unbiased inside a subtree. Using the unbiased rule everywhere loses the
#      efficiency gain; using the biased rule everywhere breaks detailed balance.
#   2. A subtree that diverges or U-turns contributes no draws, but the
#      trajectory statistics it produced still feed step size adaptation.
#   3. The termination check has to be applied at *every* merge, including the
#      outermost doubling, and with the same rule each time. Applying the extra
#      cross-subtree checks only inside the recursion, as the first version here
#      did, makes the stopping rule depend on the order in which the trajectory
#      was grown; the sampler then still looks healthy - no divergences, R-hat
#      1.00, sensible acceptance rate - while quietly understating the posterior
#      standard deviation of a rho = 0.95 Gaussian by 2 to 4 percent, roughly a
#      z of -7 against the analytic value. That is the bug the correlated
#      Gaussian test in `test/test_geometries.jl` exists to catch.

"""
    UTurnCriterion

Decides when a trajectory has doubled back on itself. Implementations define
`tree_continues(criterion, metric, tree)`, and may additionally define
`merge_continues(criterion, metric, merged, left, right)` if they need to see
the two halves being joined.
"""
abstract type UTurnCriterion end

"""
    ClassicUTurn()

The original criterion: keep going while the endpoints are still moving apart,
`(y+ - y-)' M^{-1} p± > 0`.
"""
struct ClassicUTurn <: UTurnCriterion end

"""
    GeneralizedUTurn()

Betancourt's criterion, on the summed momentum `rho` of the trajectory rather
than on the endpoint separation. Better behaved than the classic form when the
metric is far from the local geometry.
"""
struct GeneralizedUTurn <: UTurnCriterion end

"""
    StrictGeneralizedUTurn()

The generalised criterion plus the two checks across the boundary of the two
halves being merged, which catch trajectories that turn within a subtree
boundary and would otherwise run to the maximum tree depth. This is the default
here and in Stan.
"""
struct StrictGeneralizedUTurn <: UTurnCriterion end

"""
    Tree

One subtree of a NUTS trajectory: its two endpoints in position order, the
summed momentum, the draw selected from within it, and its log weight
`log sum exp(H0 - H)`.
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
    alpha_sum::Float64
    n_alpha::Int
    n_leapfrog::Int
end

"""
    uturn_ok(metric, p_minus, p_plus, rho)

The generalised no-U-turn condition on a momentum sum: `rho' M^{-1} p± > 0`.
"""
uturn_ok(metric::AbstractMetric, p_minus, p_plus, rho) =
    dot(velocity(metric, p_minus), rho) > 0 && dot(velocity(metric, p_plus), rho) > 0

"""
    tree_continues(criterion, metric, tree)

Whether a single trajectory has not yet turned.
"""
function tree_continues(::ClassicUTurn, metric::AbstractMetric, t::Tree)
    delta = t.y_plus .- t.y_minus
    return dot(velocity(metric, t.p_minus), delta) > 0 &&
           dot(velocity(metric, t.p_plus), delta) > 0
end

tree_continues(::Union{GeneralizedUTurn,StrictGeneralizedUTurn}, metric::AbstractMetric, t::Tree) =
    uturn_ok(metric, t.p_minus, t.p_plus, t.rho)

"""
    merge_continues(criterion, metric, merged, left, right)

Whether the trajectory formed by joining `left` and `right` may keep growing.
The default consults only the merged trajectory; the strict criterion also
checks the two ranges that span the boundary between the halves.
"""
merge_continues(c::UTurnCriterion, metric::AbstractMetric, merged::Tree, left::Tree, right::Tree) =
    tree_continues(c, metric, merged)

function merge_continues(c::StrictGeneralizedUTurn, metric::AbstractMetric, merged::Tree,
                         left::Tree, right::Tree)
    return tree_continues(c, metric, merged) &&
           uturn_ok(metric, merged.p_minus, right.p_minus, left.rho .+ right.p_minus) &&
           uturn_ok(metric, left.p_plus, merged.p_plus, right.rho .+ left.p_plus)
end

"""
    NUTS(; max_treedepth = 10, target_accept = 0.8, metric = :diag,
           adapt_step_size = true, adapt_metric = true,
           uturn = StrictGeneralizedUTurn(), divergence_threshold = 1000.0,
           backend = ForwardDiffAD())

No-U-Turn sampler with dual-averaged step size and a windowed metric.
"""
Base.@kwdef struct NUTS{U<:UTurnCriterion,B<:ADBackend} <: AbstractSampler
    max_treedepth::Int = 10
    step_size::Float64 = NaN
    metric::Symbol = :diag
    adapt_step_size::Bool = true
    adapt_metric::Bool = true
    target_accept::Float64 = 0.8
    uturn::U = StrictGeneralizedUTurn()
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
    leaf(rng, model, s, st, y, p, grad, v, H0)

One leapfrog step, packaged as a depth-zero tree. Returns `(tree, diverged)`.
"""
function leaf(model::Model, s::NUTS, st::HamiltonianState, y, p, grad, v::Int, H0::Float64)
    ynew, pnew, gradnew, lp = leapfrog(model, st.metric, y, p, grad, v * st.step_size, s.backend)
    finite = isfinite(lp) && all(isfinite, gradnew) && all(isfinite, pnew)
    H = finite ? hamiltonian(lp, st.metric, pnew) : Inf
    dH = H0 - H
    diverged = !isfinite(H) || -dH > s.divergence_threshold
    logw = isfinite(dH) ? dH : -Inf
    alpha = dH > 0 ? 1.0 : (isfinite(dH) ? exp(dH) : 0.0)
    tree = Tree(ynew, pnew, gradnew, ynew, pnew, gradnew, copy(pnew),
                ynew, pnew, gradnew, lp, logw, alpha, 1, 1)
    return tree, diverged
end

"""
    join_trees(rng, left, right, prefer)

Merge two trees that are adjacent in position order (`left` holds the backward
end) and choose the surviving draw. `prefer` selects the sampling rule:
`:multinomial` inside a subtree, `:biased` at the top level, where preferring
the newest subtree is what lets NUTS travel further than an unbiased draw would.
"""
function join_trees(rng::AbstractRNG, left::Tree, right::Tree, prefer::Symbol, newer::Symbol)
    logw = logsumexp((left.logw, right.logw))
    take_right = if prefer === :biased
        # Bias towards the newly built half, whichever side it is on.
        target, other = newer === :right ? (right, left) : (left, right)
        chosen_new = isfinite(target.logw) &&
                     (target.logw >= other.logw || log(rand(rng)) < target.logw - other.logw)
        newer === :right ? chosen_new : !chosen_new
    else
        isfinite(right.logw) && (right.logw >= logw || log(rand(rng)) < right.logw - logw)
    end
    chosen = take_right ? right : left
    return Tree(left.y_minus, left.p_minus, left.grad_minus,
                right.y_plus, right.p_plus, right.grad_plus,
                left.rho .+ right.rho,
                chosen.y_sample, chosen.p_sample, chosen.grad_sample, chosen.lp_sample,
                logw, left.alpha_sum + right.alpha_sum, left.n_alpha + right.n_alpha,
                left.n_leapfrog + right.n_leapfrog)
end

"""
    build_tree(rng, model, s, st, y, p, grad, v, depth, H0)

Recursive doubling. Returns `(tree, valid, diverged)`, where `valid` is false if
the subtree diverged or turned; an invalid subtree contributes no draw but its
acceptance statistics are still counted.
"""
function build_tree(rng::AbstractRNG, model::Model, s::NUTS, st::HamiltonianState,
                    y, p, grad, v::Int, depth::Int, H0::Float64)
    if depth == 0
        tree, diverged = leaf(model, s, st, y, p, grad, v, H0)
        return tree, !diverged, diverged
    end

    first_half, valid, diverged = build_tree(rng, model, s, st, y, p, grad, v, depth - 1, H0)
    valid || return first_half, false, diverged

    edge = v == 1 ? (first_half.y_plus, first_half.p_plus, first_half.grad_plus) :
                    (first_half.y_minus, first_half.p_minus, first_half.grad_minus)
    second_half, valid2, diverged2 = build_tree(rng, model, s, st, edge..., v, depth - 1, H0)

    left, right = v == 1 ? (first_half, second_half) : (second_half, first_half)
    merged = join_trees(rng, left, right, :multinomial, v == 1 ? :right : :left)
    ok = valid2 && merge_continues(s.uturn, st.metric, merged, left, right)
    return merged, ok, diverged || diverged2
end

function step!(rng::AbstractRNG, model::Model, s::NUTS, st::HamiltonianState, warmup::Bool)
    p0 = rand_momentum(rng, st.metric)
    H0 = hamiltonian(st.lp, st.metric, p0)

    # The initial point is a depth-zero tree with weight exp(H0 - H0) = 1.
    tree = Tree(copy(st.y), copy(p0), copy(st.grad), copy(st.y), copy(p0), copy(st.grad),
                copy(p0), copy(st.y), copy(p0), copy(st.grad), st.lp, 0.0, 0.0, 0, 0)

    depth = 0
    diverged = false
    keep_going = true
    while keep_going && depth < s.max_treedepth
        v = rand(rng, Bool) ? 1 : -1
        edge = v == 1 ? (tree.y_plus, tree.p_plus, tree.grad_plus) :
                        (tree.y_minus, tree.p_minus, tree.grad_minus)
        new_tree, valid, div = build_tree(rng, model, s, st, edge..., v, depth, H0)
        diverged |= div

        left, right = v == 1 ? (tree, new_tree) : (new_tree, tree)
        if valid
            # Sample only from a valid subtree, but merge either way so that the
            # termination check sees the whole trajectory.
            tree = join_trees(rng, left, right, :biased, v == 1 ? :right : :left)
            keep_going = merge_continues(s.uturn, st.metric, tree, left, right)
            depth += 1
        else
            merged = join_trees(rng, left, right, :multinomial, v == 1 ? :right : :left)
            tree = Tree(merged.y_minus, merged.p_minus, merged.grad_minus,
                        merged.y_plus, merged.p_plus, merged.grad_plus, merged.rho,
                        tree.y_sample, tree.p_sample, tree.grad_sample, tree.lp_sample,
                        tree.logw, merged.alpha_sum, merged.n_alpha, merged.n_leapfrog)
            keep_going = false
        end
    end

    st.y, st.lp, st.grad = tree.y_sample, tree.lp_sample, tree.grad_sample
    diverged && (st.n_divergent += 1)
    alpha = tree.n_alpha > 0 ? tree.alpha_sum / tree.n_alpha : 0.0
    adapt!(rng, model, st, alpha, warmup, s.adapt_step_size, s.adapt_metric, s.backend)

    return copy(st.y), (accept_prob = alpha, divergent = diverged, treedepth = depth,
                        n_leapfrog = tree.n_leapfrog, step_size = st.step_size,
                        energy = hamiltonian(tree.lp_sample, st.metric, tree.p_sample),
                        log_density = st.lp)
end

refresh!(model::Model, s::NUTS, st::HamiltonianState) =
    refresh_hamiltonian!(model, st, s.backend)

finish_warmup!(rng, model, s::NUTS, st::HamiltonianState) =
    finish_warmup_hamiltonian!(st, s.adapt_step_size)
