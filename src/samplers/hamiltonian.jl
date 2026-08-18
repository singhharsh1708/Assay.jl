# Shared machinery for gradient-based samplers: metrics, the leapfrog
# integrator, dual averaging for the step size, and Stan's windowed schedule for
# the mass matrix.
#
# HMC and NUTS differ only in how they build a trajectory. Everything else -
# momentum sampling, energy accounting, divergence detection, adaptation - is
# here and is shared, so the two sampler files contain the trajectory logic and
# nothing else.

"""
    AbstractMetric

A Euclidean metric, i.e. a choice of mass matrix `M`. Momenta are drawn from
`N(0, M)` and the kinetic energy is `p' M^{-1} p / 2`, so the metric is what
makes a badly scaled posterior tractable: `M^{-1}` should look like the
posterior covariance.
"""
abstract type AbstractMetric end

"""
    UnitMetric(d)

`M = I`. No adaptation; the honest baseline that shows why adaptation matters.
"""
struct UnitMetric <: AbstractMetric
    d::Int
end

"""
    DiagMetric(inv_mass)

`M^{-1} = Diagonal(inv_mass)`. The default: it costs `O(d)` and captures the
scale differences that dominate in practice.
"""
struct DiagMetric <: AbstractMetric
    inv_mass::Vector{Float64}
end
DiagMetric(d::Int) = DiagMetric(ones(d))

"""
    DenseMetric(inv_mass)

`M^{-1}` is a full covariance. Captures correlation as well as scale, at
`O(d^2)` per leapfrog step and with `O(d^2)` parameters to estimate.
"""
struct DenseMetric{C} <: AbstractMetric
    inv_mass::Matrix{Float64}
    chol::C
end
DenseMetric(Sigma::AbstractMatrix) = DenseMetric(Matrix(Sigma), cholesky(Symmetric(Sigma)))
DenseMetric(d::Int) = DenseMetric(Matrix{Float64}(I, d, d))

dimension(m::UnitMetric) = m.d
dimension(m::DiagMetric) = length(m.inv_mass)
dimension(m::DenseMetric) = size(m.inv_mass, 1)

"""
    rand_momentum(rng, metric)

Draw `p ~ N(0, M)`.
"""
rand_momentum(rng::AbstractRNG, m::UnitMetric) = randn(rng, m.d)
rand_momentum(rng::AbstractRNG, m::DiagMetric) = randn(rng, length(m.inv_mass)) ./ sqrt.(m.inv_mass)
function rand_momentum(rng::AbstractRNG, m::DenseMetric)
    # Sigma = M^{-1} = L L', so p = L^{-T} z has covariance M.
    return m.chol.U \ randn(rng, size(m.inv_mass, 1))
end

"""
    kinetic(metric, p)

`p' M^{-1} p / 2`.
"""
kinetic(::UnitMetric, p::AbstractVector) = 0.5 * dot(p, p)
kinetic(m::DiagMetric, p::AbstractVector) = 0.5 * sum(m.inv_mass .* p .^ 2)
kinetic(m::DenseMetric, p::AbstractVector) = 0.5 * dot(p, m.inv_mass * p)

"""
    velocity(metric, p)

`M^{-1} p`, the position update direction in the leapfrog step and the vector
the U-turn criterion is evaluated against.
"""
velocity(::UnitMetric, p::AbstractVector) = p
velocity(m::DiagMetric, p::AbstractVector) = m.inv_mass .* p
velocity(m::DenseMetric, p::AbstractVector) = m.inv_mass * p

"""
    leapfrog(model, metric, y, p, grad, eps, backend)

One leapfrog step of size `eps`, returning `(y, p, grad, logdensity)`. The
integrator is volume preserving and reversible, which is what makes the
Metropolis correction on `exp(-H)` exact; any change here has to preserve both
properties or the sampler stops being valid.
"""
function leapfrog(model::Model, metric::AbstractMetric, y::AbstractVector, p::AbstractVector,
                  grad::AbstractVector, eps::Real, backend::ADBackend)
    phalf = p .+ (eps / 2) .* grad
    ynew = y .+ eps .* velocity(metric, phalf)
    lp, gradnew = logdensity_and_gradient(model, ynew; backend = backend)
    pnew = phalf .+ (eps / 2) .* gradnew
    return ynew, pnew, gradnew, lp
end

"""
    hamiltonian(lp, metric, p)

`H = -log p(y) + p' M^{-1} p / 2`.
"""
hamiltonian(lp::Real, metric::AbstractMetric, p::AbstractVector) = -lp + kinetic(metric, p)

# --------------------------------------------------------------------------
# Step size adaptation
# --------------------------------------------------------------------------

"""
    DualAveraging(; target = 0.8, gamma = 0.05, t0 = 10, kappa = 0.75)

Nesterov dual averaging as used by Hoffman and Gelman (2014) to drive the mean
acceptance statistic to `target`. The averaged iterate `epsbar` is what the
sampling phase uses, because the last raw iterate is noisy.
"""
mutable struct DualAveraging
    mu::Float64
    logeps::Float64
    logepsbar::Float64
    Hbar::Float64
    t::Int
    target::Float64
    gamma::Float64
    t0::Float64
    kappa::Float64
end

function DualAveraging(eps0::Real; target::Real = 0.8, gamma::Real = 0.05,
                       t0::Real = 10, kappa::Real = 0.75)
    return DualAveraging(log(10 * eps0), log(eps0), 0.0, 0.0, 0, target, gamma, t0, kappa)
end

"""
    da_update!(da, alpha)

Feed one acceptance statistic and return the step size for the next iteration.
"""
function da_update!(da::DualAveraging, alpha::Real)
    da.t += 1
    a = clamp(alpha, 0.0, 1.0)
    eta = 1 / (da.t + da.t0)
    da.Hbar = (1 - eta) * da.Hbar + eta * (da.target - a)
    da.logeps = da.mu - sqrt(da.t) / da.gamma * da.Hbar
    w = da.t^(-da.kappa)
    da.logepsbar = w * da.logeps + (1 - w) * da.logepsbar
    return exp(da.logeps)
end

"""
    da_restart!(da, eps0)

Restart dual averaging around a new step size, done whenever the metric changes
under it.
"""
function da_restart!(da::DualAveraging, eps0::Real)
    da.mu = log(10 * eps0)
    da.logeps = log(eps0)
    da.logepsbar = 0.0
    da.Hbar = 0.0
    da.t = 0
    return da
end

da_final(da::DualAveraging) = exp(da.logepsbar)

"""
    find_reasonable_step_size(rng, model, metric, y, backend; target_logratio = log(0.8))

Heuristic of Hoffman and Gelman (2014): double or halve the step size until one
leapfrog step crosses an acceptance probability of about 0.5. Only a starting
point for dual averaging, but a bad one costs hundreds of wasted iterations.
"""
function find_reasonable_step_size(rng::AbstractRNG, model::Model, metric::AbstractMetric,
                                   y::AbstractVector, backend::ADBackend)
    eps = 1.0
    lp, grad = logdensity_and_gradient(model, y; backend = backend)
    p = rand_momentum(rng, metric)
    H0 = hamiltonian(lp, metric, p)
    _, pn, _, lpn = leapfrog(model, metric, y, p, grad, eps, backend)
    dH = H0 - hamiltonian(lpn, metric, pn)
    direction = (isfinite(dH) && dH > log(0.8)) ? 1 : -1
    for _ in 1:100
        eps = direction == 1 ? 2eps : eps / 2
        _, pn, _, lpn = leapfrog(model, metric, y, p, grad, eps, backend)
        dH = H0 - hamiltonian(lpn, metric, pn)
        if direction == 1
            (isfinite(dH) && dH > log(0.8)) || break
        else
            (!isfinite(dH) || dH < log(0.8)) || break
        end
        (eps < 1e-10 || eps > 1e7) && break
    end
    return eps
end

# --------------------------------------------------------------------------
# Mass matrix adaptation
# --------------------------------------------------------------------------

"""
    WelfordAccumulator(d; dense = false)

Streaming mean and (co)variance of the warmup draws, used to build the metric.
"""
mutable struct WelfordAccumulator
    n::Int
    mean::Vector{Float64}
    M2::Vector{Float64}
    M2dense::Matrix{Float64}
    dense::Bool
end
WelfordAccumulator(d::Int; dense::Bool = false) =
    WelfordAccumulator(0, zeros(d), zeros(d), dense ? zeros(d, d) : zeros(0, 0), dense)

function welford_add!(w::WelfordAccumulator, y::AbstractVector)
    w.n += 1
    delta = y .- w.mean
    w.mean .+= delta ./ w.n
    if w.dense
        w.M2dense .+= delta * (y .- w.mean)'
    else
        w.M2 .+= delta .* (y .- w.mean)
    end
    return w
end

function welford_reset!(w::WelfordAccumulator)
    w.n = 0
    fill!(w.mean, 0)
    fill!(w.M2, 0)
    w.dense && fill!(w.M2dense, 0)
    return w
end

"""
    welford_metric(w)

Regularised variance estimate, shrunk towards the identity exactly as Stan does:

    (n / (n + 5)) * var + 1e-3 * (5 / (n + 5))

Without the shrinkage a short adaptation window can produce a near-singular
metric, which then makes every subsequent trajectory diverge.
"""
function welford_metric(w::WelfordAccumulator)
    n = w.n
    n < 3 && return w.dense ? Matrix{Float64}(I, length(w.mean), length(w.mean)) : ones(length(w.mean))
    shrink = n / (n + 5)
    floorterm = 1e-3 * (5 / (n + 5))
    if w.dense
        return (w.M2dense ./ (n - 1)) .* shrink + floorterm * I    # `+`, not `.+`: I is a UniformScaling
    else
        return shrink .* (w.M2 ./ (n - 1)) .+ floorterm
    end
end

"""
    WindowSchedule(n_warmup; init_buffer = 75, term_buffer = 50, base_window = 25)

Stan's three-phase warmup: an initial fast interval where only the step size
moves, a sequence of doubling windows where the metric is re-estimated, and a
final interval where the step size is polished under the last metric.
"""
struct WindowSchedule
    n_warmup::Int
    init_buffer::Int
    term_buffer::Int
    window_ends::Vector{Int}
end

function WindowSchedule(n_warmup::Int; init_buffer::Int = 75, term_buffer::Int = 50,
                        base_window::Int = 25)
    if n_warmup < init_buffer + term_buffer + base_window
        # Too short for the full schedule: fall back to a single window over the
        # middle 80% of warmup rather than silently skipping metric adaptation.
        ib = max(div(n_warmup, 10), 1)
        tb = max(div(n_warmup, 10), 1)
        return WindowSchedule(n_warmup, ib, tb, [n_warmup - tb])
    end
    ends = Int[]
    next_window = init_buffer + base_window
    window = base_window
    while next_window <= n_warmup - term_buffer
        push!(ends, next_window)
        window *= 2
        # Absorb a window that would overrun the terminal buffer into this one.
        next_window + 2window > n_warmup - term_buffer && (window = n_warmup - term_buffer - next_window)
        next_window += window
        window <= 0 && break
    end
    isempty(ends) && push!(ends, n_warmup - term_buffer)
    return WindowSchedule(n_warmup, init_buffer, term_buffer, ends)
end

in_adaptation_window(s::WindowSchedule, i::Int) = i > s.init_buffer && i <= s.n_warmup - s.term_buffer
is_window_end(s::WindowSchedule, i::Int) = i in s.window_ends

# --------------------------------------------------------------------------
# Shared state
# --------------------------------------------------------------------------

"""
    HamiltonianState

Position, log density, gradient, metric and adaptation state. Shared by HMC and
NUTS; the samplers differ only in `step!`.
"""
mutable struct HamiltonianState
    y::Vector{Float64}
    lp::Float64
    grad::Vector{Float64}
    metric::AbstractMetric
    da::DualAveraging
    welford::WelfordAccumulator
    schedule::WindowSchedule
    step_size::Float64
    iter::Int
    n_divergent::Int
end

function make_metric(kind::Symbol, d::Int)
    kind === :unit && return UnitMetric(d)
    kind === :diag && return DiagMetric(d)
    kind === :dense && return DenseMetric(d)
    throw(ArgumentError("metric must be :unit, :diag or :dense, got :$kind"))
end

function init_hamiltonian_state(rng::AbstractRNG, model::Model, y0::AbstractVector;
                                metric_kind::Symbol, step_size, target_accept::Real,
                                n_warmup::Int, backend::ADBackend,
                                init_buffer::Int = 75, term_buffer::Int = 50,
                                base_window::Int = 25)
    d = dimension(model)
    metric = make_metric(metric_kind, d)
    lp, grad = logdensity_and_gradient(model, collect(float.(y0)); backend = backend)
    isfinite(lp) || error("initial point has non-finite log density")
    all(isfinite, grad) || error("initial point has a non-finite gradient")
    eps = isnan(step_size) ? find_reasonable_step_size(rng, model, metric, y0, backend) : float(step_size)
    da = DualAveraging(eps; target = target_accept)
    welford = WelfordAccumulator(d; dense = metric_kind === :dense)
    schedule = WindowSchedule(n_warmup; init_buffer = init_buffer,
                              term_buffer = term_buffer, base_window = base_window)
    return HamiltonianState(collect(float.(y0)), lp, grad, metric, da, welford, schedule,
                            eps, 0, 0)
end

"""
    adapt!(rng, model, state, alpha, warmup, adapt_step_size, adapt_metric, backend)

One adaptation update: dual averaging on the acceptance statistic, plus the
windowed metric update. Called by every gradient-based sampler at the end of its
step.
"""
function adapt!(rng::AbstractRNG, model::Model, st::HamiltonianState, alpha::Real, warmup::Bool,
                adapt_step_size::Bool, adapt_metric::Bool, backend::ADBackend)
    if !warmup
        return st
    end
    st.iter += 1
    i = st.iter
    if adapt_step_size
        st.step_size = da_update!(st.da, alpha)
    end
    if adapt_metric && in_adaptation_window(st.schedule, i)
        welford_add!(st.welford, st.y)
        if is_window_end(st.schedule, i)
            est = welford_metric(st.welford)
            st.metric = est isa Matrix ? DenseMetric(est) : DiagMetric(est)
            welford_reset!(st.welford)
            if adapt_step_size
                st.step_size = find_reasonable_step_size(rng, model, st.metric, st.y, backend)
                da_restart!(st.da, st.step_size)
            end
        end
    end
    return st
end

"""
    finish_warmup_hamiltonian!(st, adapt_step_size)

Freeze the step size at the dual-averaged value for the sampling phase.
"""
function finish_warmup_hamiltonian!(st::HamiltonianState, adapt_step_size::Bool)
    if adapt_step_size && st.da.t > 0
        st.step_size = da_final(st.da)
    end
    return st
end

"""
    refresh_hamiltonian!(model, st, backend)

Recompute the log density and gradient at the current position under a new
model, keeping the step size and metric.
"""
function refresh_hamiltonian!(model::Model, st::HamiltonianState, backend::ADBackend)
    lp, grad = logdensity_and_gradient(model, st.y; backend = backend)
    st.lp = lp
    st.grad = grad
    return st
end

summary_of(st::HamiltonianState) = (step_size = st.step_size, n_divergent = st.n_divergent,
                                    metric = typeof(st.metric).name.name)
