# A discrete-time dynamical system whose state is a point on the probability
# simplex, with the state observed only through counts.
#
#     x_{t+1} = normalise(x_t .* exp(f)),      y_t ~ Multinomial(n, x_t)
#
# This is the replicator map of evolutionary dynamics, and it is here because it
# is the smallest problem that puts real pressure on the transform layer: the
# initial state is a simplex parameter, the dynamics must keep the state on the
# simplex for every parameter value the sampler proposes, and the fitness vector
# is identifiable only up to an additive constant.
#
# The identifiability constraint is handled by pinning the first component of `f`
# to zero rather than by adding a soft prior that pretends to identify it. A
# model that is unidentified by construction produces a posterior with a ridge,
# and no amount of sampler quality fixes that.

"""
    simplex_step(x, f)

One step of the replicator map: reweight by `exp(f)` and renormalise. The result
is on the simplex for any finite `f`, which is what makes the dynamics safe to
put inside a log density that a sampler will evaluate at arbitrary proposals.
"""
function simplex_step(x::AbstractVector, f::AbstractVector)
    w = x .* exp.(f .- maximum(f))          # shift for numerical safety; the map is shift invariant
    return w ./ sum(w)
end

"""
    simplex_trajectory(x0, f, T)

States `x_1 = x0, ..., x_T` under [`simplex_step`](@ref).
"""
function simplex_trajectory(x0::AbstractVector, f::AbstractVector, T::Int)
    states = Vector{typeof(x0 ./ 1)}(undef, T)
    x = x0 ./ sum(x0)
    for t in 1:T
        states[t] = x
        x = simplex_step(x, f)
    end
    return states
end

"""
    replicator_model(counts, n; alpha = 1.0, fitness_scale = 1.0)

Model for count observations of a replicator trajectory. Parameters are the
initial state on the `K`-simplex and `K - 1` free fitness components, the first
being pinned to zero for identifiability.

`counts[t]` is the vector of `K` counts observed at time `t`, summing to `n`.
"""
function replicator_model(counts::Vector{<:AbstractVector}, n::Int; alpha::Real = 1.0,
                          fitness_scale::Real = 1.0)
    K = length(first(counts))
    T = length(counts)
    return Model((x0 = simplex(K), f = unconstrained(K - 1)),
                 function (theta)
                     lp = logpdf(Dirichlet(K, alpha), theta.x0)
                     for fi in theta.f
                         lp += logpdf(Normal(0.0, fitness_scale), fi)
                     end
                     fvec = vcat(zero(eltype(theta.f)), theta.f)
                     x = theta.x0
                     for t in 1:T
                         lp += logpdf(Multinomial(n, x), counts[t])
                         t < T && (x = simplex_step(x, fvec))
                     end
                     return lp
                 end)
end

"""
    replicator_problem(K, T, n; alpha = 1.0, fitness_scale = 1.0)

The same model packaged as a [`CalibrationProblem`](@ref), so that simulation
based calibration can be run on it. There is no closed-form posterior here; the
calibration check is the verification.
"""
function replicator_problem(K::Int, T::Int, n::Int; alpha::Real = 1.0, fitness_scale::Real = 1.0)
    prior_rand = function (rng)
        return (x0 = rand(rng, Dirichlet(K, alpha)),
                f = [rand(rng, Normal(0.0, fitness_scale)) for _ in 1:(K - 1)])
    end
    simulate = function (theta, rng)
        fvec = vcat(0.0, theta.f)
        states = simplex_trajectory(theta.x0, fvec, T)
        return [rand(rng, Multinomial(n, x)) for x in states]
    end
    build = counts -> replicator_model(counts, n; alpha = alpha, fitness_scale = fitness_scale)
    return CalibrationProblem(build, prior_rand, simulate)
end
