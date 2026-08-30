# Leave-one-out probability integral transforms.
#
# A posterior predictive check compares a statistic of the data against its
# distribution under the fitted model, and it has a known weakness: the data it
# is checking is the data the model was fitted to, so the model has already been
# pulled towards it and the check is optimistic.
#
# LOO-PIT asks a sharper question. For each observation, where does it sit in
# the predictive distribution of a model fitted without it? Those positions
# should be uniform on (0, 1). They catch a model that is right on average and
# wrong observation by observation, which is what a check on the mean of the
# whole data set cannot see.
#
# The importance weights that make the leave-one-out predictive available are
# the ones `loo` already computes, so this costs one extra quantity from the
# user: the predictive cdf at each observation under each draw.

"""
    LOOPITResult

Leave-one-out probability integral transforms, the verdict on their uniformity,
and the Pareto shapes from the importance sampling that produced them.

`khat` matters here for the same reason it matters in [`loo`](@ref): a PIT value
computed from weights the smoothing could not fix is not a PIT value, and
`problematic(r)` says which ones those are.
"""
struct LOOPITResult
    pit::Vector{Float64}
    inside::Bool
    max_deviation::Float64
    khat::Vector{Float64}
    grid::Vector{Float64}
    ecdf::Vector{Float64}
    lower::Vector{Float64}
    upper::Vector{Float64}
end

problematic(r::LOOPITResult; threshold::Real = 0.7) = findall(>=(threshold), r.khat)

"""
    calibrated(r::LOOPITResult)

Whether the transforms stayed inside the simultaneous uniformity band.
"""
calibrated(r::LOOPITResult) = r.inside

function Base.show(io::IO, r::LOOPITResult)
    verdict = r.inside ? "uniform" : "not uniform"
    @printf(io, "LOOPITResult(%d observations, %s, largest deviation %.3f",
            length(r.pit), verdict, r.max_deviation)
    bad = length(problematic(r))
    return bad == 0 ? print(io, ")") : @printf(io, ", %d with k >= 0.7)", bad)
end

"""
    loo_pit(log_likelihood, predictive_cdf; confidence = 0.95, rng = nothing,
            predictive_cdf_lower = nothing)

Leave-one-out probability integral transforms from a `draws x observations`
matrix of pointwise log likelihoods and the matching matrix of predictive cdf
values, `P(y_new <= y_i | theta_s)`.

Each transform is the importance weighted average of the cdf values for that
observation, with the same Pareto smoothed weights [`loo`](@ref) uses. Uniformity
is judged against the simultaneous band from [`uniformity_ecdf`](@ref), which is
the band whose five percent means five percent across the whole curve rather
than at each point separately.

For discrete observations the transform is not uniform even under a perfect
model, because the cdf jumps: the value lands on a lattice. Pass
`predictive_cdf_lower`, the cdf just below each observation, and an `rng`, and
the transform is randomised uniformly across the jump, which restores
uniformity. Leaving them out for discrete data produces a result that fails for
a reason that has nothing to do with the model.
"""
function loo_pit(log_likelihood::AbstractMatrix{<:Real}, predictive_cdf::AbstractMatrix{<:Real};
                 confidence::Real = 0.95, n_grid::Int = 100,
                 rng::Union{Nothing,AbstractRNG} = nothing,
                 predictive_cdf_lower::Union{Nothing,AbstractMatrix{<:Real}} = nothing)
    size(log_likelihood) == size(predictive_cdf) ||
        throw(DimensionMismatch("log likelihood is $(size(log_likelihood)) and the predictive " *
                                "cdf is $(size(predictive_cdf)); they describe the same draws " *
                                "and the same observations"))
    S, n = size(log_likelihood)
    S >= 10 || throw(ArgumentError("need at least 10 draws, got $S"))
    randomise = predictive_cdf_lower !== nothing
    randomise && rng === nothing &&
        throw(ArgumentError("randomising the transform for discrete data needs an `rng`"))
    randomise && size(predictive_cdf_lower) != size(predictive_cdf) &&
        throw(DimensionMismatch("the lower cdf must have the same shape as the cdf"))

    pit = Vector{Float64}(undef, n)
    khat = Vector{Float64}(undef, n)
    for i in 1:n
        ll = view(log_likelihood, :, i)
        r = psis(-ll)                       # the leave-one-out weights, as in `loo`
        w = exp.(r.log_weights)
        if randomise
            u = rand(rng)
            upper = view(predictive_cdf, :, i)
            lower = view(predictive_cdf_lower, :, i)
            pit[i] = sum(w .* (lower .+ u .* (upper .- lower)))
        else
            pit[i] = sum(w .* view(predictive_cdf, :, i))
        end
        khat[i] = r.k
    end
    pit .= clamp.(pit, 0.0, 1.0)

    u = uniformity_ecdf(pit; n_grid = n_grid, confidence = confidence)
    return LOOPITResult(pit, u.inside, u.max_deviation, khat, u.grid, u.ecdf, u.lower, u.upper)
end

"""
    pointwise_cdf(model, chains, predictive_cdf; n_obs, thin = 1)

A `draws x observations` matrix of predictive cdf values, the second argument
[`loo_pit`](@ref) needs.

`predictive_cdf(theta, i)` returns `P(y_new <= y_i)` for observation `i` under
one posterior draw. This mirrors [`pointwise_log_likelihood`](@ref) and shares
its reason for existing: the draws have to be in the same order for every
observation, and building the two matrices separately by hand is where that goes
wrong.
"""
function pointwise_cdf(model::AbstractModel, c::Chains, predictive_cdf; n_obs::Int, thin::Int = 1)
    n_obs > 0 || throw(ArgumentError("n_obs must be positive"))
    S = n_parameter_draws(c; thin = thin)
    out = Matrix{Float64}(undef, S, n_obs)
    for (s, theta) in enumerate(parameter_draws(model, c; thin = thin))
        for i in 1:n_obs
            out[s, i] = predictive_cdf(theta, i)
        end
    end
    return out
end
