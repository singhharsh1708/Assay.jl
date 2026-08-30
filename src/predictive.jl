# Working with a posterior once you have one.
#
# Fitting a model is the middle of the job, not the end. What follows is
# usually: draw new data from the fitted model, ask whether it looks like the
# data you actually saw, and compute quantities that are functions of the
# parameters rather than the parameters themselves. Without these a user has to
# reach into the chain object and rebuild the constrained parameters by hand,
# which for a simplex or a correlation matrix they cannot do from the reported
# columns at all.

"""
    parameter_draws(model, chains; thin = 1)

Iterate the posterior draws as the `NamedTuple` the model was written against,
reconstructed exactly from the unconstrained draws.

This is the reason chains retain those draws. The reported columns of a simplex
or a correlation parameter are a summary of the point rather than the point
itself, so a `NamedTuple` cannot be rebuilt from them.

    for theta in parameter_draws(model, chains)
        theta.sigma, theta.w      # exactly what the log density saw
    end
"""
function parameter_draws(model::AbstractModel, c::Chains; thin::Int = 1)
    has_unconstrained(c) ||
        throw(ArgumentError("this chain does not carry its unconstrained draws, so the " *
                            "parameters cannot be reconstructed. Chains produced by `sample` do."))
    size(c.unconstrained, 2) == dimension(model) ||
        throw(DimensionMismatch("chain is on R^$(size(c.unconstrained, 2)) but the model is on " *
                                "R^$(dimension(model))"))
    idx = [(i, ch) for ch in 1:nchains(c) for i in 1:thin:ndraws(c)]
    return (first(constrain(model, view(c.unconstrained, i, :, ch))) for (i, ch) in idx)
end

"""
    n_parameter_draws(chains; thin = 1)

How many draws [`parameter_draws`](@ref) will yield.
"""
n_parameter_draws(c::Chains; thin::Int = 1) = length(1:thin:ndraws(c)) * nchains(c)

"""
    predictive(model, chains, simulate; rng, thin = 1)

Draw one replicated data set per posterior draw, by calling
`simulate(theta, rng)`.

This is the posterior predictive distribution: parameter uncertainty is carried
through by using a different draw each time rather than a point estimate, which
is the entire difference between a Bayesian predictive interval and a plug-in
one.

    yrep = predictive(model, chains, (theta, rng) -> [rand(rng, Normal(theta.mu, theta.sigma))
                                                      for _ in 1:n])
"""
function predictive(model::AbstractModel, c::Chains, simulate;
                    rng::AbstractRNG = Random.default_rng(), thin::Int = 1)
    out = Vector{Any}()
    sizehint!(out, n_parameter_draws(c; thin = thin))
    for theta in parameter_draws(model, c; thin = thin)
        push!(out, simulate(theta, rng))
    end
    return identity.(out)          # narrow the element type once it is known
end

"""
    pointwise_log_likelihood(model, chains, loglik; thin = 1)

A `draws x observations` matrix of pointwise log likelihoods, which is what
[`loo`](@ref) consumes.

`loglik(theta, i)` returns the log likelihood of observation `i` under one
posterior draw. Building this matrix by hand is where cross validation usually
goes wrong: the draws have to be in the same order for every observation, and
the likelihood has to be pointwise rather than summed.

    ll = pointwise_log_likelihood(model, chains, (theta, i) -> logpdf(Normal(theta.mu, 1.0), y[i]),
                                  n_obs = length(y))
"""
pointwise_log_likelihood(model::AbstractModel, c::Chains, loglik; n_obs::Int, thin::Int = 1) =
    pointwise(model, c, loglik; n_obs = n_obs, thin = thin)

"""
    pointwise(model, chains, f; n_obs, thin = 1)

A `draws x observations` matrix of `f(theta, i)`.

[`pointwise_log_likelihood`](@ref) and [`pointwise_cdf`](@ref) are this with a
docstring saying which quantity `f` should return; the ordering constraint that
makes both usable is here.
"""
function pointwise(model::AbstractModel, c::Chains, f; n_obs::Int, thin::Int = 1)
    n_obs > 0 || throw(ArgumentError("n_obs must be positive"))
    S = n_parameter_draws(c; thin = thin)
    out = Matrix{Float64}(undef, S, n_obs)
    for (s, theta) in enumerate(parameter_draws(model, c; thin = thin))
        for i in 1:n_obs
            out[s, i] = f(theta, i)
        end
    end
    return out
end

"""
    PredictiveCheck

The result of a posterior predictive check: the statistic on the observed data,
its distribution over replicated data sets, and the posterior predictive p
value.
"""
struct PredictiveCheck
    observed::Float64
    replicated::Vector{Float64}
    pvalue::Float64
end

function Base.show(io::IO, r::PredictiveCheck)
    verdict = 0.05 <= r.pvalue <= 0.95 ? "consistent" : "in the tail"
    @printf(io, "PredictiveCheck(observed = %.4g, p = %.3f, %s of %d replications)",
            r.observed, r.pvalue, verdict, length(r.replicated))
end

"""
    predictive_check(observed, replicated, statistic)

Compare a statistic of the observed data against its distribution under the
posterior predictive, returning the proportion of replications at least as
extreme.

A p value near 0 or 1 says the model cannot reproduce that feature of the data.
It is not a test with a rejection threshold, and the statistic is the whole
question: a model that gets the mean right and the tails wrong will look perfect
under `mean` and fail under `maximum`, and only the second one tells you
anything you did not already assume.
"""
function predictive_check(observed, replicated::AbstractVector, statistic)
    tobs = float(statistic(observed))
    trep = [float(statistic(r)) for r in replicated]
    p = count(>=(tobs), trep) / length(trep)
    return PredictiveCheck(tobs, trep, p)
end
