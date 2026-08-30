# Cross validation that refits, for when importance sampling gives up.
#
# `loo` reports a Pareto shape per observation and says plainly that anything
# above 0.7 should not be believed. Until now it offered no way to get a number
# that can be. That is an uncomfortable place to leave a user: the diagnostic
# that makes the estimate trustworthy is also the one that tells them the
# estimate is unusable, with nothing after it.
#
# K-fold refits, so there is no importance sampling and nothing to diagnose. It
# costs k fits instead of one, which is the whole reason `loo` exists, so the
# useful version is not "refit everything" but "refit the handful of
# observations PSIS could not handle and keep its answer for the rest". Both are
# here.

"""
    KFoldResult

Expected log pointwise predictive density by k-fold cross validation.

The same quantity [`loo`](@ref) estimates, computed by refitting rather than by
reweighting, so there is no `khat`: nothing was approximated. `refits` records
what it cost.
"""
struct KFoldResult
    elpd::Float64
    se::Float64
    pointwise::Vector{Float64}
    folds::Vector{Vector{Int}}
    refits::Int
end

function Base.show(io::IO, r::KFoldResult)
    @printf(io, "KFoldResult(elpd = %.2f ± %.2f, %d folds, %d observations)",
            r.elpd, r.se, r.refits, length(r.pointwise))
end

"""
    stratified_folds(n_obs, k; rng)

Split `1:n_obs` into `k` folds of nearly equal size, in random order.

Random rather than contiguous: contiguous folds on data that arrived in some
order are folds that share whatever that order encodes, and the resulting
estimate is of predicting a different data set from the one the user has.
"""
function stratified_folds(n_obs::Int, k::Int; rng::AbstractRNG = Random.default_rng())
    2 <= k <= n_obs ||
        throw(ArgumentError("need 2 <= k <= n_obs, got k = $k and n_obs = $n_obs"))
    perm = Random.randperm(rng, n_obs)
    return [perm[i:k:n_obs] for i in 1:k]
end

"""
    kfold(rng, build, loglik, n_obs; k = 10, folds = nothing, sampler = NUTS(), ...)

K-fold cross validation, refitting once per fold.

`build(train)` returns the model fitted to the observations whose indices are in
`train`, and `loglik(theta, i)` is the log likelihood of observation `i` under
one draw, indexed into the whole data set rather than into the fold. Keeping
those two separate is what lets the held-out observations be scored by a model
that never saw them, which is the only thing cross validation is.

Pass `folds` directly to control the split, which is how
[`refit_problematic`](@ref) reuses this.
"""
function kfold(rng::AbstractRNG, build, loglik, n_obs::Int; k::Int = 10,
               folds::Union{Nothing,AbstractVector} = nothing,
               sampler::AbstractSampler = NUTS(), n_draws::Int = 1000,
               n_warmup::Int = 500, n_chains::Int = 2)
    fs = folds === nothing ? stratified_folds(n_obs, k; rng = rng) :
         [collect(f) for f in folds]
    isempty(fs) && throw(ArgumentError("no folds to fit"))
    all_idx = reduce(vcat, fs)
    length(unique(all_idx)) == length(all_idx) ||
        throw(ArgumentError("folds overlap: an observation held out twice is scored twice"))
    # Validated here rather than inside the loop below: that loop is threaded,
    # so an error raised in it arrives wrapped and cannot be dispatched on.
    for f in eachindex(fs)
        length(fs[f]) < n_obs ||
            throw(ArgumentError("fold $f holds out every observation, leaving nothing to fit"))
    end

    pointwise = fill(NaN, n_obs)
    seeds = rand(rng, UInt64, length(fs))
    Threads.@threads for f in eachindex(fs)
        held = fs[f]
        train = setdiff(1:n_obs, held)
        r = Random.Xoshiro(seeds[f])
        model = build(train)
        chn = sample(model, sampler, n_draws; n_warmup = n_warmup, n_chains = n_chains, rng = r)
        thetas = collect(parameter_draws(model, chn))
        logS = log(length(thetas))
        for i in held
            pointwise[i] = logsumexp([loglik(theta, i) for theta in thetas]) - logS
        end
    end

    scored = findall(!isnan, pointwise)
    elpd = sum(view(pointwise, scored))
    se = sqrt(length(scored) * Statistics.var(view(pointwise, scored)))
    return KFoldResult(elpd, se, pointwise, fs, length(fs))
end

kfold(build, loglik, n_obs::Int; kwargs...) =
    kfold(Random.default_rng(), build, loglik, n_obs; kwargs...)

"""
    refit_problematic(rng, r::LOOResult, build, loglik; threshold = 0.7, ...)

Refit exactly the observations importance sampling could not handle, and keep
the leave-one-out estimate for the rest.

This is the version worth running. A model with one influential observation
needs one refit, not `n` of them, and the result is exact where PSIS failed and
unchanged where it did not. Returns a [`LOOResult`](@ref) whose `khat` is set to
zero for the refitted observations, because nothing about them was approximated
any more.
"""
function refit_problematic(rng::AbstractRNG, r::LOOResult, build, loglik;
                           threshold::Real = 0.7, sampler::AbstractSampler = NUTS(),
                           n_draws::Int = 1000, n_warmup::Int = 500, n_chains::Int = 2)
    bad = problematic(r; threshold = threshold)
    isempty(bad) && return r
    n_obs = length(r.pointwise)
    exact = kfold(rng, build, loglik, n_obs; folds = [[i] for i in bad], sampler = sampler,
                  n_draws = n_draws, n_warmup = n_warmup, n_chains = n_chains)

    pointwise = copy(r.pointwise)
    khat = copy(r.khat)
    for i in bad
        pointwise[i] = exact.pointwise[i]
        khat[i] = 0.0
    end
    elpd = sum(pointwise)
    se = sqrt(n_obs * Statistics.var(pointwise))
    # p_loo moves with the pointwise values it is defined from, so recompute the
    # part that changed rather than carrying the old number forward
    p_loo = r.p_loo + (sum(r.pointwise) - elpd)
    return LOOResult(elpd, se, p_loo, pointwise, khat, r.n_draws)
end

refit_problematic(r::LOOResult, build, loglik; kwargs...) =
    refit_problematic(Random.default_rng(), r, build, loglik; kwargs...)
