"""
The sampler contract and the generic driver.

A sampler implements two methods, plus two optional hooks:

    init_state(rng, model, sampler, y0; n_warmup)  -> state       (required)
    step!(rng, model, sampler, state, warmup)      -> (y, stats)  (required)
    finish_warmup!(rng, model, sampler, state)     -> state       (optional)
    refresh!(model, sampler, state)                -> state       (optional)

Everything else - chain allocation, initialisation, warmup bookkeeping, storing
constrained draws, threading across chains, timing - lives in [`sample`](@ref)
here. That is the boundary that lets a new sampler be a new file: HMC, NUTS and
random-walk Metropolis share this driver unchanged.
"""
abstract type AbstractSampler end

"""
    init_state(rng, model, sampler, y0; n_warmup)

Build the sampler state at unconstrained starting point `y0`. `n_warmup` is
passed because adaptive samplers need to know the length of the warmup phase in
advance in order to schedule it.
"""
function init_state end

"""
    step!(rng, model, sampler, state, warmup)

Advance one iteration and return `(y, stats)` where `y` is the new unconstrained
draw and `stats` is a `NamedTuple` of per-draw diagnostics. `warmup` says
whether adaptation should update.
"""
function step! end

"""
    refresh!(model, sampler, state)

Recompute the cached log density (and gradient) of `state` under a *different*
model at the same position, leaving every adapted quantity alone.

This exists for the Geweke test, which alternates between data sets: the
transition kernel has to stay fixed across the sweep, because a kernel whose
step size is re-chosen from the current position is not reversible and the test
would then be measuring that violation rather than the sampler. Discovering
exactly that is what made this function necessary.
"""
function refresh! end

"""
    finish_warmup!(rng, model, sampler, state)

Hook called once between warmup and sampling; the default does nothing.
"""
finish_warmup!(rng, model, sampler::AbstractSampler, state) = state

"""
    check_model(model, y)

Evaluate the log density twice at the same point and complain if the two
answers differ.

A log density that is not a function of its argument alone cannot be sampled by
anything, and the usual cause is data generated inside the closure rather than
outside it:

    Model(params, theta -> loglikelihood(Normal(theta.mu, 1), randn(30)))   # wrong
    data = randn(30)
    Model(params, theta -> loglikelihood(Normal(theta.mu, 1), data))        # right

Without this check the symptom is an acceptance rate near zero and an effective
sample size of two, which reads like a badly tuned sampler rather than a broken
model. The check costs two evaluations per run.
"""
function check_model(model::AbstractModel, y::AbstractVector)
    first_value = logdensity(model, y)
    second_value = logdensity(model, y)
    if !isequal(first_value, second_value)
        throw(NonDeterministicModelError(collect(float.(y)), first_value, second_value))
    end
    isfinite(first_value) || throw(NonFiniteDensityError(y, first_value, :logdensity))
    return model
end

"""
    random_init(rng, model; scale = 2.0, tries = 100)

Stan's initialisation rule: draw each unconstrained coordinate uniformly on
`[-scale, scale]` and reject points where the log density or its gradient is not
finite.
"""
function random_init(rng::AbstractRNG, model::AbstractModel; scale::Real = 2.0, tries::Int = 100)
    d = dimension(model)
    y = zeros(d)
    lp = -Inf
    for _ in 1:tries
        y = (2 .* rand(rng, d) .- 1) .* scale
        lp = logdensity(model, y)
        isfinite(lp) && return y
    end
    throw(InitialisationError(tries, float(scale), d, y, lp))
end

"""
    ChainFailure

One chain that did not finish: which chain, how far it got, the error, and the
draws it managed before dying.

A log density that throws is not an exotic case. It indexes past the end of an
array for one parameter value in a million, or takes the log of something that
has just gone negative, or calls a solver that fails to converge. A run that hit
one of those returns the chains that survived and one of these for each that did
not, rather than losing the lot.

`last_position` is the point the chain was at when it died, which is the field
that makes the failure reproducible: feed it back into the log density and watch
it happen again. It tracks every step rather than every stored draw, so it is
still the right point under thinning, and it is the initial point when the chain
never took a step at all.
"""
struct ChainFailure
    chain::Int
    phase::Symbol                  # :initialisation, :warmup or :sampling
    iteration::Int
    error::Any
    last_position::Vector{Float64}
    value::Matrix{Float64}         # constrained draws kept before the failure
    unconstrained::Matrix{Float64}
end

function Base.show(io::IO, f::ChainFailure)
    @printf(io, "ChainFailure(chain %d, %s iteration %d, %d draws kept, %s)",
            f.chain, f.phase, f.iteration, size(f.value, 1), typeof(f.error))
end

"""
    failures(chains)

The chains that did not finish, as [`ChainFailure`](@ref) values. Empty when the
run was clean.
"""
failures(c::Chains) = get(c.info, :failures, ChainFailure[])

"""
    failed(chains)

Whether any chain of the run died. A run that lost chains is still returned, so
this is the way to find out.
"""
failed(c::Chains) = !isempty(failures(c))

"""
    sample(model, sampler, n_draws; kwargs...)

Run `sampler` on `model` and return [`Chains`](@ref) of constrained draws.

Keyword arguments:

  * `n_warmup`     : adaptation iterations, discarded unless `keep_warmup`
  * `n_chains`     : independent chains, run in parallel when Julia has threads
  * `rng`          : seeds each chain reproducibly
  * `init`         : unconstrained starting point, or a vector of one per chain
  * `keep_warmup`  : also store the warmup draws
  * `thin`         : keep every `thin`-th draw after warmup
  * `progress`     : report progress as log messages, see [`ProgressReporter`](@ref)
  * `progress_interval` : seconds between the throttled progress lines
"""
function sample(model::AbstractModel, sampler::AbstractSampler, n_draws::Int;
                n_warmup::Int = max(n_draws ÷ 2, 100),
                n_chains::Int = 1,
                rng::AbstractRNG = Random.default_rng(),
                init = nothing,
                keep_warmup::Bool = false,
                thin::Int = 1,
                progress::Bool = false,
                progress_interval::Real = 10.0)
    n_draws > 0 || throw(ArgumentError("n_draws must be positive"))
    thin >= 1 || throw(ArgumentError("thin must be at least 1"))
    seeds = rand(rng, UInt64, n_chains)
    inits = if init === nothing
        [nothing for _ in 1:n_chains]
    elseif init isa AbstractVector{<:Real}
        [collect(float.(init)) for _ in 1:n_chains]
    else
        collect(init)
    end
    kept = length(1:thin:n_draws)
    total_stored = keep_warmup ? n_warmup + kept : kept
    P = flat_dimension(model)
    D = dimension(model)
    value = Array{Float64,3}(undef, total_stored, P, n_chains)
    raw = Array{Float64,3}(undef, total_stored, D, n_chains)
    statbuf = Vector{Dict{Symbol,Vector{Float64}}}(undef, n_chains)
    infos = Vector{Dict{Symbol,Any}}(undef, n_chains)

    failures = Vector{Union{Nothing,ChainFailure}}(nothing, n_chains)
    reporter = ProgressReporter(n_chains * (n_warmup + n_draws); on = progress,
                                interval = progress_interval)

    t0 = time()
    Threads.@threads for c in 1:n_chains
        stats_c = Dict{Symbol,Vector{Float64}}()
        statbuf[c] = stats_c
        infos[c] = Dict{Symbol,Any}()
        # Declared out here so the catch block can say where the chain was.
        # `pos` follows every step rather than every stored draw: under thinning
        # the last stored draw is not the point that reproduces the failure.
        row = 0
        phase = :initialisation
        iter = 0
        pos = Float64[]
        try
            crng = Random.Xoshiro(seeds[c])
            y0 = inits[c] === nothing ? random_init(crng, model) : inits[c]
            pos = collect(float.(y0))
            c == 1 && check_model(model, y0)
            state = init_state(crng, model, sampler, y0; n_warmup = n_warmup)
            phase = :warmup
            for i in 1:n_warmup
                iter = i
                y, st = step!(crng, model, sampler, state, true)
                pos = y
                tick!(reporter)
                if keep_warmup
                    row += 1
                    value[row, :, c] = flatten_draw(model, y)
                    raw[row, :, c] = y
                    _push_stats!(stats_c, st, total_stored, row)
                end
            end
            state = finish_warmup!(crng, model, sampler, state)
            phase = :sampling
            for i in 1:n_draws
                iter = i
                y, st = step!(crng, model, sampler, state, false)
                pos = y
                tick!(reporter)
                if (i - 1) % thin == 0
                    row += 1
                    value[row, :, c] = flatten_draw(model, y)
                    raw[row, :, c] = y
                    _push_stats!(stats_c, st, total_stored, row)
                end
            end
            infos[c] = Dict{Symbol,Any}(:final_state => summary_of(state))
        catch e
            e isa InterruptException && rethrow()
            failures[c] = ChainFailure(c, phase, iter, e, collect(float.(pos)),
                                       row > 0 ? value[1:row, :, c] : zeros(0, P),
                                       row > 0 ? raw[1:row, :, c] : zeros(0, D))
        end
    end
    elapsed = time() - t0
    finish!(reporter)

    kept_chains = [c for c in 1:n_chains if failures[c] === nothing]
    broke = ChainFailure[failures[c] for c in 1:n_chains if failures[c] !== nothing]
    if isempty(kept_chains)
        # Nothing came back, so there is no partial result to hand over and the
        # honest thing is the error itself rather than an empty container.
        throw(first(broke).error)
    end
    if !isempty(broke)
        @warn "$(length(broke)) of $n_chains chains failed and were dropped; " *
              "`failures(chains)` has the errors and the draws each one managed."
    end

    keys_all = sort(collect(union((Set(keys(s)) for s in statbuf)...)); by = String)
    stats = Dict{Symbol,Array{Float64,2}}()
    for k in keys_all
        M = Array{Float64,2}(undef, total_stored, length(kept_chains))
        for (j, c) in enumerate(kept_chains)
            v = get(statbuf[c], k, Float64[])
            M[:, j] .= length(v) == total_stored ? v : NaN
        end
        stats[k] = M
    end
    info = Dict{Symbol,Any}(:sampler => sampler, :n_warmup => n_warmup, :thin => thin,
                            :time_seconds => elapsed, :chain_info => infos[kept_chains],
                            :warmup_kept => keep_warmup, :failures => broke,
                            :n_chains_requested => n_chains)
    # Slicing would copy the whole draw array, which for a large run is the
    # single biggest allocation the sampler makes. Nothing failed, nothing to
    # slice.
    clean = length(kept_chains) == n_chains
    return Chains(clean ? value : value[:, :, kept_chains], parameter_names(model), stats, info,
                  clean ? raw : raw[:, :, kept_chains])
end

function _push_stats!(d::Dict{Symbol,Vector{Float64}}, st::NamedTuple, n::Int, row::Int)
    for (k, v) in pairs(st)
        arr = get!(d, k) do
            fill(NaN, n)
        end
        arr[row] = float(v)
    end
    return d
end

"""
    summary_of(state)

Small `NamedTuple` describing the adapted state of a chain (step size, mass
matrix scale, ...). Samplers override it; the default reports nothing.
"""
summary_of(state) = NamedTuple()
