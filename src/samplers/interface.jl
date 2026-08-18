"""
The sampler contract and the generic driver.

A sampler implements three methods and nothing else:

    init_state(rng, model, sampler, y0; n_warmup)  -> state
    step!(rng, model, sampler, state, warmup)      -> (y, stats::NamedTuple)
    finish_warmup!(rng, model, sampler, state)     -> state     (optional)

Everything else — chain allocation, initialisation, warmup bookkeeping, storing
constrained draws, threading across chains, timing — lives in [`sample`](@ref)
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
    finish_warmup!(rng, model, sampler, state)

Hook called once between warmup and sampling; the default does nothing.
"""
finish_warmup!(rng, model, sampler::AbstractSampler, state) = state

"""
    random_init(rng, model; scale = 2.0, tries = 100)

Stan's initialisation rule: draw each unconstrained coordinate uniformly on
`[-scale, scale]` and reject points where the log density or its gradient is not
finite.
"""
function random_init(rng::AbstractRNG, model::Model; scale::Real = 2.0, tries::Int = 100)
    for _ in 1:tries
        y = (2 .* rand(rng, dimension(model)) .- 1) .* scale
        lp = logdensity(model, y)
        isfinite(lp) && return y
    end
    error("could not find a finite-log-density initial point in $tries tries; supply `init` explicitly")
end

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
"""
function sample(model::Model, sampler::AbstractSampler, n_draws::Int;
                n_warmup::Int = max(n_draws ÷ 2, 100),
                n_chains::Int = 1,
                rng::AbstractRNG = Random.default_rng(),
                init = nothing,
                keep_warmup::Bool = false,
                thin::Int = 1,
                progress::Bool = false)
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
    value = Array{Float64,3}(undef, total_stored, P, n_chains)
    statbuf = Vector{Dict{Symbol,Vector{Float64}}}(undef, n_chains)
    infos = Vector{Dict{Symbol,Any}}(undef, n_chains)

    t0 = time()
    Threads.@threads for c in 1:n_chains
        crng = Random.Xoshiro(seeds[c])
        y0 = inits[c] === nothing ? random_init(crng, model) : inits[c]
        state = init_state(crng, model, sampler, y0; n_warmup = n_warmup)
        stats_c = Dict{Symbol,Vector{Float64}}()
        row = 0
        for i in 1:n_warmup
            y, st = step!(crng, model, sampler, state, true)
            if keep_warmup
                row += 1
                value[row, :, c] = flatten_draw(model, y)
                _push_stats!(stats_c, st, total_stored, row)
            end
        end
        state = finish_warmup!(crng, model, sampler, state)
        for i in 1:n_draws
            y, st = step!(crng, model, sampler, state, false)
            if (i - 1) % thin == 0
                row += 1
                value[row, :, c] = flatten_draw(model, y)
                _push_stats!(stats_c, st, total_stored, row)
            end
        end
        statbuf[c] = stats_c
        infos[c] = Dict{Symbol,Any}(:final_state => summary_of(state))
    end
    elapsed = time() - t0

    keys_all = sort(collect(union((Set(keys(s)) for s in statbuf)...)); by = String)
    stats = Dict{Symbol,Array{Float64,2}}()
    for k in keys_all
        M = Array{Float64,2}(undef, total_stored, n_chains)
        for c in 1:n_chains
            v = get(statbuf[c], k, Float64[])
            M[:, c] .= length(v) == total_stored ? v : NaN
        end
        stats[k] = M
    end
    info = Dict{Symbol,Any}(:sampler => sampler, :n_warmup => n_warmup, :thin => thin,
                            :time_seconds => elapsed, :chain_info => infos,
                            :warmup_kept => keep_warmup)
    return Chains(value, parameter_names(model), stats, info)
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
