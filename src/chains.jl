"""
Storage for sampler output.

Draws are held in constrained space as `draws x parameters x chains`, with the
per-draw sampler statistics (acceptance probability, divergence flag, tree
depth, ...) kept alongside in the same shape. Diagnostics read this object and
nothing else, so a new sampler becomes visible to every diagnostic simply by
filling in the same fields.
"""
struct Chains
    value::Array{Float64,3}
    names::Vector{Symbol}
    stats::Dict{Symbol,Array{Float64,2}}
    info::Dict{Symbol,Any}
end

"""
    ndraws(chains)

Number of stored draws per chain.
"""
ndraws(c::Chains) = size(c.value, 1)

"""
    nparams(chains)

Number of scalar parameter columns.
"""
nparams(c::Chains) = size(c.value, 2)

"""
    nchains(chains)

Number of chains.
"""
nchains(c::Chains) = size(c.value, 3)

"""
    getindex(chains, name::Symbol)

The `draws x chains` matrix for one parameter column.
"""
function Base.getindex(c::Chains, name::Symbol)
    i = findfirst(==(name), c.names)
    i === nothing && throw(KeyError(name))
    return c.value[:, i, :]
end
Base.getindex(c::Chains, i::Integer) = c.value[:, i, :]

"""
    vec_of_draws(chains, name)

All draws of one parameter, chains concatenated.
"""
vec_of_draws(c::Chains, name::Symbol) = vec(c[name])

"""
    sampler_stat(chains, name)

A per-draw sampler statistic as a `draws x chains` matrix, e.g. `:accept_prob`,
`:divergent`, `:treedepth`, `:step_size`, `:energy`.
"""
sampler_stat(c::Chains, name::Symbol) = c.stats[name]

"""
    divergences(chains)

Number of divergent transitions across all chains, or `0` for samplers that do
not report them. Use `haskey(chains.stats, :divergent)` to tell the two apart.
"""
divergences(c::Chains) = haskey(c.stats, :divergent) ? Int(sum(c.stats[:divergent])) : 0

"""
    acceptance_rate(chains)

Mean acceptance probability across all draws.
"""
acceptance_rate(c::Chains) = haskey(c.stats, :accept_prob) ? Statistics.mean(c.stats[:accept_prob]) : NaN

"""
    ChainSummary

Per-parameter posterior summary and convergence diagnostics, as produced by
[`summarize`](@ref).
"""
struct ChainSummary
    names::Vector{Symbol}
    mean::Vector{Float64}
    std::Vector{Float64}
    mcse::Vector{Float64}
    q025::Vector{Float64}
    q500::Vector{Float64}
    q975::Vector{Float64}
    ess_bulk::Vector{Float64}
    ess_tail::Vector{Float64}
    rhat::Vector{Float64}
    extra::Dict{Symbol,Any}
end

"""
    summarize(chains)

Posterior mean, standard deviation, Monte Carlo standard error, quantiles, bulk
and tail effective sample size, and rank-normalised split R-hat, all computed by
this package (see `src/diagnostics.jl`).
"""
function summarize(c::Chains)
    p = nparams(c)
    m = Vector{Float64}(undef, p); s = similar(m); mc = similar(m)
    q1 = similar(m); q2 = similar(m); q3 = similar(m)
    eb = similar(m); et = similar(m); rh = similar(m)
    for j in 1:p
        x = c.value[:, j, :]
        v = vec(x)
        m[j] = Statistics.mean(v)
        s[j] = Statistics.std(v)
        q1[j] = Statistics.quantile(v, 0.025)
        q2[j] = Statistics.quantile(v, 0.5)
        q3[j] = Statistics.quantile(v, 0.975)
        eb[j] = ess_bulk(x)
        et[j] = ess_tail(x)
        rh[j] = rhat(x)
        mc[j] = s[j] / sqrt(max(eb[j], 1.0))
    end
    # Only report a statistic the sampler actually produced. Variational draws
    # have no acceptance probability, and printing `NaN` for one invites the
    # reader to wonder whether something failed.
    extra = Dict{Symbol,Any}()
    haskey(c.stats, :divergent) && (extra[:divergences] = divergences(c))
    haskey(c.stats, :accept_prob) && (extra[:acceptance_rate] = acceptance_rate(c))
    haskey(c.info, :time_seconds) && (extra[:time_seconds] = c.info[:time_seconds])
    return ChainSummary(copy(c.names), m, s, mc, q1, q2, q3, eb, et, rh, extra)
end

function Base.show(io::IO, s::ChainSummary)
    @printf(io, "%-12s %10s %10s %10s %10s %10s %10s %9s %9s %7s\n",
            "parameter", "mean", "std", "mcse", "2.5%", "50%", "97.5%", "ess_bulk", "ess_tail", "rhat")
    for i in eachindex(s.names)
        @printf(io, "%-12s %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %9.0f %9.0f %7.3f\n",
                String(s.names[i]), s.mean[i], s.std[i], s.mcse[i], s.q025[i], s.q500[i],
                s.q975[i], s.ess_bulk[i], s.ess_tail[i], s.rhat[i])
    end
    parts = String[]
    haskey(s.extra, :divergences) && push!(parts, @sprintf("divergences: %d", s.extra[:divergences]))
    haskey(s.extra, :acceptance_rate) &&
        push!(parts, @sprintf("mean accept: %.3f", s.extra[:acceptance_rate]))
    haskey(s.extra, :time_seconds) && push!(parts, @sprintf("time: %.2fs", s.extra[:time_seconds]))
    isempty(parts) || println(io, join(parts, "   "))
end

function Base.show(io::IO, c::Chains)
    print(io, "Chains(", ndraws(c), " draws x ", nparams(c), " params x ", nchains(c), " chains)\n")
    show(io, summarize(c))
end
