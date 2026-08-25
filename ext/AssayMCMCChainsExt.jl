module AssayMCMCChainsExt

# Conversion to MCMCChains, which is the currency of the Julia Bayesian
# ecosystem: StatsPlots recipes, ArviZ, and most published analysis code accept
# it. Converting rather than adopting it as the native container keeps this
# package's own diagnostics and sampler statistics first class, and still lets
# a user hand the result to tooling they already have.

using Assay: Assay, Chains
using MCMCChains: MCMCChains

"""
    MCMCChains.Chains(c::Assay.Chains)

Convert to an MCMCChains object, carrying the sampler statistics across as an
`:internals` section so that `describe`, the plotting recipes and ArviZ all see
divergences, tree depth and step size alongside the parameters.
"""
function MCMCChains.Chains(c::Chains)
    n, p, m = Assay.ndraws(c), Assay.nparams(c), Assay.nchains(c)
    statnames = sort(collect(keys(c.stats)); by = String)
    total = p + length(statnames)
    value = Array{Float64,3}(undef, n, total, m)
    value[:, 1:p, :] = c.value
    for (k, name) in enumerate(statnames)
        value[:, p + k, :] = c.stats[name]
    end
    names = vcat(c.names, statnames)
    sections = Dict(:parameters => c.names, :internals => statnames)
    return MCMCChains.Chains(value, names, sections)
end

end # module
