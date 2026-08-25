module AssayTablesExt

# Chains as a table.
#
# Implementing the Tables interface is what makes the output of this package
# work with everything else: `DataFrame(chain)`, CSV writing, Query, plotting
# libraries that accept tables. It is one interface rather than one integration
# per package, which is the whole point of the interface existing.
#
# The layout is long rather than wide: one row per draw per chain, with `chain`
# and `draw` as columns alongside the parameters. That is the shape every
# grouping and plotting operation expects, and it keeps the chain index
# available rather than folded away.

using Assay: Assay, Chains
using Tables: Tables

Tables.istable(::Type{<:Chains}) = true
Tables.columnaccess(::Type{<:Chains}) = true

function Tables.columns(c::Chains)
    n, p, m = Assay.ndraws(c), Assay.nparams(c), Assay.nchains(c)
    total = n * m
    chain = Vector{Int}(undef, total)
    draw = Vector{Int}(undef, total)
    row = 0
    for ch in 1:m, i in 1:n
        row += 1
        chain[row] = ch
        draw[row] = i
    end
    cols = Any[chain, draw]
    for j in 1:p
        col = Vector{Float64}(undef, total)
        row = 0
        for ch in 1:m, i in 1:n
            row += 1
            col[row] = c.value[i, j, ch]
        end
        push!(cols, col)
    end
    names = vcat([:chain, :draw], c.names)
    return NamedTuple{Tuple(names)}(Tuple(cols))
end

Tables.schema(c::Chains) =
    Tables.Schema(vcat([:chain, :draw], c.names),
                  vcat([Int, Int], fill(Float64, Assay.nparams(c))))

Tables.rowaccess(::Type{<:Chains}) = true
Tables.rows(c::Chains) = Tables.rows(Tables.columns(c))

end # module
