# Sum-product networks.
#
# An SPN is a rooted directed acyclic graph of sums, products and univariate
# leaves. Its value at a point is computed in one upward pass, and - this is the
# reason the model class exists - so is any marginal, provided the graph
# satisfies two structural conditions:
#
#   * completeness: every child of a sum node has the same scope;
#   * decomposability: the children of a product node have disjoint scopes.
#
# Under those conditions a sum node is a mixture and a product node is an
# independent factorisation, so marginalising a variable is exactly setting its
# leaves to 1. Both conditions are checked at construction here, because a graph
# that violates them still evaluates to a number - just not to the number the
# user thinks it is.
#
# The connection to the rest of the package is the sum weights: they live on a
# simplex, which is precisely what `simplex(K)` and its stick-breaking transform
# exist for, so inferring them is an ordinary model. `test/test_spn.jl` runs
# simulation based calibration on that inference, which is a non-conjugate model
# with a latent mixture structure and no closed form to check against.

"""
    SPNNode

A node of a sum-product network: [`LeafNode`](@ref), [`SumNode`](@ref) or
[`ProductNode`](@ref).
"""
abstract type SPNNode end

"""
    LeafNode(variable, density)

A univariate density over one variable index.
"""
struct LeafNode{D<:UnivariateDensity} <: SPNNode
    variable::Int
    density::D
end

"""
    SumNode(children, weights)

Mixture over children that share a scope. `weights` must be non-negative and sum
to one; they are stored in logs.
"""
struct SumNode{T<:Real} <: SPNNode
    children::Vector{SPNNode}
    logweights::Vector{T}
end

"""
    ProductNode(children)

Independent factorisation over children with disjoint scopes.
"""
struct ProductNode <: SPNNode
    children::Vector{SPNNode}
end

"""
    scope(node)

Sorted vector of variable indices the node depends on.
"""
scope(n::LeafNode) = [n.variable]
scope(n::SumNode) = sort(unique(vcat((scope(c) for c in n.children)...)))
scope(n::ProductNode) = sort(unique(vcat((scope(c) for c in n.children)...)))

"""
    sum_node(children, weights)

Construct a [`SumNode`](@ref), checking completeness and that the weights are a
probability vector.
"""
function sum_node(children::Vector{<:SPNNode}, weights::AbstractVector{T}) where {T<:Real}
    length(children) == length(weights) ||
        throw(ArgumentError("a sum node needs one weight per child"))
    all(>=(0), weights) || throw(ArgumentError("sum node weights must be non-negative"))
    isapprox(sum(weights), 1; atol = 1e-8) ||
        throw(ArgumentError("sum node weights must sum to 1, got $(sum(weights))"))
    scopes = [scope(c) for c in children]
    all(==(scopes[1]), scopes) ||
        throw(ArgumentError("sum node is not complete: children have scopes $scopes"))
    return SumNode(Vector{SPNNode}(children), log.(weights))
end

"""
    product_node(children)

Construct a [`ProductNode`](@ref), checking decomposability.
"""
function product_node(children::Vector{<:SPNNode})
    scopes = [scope(c) for c in children]
    for i in eachindex(scopes), j in (i + 1):length(scopes)
        isempty(intersect(scopes[i], scopes[j])) ||
            throw(ArgumentError("product node is not decomposable: scopes $(scopes[i]) and " *
                                "$(scopes[j]) overlap"))
    end
    return ProductNode(Vector{SPNNode}(children))
end

"""
    logpdf(node, x)

Log density of the network at `x`. Entries of `x` may be `missing`, in which case
that variable is marginalised out exactly: a missing leaf contributes zero to the
log value, which is the whole point of the model class.
"""
function logpdf(n::LeafNode, x::AbstractVector)
    v = x[n.variable]
    v === missing && return 0.0
    return logpdf(n.density, v)
end

function logpdf(n::SumNode, x::AbstractVector)
    vals = (n.logweights[i] + logpdf(n.children[i], x) for i in eachindex(n.children))
    return logsumexp(collect(vals))
end

function logpdf(n::ProductNode, x::AbstractVector)
    return sum(logpdf(c, x) for c in n.children)
end

"""
    rand(rng, node, d)

Ancestral sample of length `d`: descend from the root, choosing one child at each
sum node in proportion to its weight and visiting every child of a product node.
"""
function Base.rand(rng::AbstractRNG, n::SPNNode, d::Int)
    x = Vector{Union{Missing,Float64}}(missing, d)
    _sample_into!(rng, n, x)
    return Vector{Float64}(x)
end

function _sample_into!(rng::AbstractRNG, n::LeafNode, x)
    x[n.variable] = rand(rng, n.density)
    return x
end
function _sample_into!(rng::AbstractRNG, n::SumNode, x)
    u = log(rand(rng))
    acc = -Inf
    for i in eachindex(n.children)
        acc = logsumexp((acc, n.logweights[i]))
        if u <= acc
            return _sample_into!(rng, n.children[i], x)
        end
    end
    return _sample_into!(rng, n.children[end], x)
end
function _sample_into!(rng::AbstractRNG, n::ProductNode, x)
    for c in n.children
        _sample_into!(rng, c, x)
    end
    return x
end

"""
    n_sum_nodes(node)

Number of sum nodes in the network, i.e. the number of weight vectors that would
have to be inferred.
"""
n_sum_nodes(n::LeafNode) = 0
n_sum_nodes(n::SumNode) = 1 + sum(n_sum_nodes, n.children; init = 0)
n_sum_nodes(n::ProductNode) = sum(n_sum_nodes, n.children; init = 0)

"""
    naive_bayes_spn(weights, leaves)

The standard two-level network: one sum node over `K` components, each a product
of independent univariate leaves. `leaves[k][j]` is the density of variable `j`
in component `k`. Small enough to check exhaustively, general enough to be a real
mixture model.
"""
function naive_bayes_spn(weights::AbstractVector, leaves::Vector{<:Vector{<:UnivariateDensity}})
    length(weights) == length(leaves) ||
        throw(ArgumentError("one set of leaves per mixture component"))
    components = SPNNode[]
    for comp in leaves
        push!(components, product_node([LeafNode(j, comp[j]) for j in eachindex(comp)]))
    end
    return sum_node(components, weights)
end
