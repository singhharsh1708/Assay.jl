# Working with other packages

Three interfaces, all through package extensions, so none of them is a
dependency of this package.

## LogDensityProblems

A log density written for another package can be sampled here without being
rewritten, and a model written here can be sampled by anything in the ecosystem.

```@example interop
using Assay, Random, LogDensityProblems

struct MyProblem end
LogDensityProblems.capabilities(::Type{MyProblem}) = LogDensityProblems.LogDensityOrder{0}()
LogDensityProblems.dimension(::MyProblem) = 2
LogDensityProblems.logdensity(::MyProblem, y) = -0.5 * sum(abs2, y)

chain = sample(LogDensityModel(MyProblem()), NUTS(), 500; n_warmup = 500, rng = Xoshiro(1))
summarize(chain)
```

If the problem advertises first order capability, its own gradient is used
rather than being recomputed, which matters when that gradient is exact or
compiled.

In the other direction, an [`Model`](@ref) satisfies the interface directly,
carrying its Jacobian correction with it:

```@example interop
data = [1, 0, 1, 1, 0]
model = Model((p = unit(),),
              theta -> logpdf(Beta(2, 2), theta.p) +
                       loglikelihood(Bernoulli(theta.p), data))

LogDensityProblems.dimension(model), LogDensityProblems.logdensity(model, [0.3])
```

## Tables

Chains implement the Tables interface, in long format with `chain` and `draw`
kept as columns.

```julia
using DataFrames
DataFrame(chain)
```

That is one interface rather than one integration per package, so CSV writing,
querying and table-shaped plotting all follow.

## MCMCChains

```julia
using MCMCChains
mc = MCMCChains.Chains(chain)
```

Sampler statistics travel across into the `:internals` section, where the
ecosystem looks for divergences, tree depth and step size, so StatsPlots recipes
and ArviZ work as they would with any other chain. Their effective sample size
and R-hat agree with this package's to 10% and 2% on the same draws.

## Gradient backends

```julia
sample(model, NUTS(backend = EnzymeAD()), 1000)
sample(model, NUTS(backend = ReverseDiffAD(compile = true)), 1000)
```

Five backends are available: `ForwardDiffAD` (the default), `ReverseDiffAD`,
`EnzymeAD`, `ZygoteAD` and `FiniteDiffAD`. Measured on a normal target, one
gradient evaluation:

| dimension | ForwardDiff | ReverseDiff | Enzyme |
|---:|---:|---:|---:|
| 2 | 0.5 us | 3.3 us | 1.0 us |
| 10 | 1.1 us | 23.1 us | 1.5 us |
| 50 | 62.7 us | 410.4 us | 5.6 us |
| 200 | 3724 us | 6704 us | 54 us |

Forward mode costs one pass per parameter, so it wins only while that number is
small. Enzyme overtakes it before ten parameters and is nearly seventy times
faster at two hundred. If a model has more than a handful of parameters, that
is the backend to reach for.

Two caveats worth knowing rather than discovering:

- **Zygote cannot differentiate array mutation**, and `simplex`, `ordered` and
  `corr_cholesky` build their output that way. Models using them need another
  backend, and `ZygoteAD` says so explicitly rather than failing with a message
  about `setindex!`.
- **ReverseDiff tape compilation** is valid only for a log density whose control
  flow does not depend on parameter values.
