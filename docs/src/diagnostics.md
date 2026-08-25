# Diagnostics

Everything here is implemented in this package and checked against a closed form
or an independent implementation.

## Convergence and precision

```@example diag
using Assay, Random, Statistics

data = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1]
model = Model((p = unit(),),
              theta -> logpdf(Beta(2, 2), theta.p) +
                       loglikelihood(Bernoulli(theta.p), data))
chain = sample(model, NUTS(), 2000; n_warmup = 1000, n_chains = 4, rng = Xoshiro(1))

x = chain[:p]
(rhat = rhat(x), ess_bulk = ess_bulk(x), ess_tail = ess_tail(x),
 mcse_mean = mcse_mean(x), mcse_median = mcse_quantile(x, 0.5), mcse_std = mcse_std(x))
```

[`rhat`](@ref) is the rank-normalised split version: chains are split in half
before comparison, so a single chain that drifts is caught, and ranks are used
so the diagnostic is defined for heavy-tailed posteriors.

[`ess_bulk`](@ref) and [`ess_tail`](@ref) are separate because a chain can mix
well in the middle of a distribution and badly in its tails. Autocovariances go
through a hand-written FFT and use every lag, checked against the AR(1) closed
form `n(1-r)/(1+r)`, including at `r = -0.5` where an antithetic chain is worth
more than its length. A clamp at `n` would silently discard that, and NUTS
operates in exactly that regime.

Quantile standard errors are computed on the probability scale, where the order
statistic is Beta, rather than by a normal approximation on the value scale that
fails in the tails where the error is largest.

## Hamiltonian diagnostics

```@example diag
(divergences = divergences(chain), bfmi = bfmi(chain),
 mean_treedepth = sum(sampler_stat(chain, :treedepth)) / length(sampler_stat(chain, :treedepth)))
```

A low Bayesian fraction of missing information, below about 0.3, means momentum
resampling is not refreshing the energy fast enough, which usually points at a
badly scaled mass matrix.

## Cross validation and model comparison

```@example diag
loglik = (theta, i) -> logpdf(Bernoulli(theta.p), data[i])
ll = pointwise_log_likelihood(model, chain, loglik; n_obs = length(data))
result = loo(ll)
```

[`loo`](@ref) uses Pareto smoothed importance sampling. The shape `k` reported
per observation is the diagnostic that matters:

| `k` | meaning |
|---|---|
| below 0.5 | the weights have finite variance, estimates converge fast |
| 0.5 to 0.7 | finite mean, infinite variance, usable with more draws |
| 0.7 and above | not to be trusted, whatever the number says |

```@example diag
problematic(result)
```

[`loo_compare`](@ref) ranks models and reports the standard error of each
difference from paired pointwise differences, because two `elpd` estimates on
the same data are strongly correlated and combining their separate standard
errors would overstate the uncertainty.

[`waic`](@ref) is available and is the weaker choice: it has no diagnostic that
says when it has failed.
