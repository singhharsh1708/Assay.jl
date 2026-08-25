# Getting started

A first model, from writing it down to deciding whether to believe the answer.

## Writing the model

A model is two things: named parameters with their constraints, and a function
returning the log joint density in the space the mathematics is written in.

```@example start
using Assay, Random, Statistics

rng = Xoshiro(1)
data = 2.0 .+ 0.8 .* randn(rng, 60)

model = Model((mu = unconstrained(), sigma = positive()),
              theta -> logpdf(Normal(0, 5), theta.mu) +
                       logpdf(Gamma(2, 1), theta.sigma) +
                       loglikelihood(Normal(theta.mu, theta.sigma), data))
```

`sigma` is declared positive rather than being made positive inside the density.
That declaration is what lets the library map it to the unconstrained space the
sampler works in, and add the log absolute Jacobian determinant that keeps the
target distribution correct. Doing it by hand is the single most common way to
get a plausible wrong answer; see [Declaring a model](models.md).

## Sampling

```@example start
chain = sample(model, NUTS(), 2000; n_warmup = 1000, n_chains = 4, rng = Xoshiro(2))
summarize(chain)
```

Four chains run in parallel across threads. `n_warmup` draws are used to adapt
the step size and the mass matrix and then discarded.

## Deciding whether to believe it

Read the summary right to left. `rhat` near 1 means the chains agree with each
other and with themselves; anything above about 1.01 means keep sampling.
`ess_bulk` is how many independent draws the chain is worth for a mean, and
`ess_tail` the same for an extreme quantile. `mcse` is the standard error of the
posterior mean: if it is not small compared to the posterior standard deviation,
the answer is noise.

```@example start
divergences(chain), acceptance_rate(chain)
```

A divergence is a trajectory whose energy error blew up, which means the
sampler could not follow the geometry there. Their presence is strong evidence
of bias. Their absence is weak evidence of anything, which is worth being clear
about: on a curved target this package can produce a run with almost no
divergences whose marginal standard deviation is still wrong by ten standard
errors.

## Using the draws

Posterior draws come back as the `NamedTuple` the model was written against.

```@example start
first(parameter_draws(model, chain))
```

From there, replicated data sets and a check on whether they look like the data
you actually saw:

```@example start
yrep = predictive(model, chain,
                  (theta, r) -> [rand(r, Normal(theta.mu, theta.sigma)) for _ in 1:length(data)];
                  rng = Xoshiro(3), thin = 20)

predictive_check(data, yrep, maximum)
```

A p value near 0 or 1 says the model cannot reproduce that feature of the data.
The statistic is the whole question: a model that gets the mean right and the
tails wrong looks perfect under `mean` and fails under `maximum`.

## Comparing models

```@example start
loglik = (theta, i) -> logpdf(Normal(theta.mu, theta.sigma), data[i])
ll = pointwise_log_likelihood(model, chain, loglik; n_obs = length(data), thin = 4)
loo(ll)
```

`loo` reports the expected log predictive density under leave-one-out cross
validation, and a Pareto shape per observation. Any observation with `k` above
0.7 is one the approximation could not handle, and its contribution should not
be believed. That diagnostic is the reason to prefer this over an information
criterion that cannot tell you when it has failed.
