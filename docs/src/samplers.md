# Samplers

Every sampler takes a model, a number of draws, and returns [`Chains`](@ref).

```julia
sample(model, sampler, n_draws; n_warmup, n_chains, rng, init, thin, keep_warmup)
```

## Choosing one

| Sampler | Use it when |
|---|---|
| [`NUTS`](@ref) | almost always. Adapts its own trajectory length |
| [`HMC`](@ref) | you want a fixed trajectory length, usually to study one |
| [`RandomWalkMH`](@ref) | no gradient is available, or as a control |
| [`SMC`](@ref) | you need the normalising constant, or the posterior arrives in stages |
| [`ADVI`](@ref) | you need an answer faster than sampling can give one, and can check it |

Measured cost per effective draw on a ten dimensional ill-conditioned normal,
from `bench/benchmarks.jl`: NUTS with a diagonal metric needs about 7 gradient
evaluations per effective draw, static HMC with ten leapfrog steps needs 689,
and a random walk needs none but produces far fewer effective draws per second.

## Gradient-based sampling

```@example samplers
using Assay, Random

data = 1.5 .+ randn(Xoshiro(1), 50)
model = Model((mu = unconstrained(),),
              theta -> logpdf(Normal(0, 5), theta.mu) +
                       loglikelihood(Normal(theta.mu, 1.0), data))

chain = sample(model, NUTS(target_accept = 0.9, metric = :diag), 1000;
               n_warmup = 1000, rng = Xoshiro(2))
summarize(chain)
```

`metric` is `:unit`, `:diag` or `:dense`. A metric is a global linear
reparameterisation: it fixes scale and correlation, and cannot fix curvature
that varies across the posterior. When a target is curved, a smaller step size
or a reparameterisation is what helps, not a richer metric.

`target_accept` drives the dual averaged step size. Raising it shortens the
steps, which is the standard response to divergences, but see
[Checking that it is right](verification.md) for how far that alone gets you.

## Termination criteria

NUTS stops doubling when the trajectory turns back on itself. Three criteria are
available: [`ClassicUTurn`](@ref), [`GeneralizedUTurn`](@ref) and
[`StrictGeneralizedUTurn`](@ref), the last being the default and matching Stan.

The rule has to be a property of the trajectory alone, not of the order it was
grown in. Applying it inconsistently produces a sampler that reports no
divergences, an R-hat of 1.00 and a posterior standard deviation that is quietly
wrong, which is a real bug this package once had and now has a regression test
for.

## Sequential Monte Carlo

SMC needs the prior and the likelihood separately, so it takes a
[`TemperedModel`](@ref).

```@example samplers
prior = Model((mu = unconstrained(),), theta -> logpdf(Normal(0, 5), theta.mu))
tempered = TemperedModel(prior,
                         theta -> loglikelihood(Normal(theta.mu, 1.0), data);
                         prior_rand = rng -> (mu = rand(rng, Normal(0, 5)),))

result = sample(tempered, SMC(n_particles = 2000); rng = Xoshiro(3))
(logZ = result.logZ, mean = weighted_mean(result, :mu))
```

`result.logZ` estimates the log marginal likelihood, which is the quantity no
MCMC sampler here produces and the one SMC is checked against: for conjugate
models it matches the analytic evidence to within 0.03 nats at 4000 particles.

The temperature ladder is chosen adaptively by bisection on effective sample
size, and any sampler in this package can serve as the rejuvenation kernel:
`SMC(kernel = NUTS())`.

## Variational inference

```@example samplers
fit = sample(model, ADVI(family = FullRank()); rng = Xoshiro(4))
draws = posterior_samples(fit, 10_000; rng = Xoshiro(5))
summarize(draws)
```

Mean field cannot represent posterior correlation and will understate variance
in a known, calculable way: for a bivariate normal its optimal marginal standard
deviation is `sqrt(1 - rho^2)`. Full rank can, at quadratic cost in parameters.

The ELBO is a lower bound on the log evidence, so `elbo_with_error(fit)` against
a known evidence is a real check rather than a progress bar.
