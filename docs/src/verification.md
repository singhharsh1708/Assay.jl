# Checking that it is right

The tools in this page are library code, not test scaffolding, so they can be
pointed at your own model. They are also how this package checks itself.

## The ladder

Each rung catches a class of error the one below it cannot.

1. **Unit tests.** Transforms round trip, their Jacobians match automatic
   differentiation, and a density pushed through one still integrates to one.
2. **Closed-form posteriors.** Conjugate models, with tolerances in Monte Carlo
   standard errors rather than as bare numbers.
3. **Simulation based calibration.** Against the whole prior rather than one
   posterior.
4. **The Geweke joint distribution test.** Catches kernels that do not leave the
   posterior invariant.
5. **Hard geometries.** Where samplers actually break.
6. **Negative controls.** Break it deliberately and check the tests notice.
7. **Independent oracles.** Turing.jl and MCMCChains, to catch a misunderstanding
   shared by all of your own tests.

## Simulation based calibration

Draw parameters from the prior, simulate data, sample the posterior, and record
the rank of the true value among the draws. If the sampler is correct those
ranks are uniform.

```@example verify
using Assay, Random

problem = conjugate_problem(gamma_poisson, 20; a = 2.0, b = 1.0)
result = sbc(Xoshiro(1), problem, NUTS(); n_sims = 100, n_draws = 64, thin = 6,
             n_warmup = 400)
```

Two uniformity tests are reported. The binned chi-square is familiar. The second
compares the rank distribution function against a band that holds
simultaneously across the whole curve, which uses the ordering of the ranks that
binning throws away and needs no bin count to be chosen. A pointwise band would
be crossed by a correct sampler most of the time, since it is checked at a
hundred places at once.

```@example verify
calibrated(result)
```

The shape of the departure diagnoses the failure: a slope is a location bias, a
U is over-dispersion, an inverted U is under-dispersion.

For your own model, build a [`CalibrationProblem`](@ref) from three functions:
one that draws from the prior, one that simulates data given parameters, and one
that builds the model given data.

## The Geweke test

Two ways of sampling the joint distribution `p(theta, y)`: drawing from the
prior and then the forward model, or alternating the sampler's transition kernel
with a fresh data set. Both leave the joint invariant, so the marginals of
`theta` must agree.

```@example verify
geweke(Xoshiro(2), problem, NUTS(); n_marginal = 20_000, n_successive = 10_000)
```

This is the sharper instrument for kernel-level errors, non-reversibility in
particular. It caught one here: re-selecting the step size from the current
position at every sweep is not reversible, and the test said so before any
sampler test did.

The number of kernel steps per sweep matters more than it looks. With one step
the successive chain stays autocorrelated, its standard error is optimistic, and
`z` occasionally reaches 7 on a correct sampler.

## Negative controls

A suite that only ever passes proves nothing. Four deliberate bugs, each
installed through a documented extension point rather than by editing the
library:

| What is broken | What happens | Caught by |
|---|---|---|
| the Jacobian term is removed | targets `Gamma(a + S - 1, b + n)` | conjugate check, calibration, Geweke |
| the Metropolis correction is removed | posterior mean 5.8e23 against a true 3.05 | anything |
| the gradient is multiplied by 1.1 | nothing measurable | **nothing** |
| the gradient is multiplied by 3 | 17,470 times fewer effective draws per gradient | R-hat and efficiency |
| the U-turn rule never fires | 10 times slower, still correct | efficiency only |

The third and fifth rows are findings rather than gaps. Leapfrog is reversible
and volume preserving for any force field it integrates, and the acceptance step
uses the true log density, so an error in the gradient or in the stopping rule
cannot change the invariant distribution. It can only waste work. Hunt gradient
bugs with effective sample size per gradient, never with bias.

## What a clean run does not prove

On a curved target, raising the target acceptance rate to 0.95 removes almost
every divergence and still leaves the marginal standard deviation wrong by 6 to
12 standard errors. Measured over three seeds on a banana with a known
covariance:

| configuration | sd of x2 (exact 4.359) | divergences |
|---|---:|---:|
| default | 3.37 to 3.68 | 234 to 656 |
| dense metric | 3.09 to 5.60 | 198 to 1746 |
| target accept 0.95 | 3.97 to 4.12 | 1 to 22 |
| target accept 0.99 | 4.27 to 4.52 | 0 |

A clean divergence count is not evidence of a correct answer.
