# Changelog

## Unreleased

### Added

- `AbstractModel`, so every sampler runs on a plain log density or on any object
  implementing the LogDensityProblems interface, and an Assay model satisfies
  that interface in return.
- Pareto smoothed importance sampling with the shape diagnostic `k`,
  leave-one-out cross validation built on it, WAIC and `loo_compare`.
- A correlation Cholesky transform, the LKJ density with its normalising
  constant, and a Cholesky-parameterised multivariate normal, which together
  make hierarchical models with correlated effects expressible.
- Monte Carlo standard errors for quantiles and for the standard deviation, and
  a hand-written FFT so effective sample size uses every lag.
- Posterior predictive draws and checks, pointwise log likelihoods for cross
  validation, exact reconstruction of parameters from a chain, and chain
  subsetting.
- Calibration by rank distribution function against a simultaneous band, in
  addition to the binned chi-square.
- Tables and MCMCChains interfaces, both as package extensions.
- A documentation site whose examples are executed at build time.

### Changed

- Time to first sample fell from 3.6 seconds to 0.87 in a fresh process, through
  a precompilation workload.
- A log density that is not a function of its argument is now refused with an
  explanation rather than producing an acceptance rate near zero.

### Fixed

- Effective sample size omitted the lag-zero term of Geyer's initial positive
  sequence, so independent draws reported five times too many effective samples.
- The NUTS termination rule was applied inconsistently between the recursion and
  the outer doubling, understating the standard deviation of a correlated
  Gaussian by 2 to 4 percent while every diagnostic read healthy.
- The stick-breaking simplex map could return a final component of `-1e-17`, so
  its output was not quite a point of the simplex.
- `MvNormal` pinned its log determinant to `Float64` and so could not be
  differentiated, which meant an unknown covariance could not be fitted at all.
