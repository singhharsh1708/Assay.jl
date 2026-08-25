# Declaring a model

## The shape of a model

```julia
Model(params::NamedTuple, logjoint)
```

`params` maps names to transforms, in the order they are packed into the
unconstrained vector. `logjoint` takes a `NamedTuple` of constrained values and
returns the log joint density.

```@example models
using Assay, Random, Statistics

data = randn(Xoshiro(1), 20)

model = Model((mu = unconstrained(), sigma = positive(), w = simplex(3)),
              theta -> logpdf(Normal(0, 5), theta.mu) +
                       logpdf(Gamma(2, 1), theta.sigma) +
                       logpdf(Dirichlet([2.0, 2.0, 2.0]), theta.w) +
                       loglikelihood(Normal(theta.mu, theta.sigma), data))

dimension(model), flat_dimension(model), parameter_names(model)
```

The dimensions differ because a three component simplex has two free
coordinates and three reported ones.

## The transforms

| Constructor | Support | Consumes |
|---|---|---|
| `unconstrained()`, `unconstrained(n)` | all of `R` | 1, `n` |
| `positive()`, `positive(n)` | `(0, Inf)` | 1, `n` |
| `unit()`, `unit(n)` | `(0, 1)` | 1, `n` |
| `interval(a, b)` | `(a, b)` | 1 |
| `lower(a)`, `upper(b)` | `(a, Inf)`, `(-Inf, b)` | 1 |
| `simplex(K)` | the `K` simplex | `K - 1` |
| `ordered(n)` | strictly increasing vectors | `n` |
| `corr_cholesky(K)` | correlation Cholesky factors | `K(K-1)/2` |

## Why the library owns the Jacobian

A sampler works on `R^n`. Moving a constrained parameter there is a change of
variables, and a change of variables carries a Jacobian term. Leaving that term
out does not crash and does not look wrong:

```@example models
counts = [3, 5, 2, 4, 6, 1, 4, 3]

correct = Model((lambda = positive(),),
                theta -> logpdf(Gamma(2, 1), theta.lambda) +
                         loglikelihood(Poisson(theta.lambda), counts))

# the same model written by hand in the unconstrained coordinate, with the
# change of variables term left out
without = Model((logl = unconstrained(),),
                theta -> logpdf(Gamma(2, 1), exp(theta.logl)) +
                         loglikelihood(Poisson(exp(theta.logl)), counts))

a = sample(correct, NUTS(), 4000; n_warmup = 1000, n_chains = 4, rng = Xoshiro(1))
b = sample(without, NUTS(), 4000; n_warmup = 1000, n_chains = 4, rng = Xoshiro(1))

(with_jacobian = mean(vec(a[:lambda])), without_jacobian = mean(exp.(vec(b[:logl]))))
```

Both runs are healthy: no divergences, R-hat at 1.00, a sensible acceptance
rate. One of them is sampling the wrong posterior. Omitting `log|dlambda/dy| = y`
divides the density by `lambda`, which is exactly a shift of one in the Gamma
shape, so the second is a correct sampler for `Gamma(a + S - 1, b + n)` rather
than `Gamma(a + S, b + n)`.

That is the argument for the transform layer. Nothing but a calibration test
finds this, and the user should not have to be the one who remembers.

## Correlation matrices

A correlation matrix cannot be parameterised elementwise, because positive
definiteness is not a box constraint. `corr_cholesky` goes through canonical
partial correlations instead, and hands the model a Cholesky factor.

```@example models
using LinearAlgebra

K = 3
prior_only = Model((R = corr_cholesky(K), sigma = positive(K)),
                   theta -> logpdf(LKJCholesky(K, 2.0), theta.R) +
                            sum(logpdf(Gamma(2, 1), s) for s in theta.sigma))

y = zeros(dimension(prior_only))
theta, logjac = constrain(prior_only, y)
correlation_matrix(theta.R)
```

Pair it with [`MvNormalCholesky`](@ref), which takes the covariance factor
directly: `Diagonal(sigma) * R` is the factor for scales `sigma` and correlation
factor `R`, so nothing has to be factorised again on every evaluation.

Draws are reported as the correlations themselves, `R[2,1]` and so on, rather
than as entries of the factor, because `L[3,2]` means nothing to a reader.

## Models written elsewhere

Any log density on `R^n` can be sampled without being rewritten.

```@example models
external = LogDensityModel(y -> -0.5 * sum(abs2, y), 3)
sample(external, NUTS(), 500; n_warmup = 500, rng = Xoshiro(2)) |> summarize
```

There is no transform layer in that case and there cannot be: a plain function
carries no declaration of which coordinates are constrained. If a parameter is
bounded, either handle it inside the function, Jacobian included, or use
`Model`.
