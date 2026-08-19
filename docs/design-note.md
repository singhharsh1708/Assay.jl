# Design note: the model interface and the transform layer

This note explains why the two pieces that are not algorithms are shaped the way
they are, because those are the probabilistic programming design decisions and
everything else in the repository follows from them.

## The one-sentence version

A model is an ordered set of named parameters, each carrying a transform, plus a
function of the *constrained* parameters returning the log joint density. The
library owns the map to `R^n` and the log absolute Jacobian determinant. A
sampler sees a log density and a gradient on `R^n` and nothing else.

```julia
model = Model((mu = unconstrained(), sigma = positive(), w = simplex(3)),
              theta -> logpdf(Normal(0, 5), theta.mu) +
                       logpdf(Gamma(2, 1), theta.sigma) +
                       logpdf(Dirichlet([2.0, 2.0, 2.0]), theta.w) +
                       loglikelihood(Normal(theta.mu, theta.sigma), data))
```

`dimension(model)` is 4: one for `mu`, one for `sigma`, two for the
three-component simplex. `logdensity(model, y)` for `y in R^4` returns the log
joint plus the Jacobian correction, and `logdensity_and_gradient` differentiates
through the whole thing.

## Why the user writes the density in constrained space

The alternative is to make the user write the density in the unconstrained space
directly: `sigma = exp(y[2])`, and remember the `+ y[2]`. That is the interface
most people first build, and it is wrong for three reasons.

It makes the user responsible for a term whose only purpose is to keep the
measure right, which is exactly the term they will forget. The consequence is
not a crash. Section 5 of [results.md](results.md) removes the Jacobian from the
positive transform and gets a sampler that runs cleanly, reports no divergences,
R-hat 1.00, a healthy acceptance rate, and a posterior that is wrong: for a
Gamma-Poisson model it is a correct sampler for `Gamma(a + S - 1, b + n)` rather
than `Gamma(a + S, b + n)`. That is a plausible answer, off by one unit of prior
shape, and nothing but a calibration test finds it.

It also puts the constraint in two places. If the user writes `exp` in the
density, the library still has to know the parameter is positive in order to
report draws on the right scale, to initialise, and to invert. Two sources of
truth for a constraint is one too many.

Finally it breaks composition. Sequential Monte Carlo needs the prior and the
likelihood separately; variational inference needs the same density with a
different parameterisation of the sampler; simulation based calibration needs to
map a prior draw *into* the unconstrained space to compute a rank. All three are
one line each when the library owns the map, and each is a special case when the
user owns it.

## Why transforms are types rather than closures

A transform has to answer more questions than "what is `x` given `y`":

* how many unconstrained coordinates does it consume, and how many constrained
  scalars does it produce - these differ for the simplex, `K - 1` against `K`;
* what is the inverse, needed to place a chain at a known point and to compute
  simulation based calibration ranks;
* what is the log absolute Jacobian determinant;
* what are the column names in the output, `w[1]`, `w[2]`, `w[3]`.

A pair of closures answers the first question and none of the others. A type
answers all four, and it is testable in isolation: `test/test_transforms.jl`
checks every transform's round trip, checks its analytic Jacobian against the
determinant of the automatic differentiation Jacobian, and integrates a density
pushed through it to confirm it still integrates to one. That last test is the
one that fails if the Jacobian is wrong, and it needs no sampler to run.

The interface is deliberately four functions:

```julia
udim(t)                  # unconstrained dimension
cdim(t)                  # constrained dimension
to_constrained(t, y)     # -> (x, log|det dx/dy|)
to_unconstrained(t, x)   # -> y
```

Returning the value and the Jacobian term together, rather than as two calls, is
not a micro-optimisation. The stick-breaking simplex map computes the running
remainder `1 - sum(x_1..x_{k-1})` on the way through; the Jacobian term needs
exactly that quantity. Splitting the interface would either compute it twice or
force a cache.

## Why a `NamedTuple` of transforms

Parameter order has to be fixed, because it defines the layout of the
unconstrained vector, and parameter names have to survive into the output. A
`NamedTuple` gives both, and it gives the log joint a readable argument:
`theta.sigma`, not `theta[2]`. The transforms are stored as a plain tuple in the
model's type parameter, so unpacking is a recursion the compiler unrolls; there
is no dynamic dispatch or allocation per parameter in the inner loop.

Scalars are a real ergonomic question and not a cosmetic one. `positive()` gives
a scalar parameter, `positive(3)` a vector of three. Without the distinction
every scalar in every user model becomes `theta.sigma[1]`, and that indexing
noise is where people introduce errors. It is implemented as a one-line wrapper
type, `ScalarT`, rather than by special-casing dimension one throughout.

## Why the sampler contract is this small

```julia
init_state(rng, model, sampler, y0; n_warmup)   -> state
step!(rng, model, sampler, state, warmup)       -> (y, stats)
finish_warmup!(rng, model, sampler, state)      -> state
refresh!(model, sampler, state)                 -> state
```

Everything else - chain allocation, initialisation with rejection of non-finite
starting points, warmup bookkeeping, thinning, threading across chains, mapping
draws back to constrained space, timing - lives once in the driver. The result
is that random walk Metropolis, HMC and NUTS are three files that share no code
paths through conditionals, and adding a sampler is adding a file.

`refresh!` deserves a note because it exists for one caller. The Geweke test
alternates the sampler's kernel with fresh data sets, so the same state has to
be re-pointed at a new model. The first version instead re-initialised the
sampler at each sweep, which re-selected the step size from the current
position - and a kernel whose step size depends on the current state is not
reversible. The test detected it as a failure of the sampler, which is exactly
what a good test should do and exactly the wrong conclusion to draw; the fix was
to hold the kernel fixed across the sweep.

## Why tempering produces a model rather than living inside one

Sequential Monte Carlo needs `p(theta) * p(y | theta)^beta`. The obvious design
is a temperature field on `Model`, checked in `logdensity`. Instead:

```julia
struct TemperedModel; prior::Model; loglik; prior_rand; end
at(tm, beta)::Model
```

`at` returns an ordinary `Model`. Everything that works on a model - gradients,
transforms, the whole sampler set - therefore works as a rejuvenation kernel at
any temperature with no changes at all, which is why `SMC(kernel = NUTS())` is
supported by two methods of one function rather than by an integration layer.
Keeping `Model` unaware of temperature also keeps the common case, a single
fixed target, free of a branch it never takes.

## What this is not

It is not a probabilistic programming language. There is no `~` notation, no
program trace, no graph, so there is no automatic marginalisation of discrete
parameters, no conditioning on an arbitrary subset of variables after the fact,
no plate notation, and no generated quantities block. A user writes the log
joint themselves.

That boundary is deliberate. The parts of a probabilistic programming language
that are genuinely hard - and the parts that are actually asked about in a
technical interview - are the measure-theoretic bookkeeping in the transform
layer and the interface between a model and an inference algorithm. A macro that
rewrites `x ~ Normal(0, 1)` into an accumulation of log density terms is
mechanical by comparison, and adding one here would have bought surface area
rather than substance. The natural next step, if this were to grow, is a
`@model` macro that produces exactly the `Model` object above, which is the
reason the constructor takes plain data rather than something a macro would have
to be built around.

## Alternatives considered

**Sampling in constrained space with rejection.** Propose, reject anything
outside the support. Correct for random walk Metropolis, useless for anything
gradient-based, and the acceptance rate collapses as the posterior approaches a
boundary. Rejected: it makes the constraint the sampler's problem, which is the
coupling this design exists to avoid.

**Per-sampler constraint handling.** Each sampler knows about positivity and
handles it. Rejected on sight: it is the design where a new constraint means
editing every sampler.

**Depending on Bijectors.jl and Distributions.jl.** Both are good, and using
them would have removed the transform layer and the densities. It would also
have removed the part of the repository that is worth reading, and the part that
the verification suite is pointed at. The test suite does depend on
Distributions.jl, as an oracle for the hand-written log densities, and on
Turing.jl as an independent posterior oracle - which is the right place for
both.

**A `logdensity` that returns the constrained parameters as well.** Convenient
for sampler stats, and it would have coupled the sampler to the model's
parameter structure. The driver calls `flatten_draw` once per stored draw
instead.
