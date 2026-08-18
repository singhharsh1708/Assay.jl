# Conjugate reference problems.
#
# Each constructor returns, from one place, the pieces the verification suite
# needs to be self-consistent:
#
#   * `model`       : the [`Model`](@ref) a sampler is run on
#   * `posterior`   : the exact posterior as a `NamedTuple` of densities
#   * `prior`       : the prior, for simulation based calibration and Geweke
#   * `logevidence` : the exact `log p(y)`, which SMC's normalising constant
#                     estimate is checked against
#   * `simulate`    : `θ -> data`, the forward model, for calibration tests
#
# Having the analytic answer and the sampled model come from the same object is
# deliberate: it is impossible for a test to check a sampler against the posterior
# of a different model by accident.

"""
    beta_bernoulli(data; a = 1.0, b = 1.0)

Beta(a, b) prior on a success probability with Bernoulli observations. Posterior
is `Beta(a + Σy, b + n - Σy)` and the evidence is a ratio of beta functions.
"""
function beta_bernoulli(data::AbstractVector; a::Real = 1.0, b::Real = 1.0)
    n = length(data)
    s = sum(data)
    prior_model = Model((p = unit(),), theta -> logpdf(Beta(a, b), theta.p))
    loglik = theta -> loglikelihood(Bernoulli(theta.p), data)
    model = Model((p = unit(),),
                  theta -> logpdf(Beta(a, b), theta.p) + loglikelihood(Bernoulli(theta.p), data))
    return (model = model,
            tempered = TemperedModel(prior_model, loglik;
                                     prior_rand = rng -> (p = rand(rng, Beta(a, b)),)),
            posterior = (p = Beta(a + s, b + n - s),),
            prior = (p = Beta(a, b),),
            logevidence = logbeta(a + s, b + n - s) - logbeta(a, b),
            simulate = (theta, rng) -> [rand(rng, Bernoulli(theta.p)) for _ in 1:n],
            data = data)
end

"""
    normal_normal(data; mu0 = 0.0, tau0 = 10.0, sigma = 1.0)

Normal prior on a normal mean with known observation standard deviation
`sigma`. Posterior precision is `1/tau0^2 + n/sigma^2`.
"""
function normal_normal(data::AbstractVector; mu0::Real = 0.0, tau0::Real = 10.0, sigma::Real = 1.0)
    n = length(data)
    sy = sum(data)
    prec = 1 / tau0^2 + n / sigma^2
    B = mu0 / tau0^2 + sy / sigma^2
    mun = B / prec
    taun = sqrt(1 / prec)
    logZ = -n / 2 * log(2 * pi * sigma^2) - sum(abs2, data) / (2 * sigma^2) -
           mu0^2 / (2 * tau0^2) - 0.5 * log(tau0^2 * prec) + B^2 / (2 * prec)
    prior_model = Model((mu = unconstrained(),), theta -> logpdf(Normal(mu0, tau0), theta.mu))
    loglik = theta -> loglikelihood(Normal(theta.mu, sigma), data)
    model = Model((mu = unconstrained(),),
                  theta -> logpdf(Normal(mu0, tau0), theta.mu) +
                           loglikelihood(Normal(theta.mu, sigma), data))
    return (model = model,
            tempered = TemperedModel(prior_model, loglik;
                                     prior_rand = rng -> (mu = rand(rng, Normal(mu0, tau0)),)),
            posterior = (mu = Normal(mun, taun),),
            prior = (mu = Normal(mu0, tau0),),
            logevidence = logZ,
            simulate = (theta, rng) -> [rand(rng, Normal(theta.mu, sigma)) for _ in 1:n],
            data = data)
end

"""
    gamma_poisson(data; a = 2.0, b = 1.0)

Gamma(shape `a`, rate `b`) prior on a Poisson rate. Posterior is
`Gamma(a + Σy, b + n)`; the evidence is a product of negative binomial terms in
closed form. This is the case that exercises the positive-constraint transform,
so a missing Jacobian shows up here first.
"""
function gamma_poisson(data::AbstractVector; a::Real = 2.0, b::Real = 1.0)
    n = length(data)
    s = sum(data)
    logZ = -sum(x -> loggamma(x + 1), data) + loggamma(a + s) - loggamma(a) +
           a * log(b) - (a + s) * log(b + n)
    prior_model = Model((lambda = positive(),), theta -> logpdf(Gamma(a, b), theta.lambda))
    loglik = theta -> loglikelihood(Poisson(theta.lambda), data)
    model = Model((lambda = positive(),),
                  theta -> logpdf(Gamma(a, b), theta.lambda) +
                           loglikelihood(Poisson(theta.lambda), data))
    return (model = model,
            tempered = TemperedModel(prior_model, loglik;
                                     prior_rand = rng -> (lambda = rand(rng, Gamma(a, b)),)),
            posterior = (lambda = Gamma(a + s, b + n),),
            prior = (lambda = Gamma(a, b),),
            logevidence = logZ,
            simulate = (theta, rng) -> [rand(rng, Poisson(theta.lambda)) for _ in 1:n],
            data = data)
end
