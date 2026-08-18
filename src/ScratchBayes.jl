"""
    ScratchBayes

Bayesian inference implemented from first principles in Julia: transforms,
model interface, MCMC (random walk Metropolis-Hastings, HMC, NUTS), sequential
Monte Carlo, mean-field ADVI, and the diagnostics needed to tell whether any of
it is right.

No inference library is a dependency. ForwardDiff supplies gradients; everything
else — densities, samplers, effective sample size, R-hat — is in `src/`.

The layering is strict and one-directional:

    transforms.jl  ->  model.jl  ->  samplers/*.jl  ->  diagnostics.jl

A transform knows nothing about models, a model knows nothing about samplers,
and a sampler sees only a log density and a gradient on `R^n`.
"""
module ScratchBayes

using LinearAlgebra
using Random
using Printf
using ForwardDiff
using DiffResults
using SpecialFunctions: loggamma, logbeta, erf, erfc, beta_inc, gamma_inc

import Statistics
import Statistics: mean, var, std, median, quantile

# Layer 0: numerics
include("utils.jl")
export logistic, logit, log1pexp, logsumexp

# Layer 1: densities
include("densities.jl")
export Density, UnivariateDensity, MultivariateDensity
export Normal, LogNormal, Cauchy, StudentT, Uniform, Exponential, Gamma, InverseGamma,
       Beta, Bernoulli, Binomial, Poisson, MvNormal, Dirichlet
export logpdf, loglikelihood, cdf

# Layer 2: transforms
include("transforms.jl")
export AbstractTransform, udim, cdim, to_constrained, to_unconstrained
export unconstrained, positive, unit, interval, lower, upper, simplex, ordered

# Layer 3: gradients and the model interface
include("ad.jl")
export ADBackend, ForwardDiffAD, ReverseDiffAD, FiniteDiffAD

include("model.jl")
export Model, logdensity, logdensity_and_gradient, constrain, unconstrain, dimension,
       parameter_names, flat_dimension, flatten_draw

# Layer 4: output container and diagnostics
include("chains.jl")
export Chains, ChainSummary, summarize, ndraws, nparams, nchains, stat, divergences,
       acceptance_rate

include("diagnostics.jl")
export ess, ess_bulk, ess_tail, rhat, rhat_plain, mcse_mean, bfmi, autocov,
       split_chains, rank_normalize

# Layer 5: samplers
include("samplers/interface.jl")
export AbstractSampler, sample, init_state, step!, random_init

include("samplers/mh.jl")
export RandomWalkMH, AcceptanceRule, MetropolisRule, BarkerRule, accept_prob

# Reference problems with closed-form answers
include("conjugate.jl")
export beta_bernoulli, normal_normal, gamma_poisson

end # module
