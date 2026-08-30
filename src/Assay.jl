"""
    Assay

Bayesian inference implemented from first principles in Julia: transforms,
model interface, MCMC (random walk Metropolis-Hastings, HMC, NUTS), sequential
Monte Carlo, mean-field ADVI, and the diagnostics needed to tell whether any of
it is right.

No inference library is a dependency. ForwardDiff supplies gradients; everything
else - densities, samplers, effective sample size, R-hat - is in `src/`.

The layering is strict and one-directional:

    transforms.jl  ->  model.jl  ->  samplers/*.jl  ->  diagnostics.jl

A transform knows nothing about models, a model knows nothing about samplers,
and a sampler sees only a log density and a gradient on `R^n`.
"""
module Assay

using LinearAlgebra
using Random
using Printf
using Base.CoreLogging: @logmsg
using ForwardDiff
using DiffResults
using SpecialFunctions: loggamma, logbeta, erf, erfc, beta_inc, gamma_inc

import Statistics
import Statistics: mean, var, std, median, quantile

# Layer 0: errors and numerics
include("errors.jl")
export AssayError, NonDeterministicModelError, NonFiniteDensityError, InitialisationError,
       BackendUnavailableError

include("utils.jl")
export logistic, logit, log1pexp, logsumexp, fft!, ifft!, next_power_of_two

# Layer 1: densities
include("densities.jl")
export Density, UnivariateDensity, MultivariateDensity
export Normal, LogNormal, Cauchy, StudentT, Uniform, Exponential, Gamma, InverseGamma,
       Beta, Bernoulli, Binomial, Poisson, Categorical, Multinomial, MvNormal, MvNormalCholesky, covariance, Dirichlet,
       LKJCholesky, lkj_log_constant
export logpdf, loglikelihood, cdf

include("wrappers.jl")
export MixtureDensity, n_components, Truncated, truncated, Censored, censored,
       ccdf, logcdf, logccdf

# Layer 2: transforms
include("transforms.jl")
export AbstractTransform, udim, cdim, to_constrained, to_unconstrained, flat_names, flatten
export unconstrained, positive, unit, interval, lower, upper, simplex, ordered,
       corr_cholesky, correlation_matrix

# Layer 3: gradients and the model interface
include("ad.jl")
export ADBackend, ForwardDiffAD, ReverseDiffAD, ZygoteAD, EnzymeAD, FiniteDiffAD

include("model.jl")
export AbstractModel, Model, logdensity, logdensity_and_gradient, constrain, unconstrain, dimension,
       parameter_names, flat_dimension, flatten_draw

include("external.jl")
export LogDensityModel

# Layer 4: output container and diagnostics
include("chains.jl")
export Chains, ChainSummary, summarize, ndraws, nparams, nchains, sampler_stat, divergences,
       has_unconstrained, subset,
       acceptance_rate

include("diagnostics.jl")
export ess, ess_bulk, ess_tail, ess_quantile, rhat, rhat_plain, mcse_mean, mcse_std,
       mcse_quantile, bfmi, autocov, autocov_fft,
       split_chains, rank_normalize

# Working with a fitted posterior
include("predictive.jl")
export parameter_draws, n_parameter_draws, predictive, pointwise, pointwise_log_likelihood,
       predictive_check, PredictiveCheck

# Importance sampling diagnostics and cross validation
include("psis.jl")
export psis, PSISResult, reliable, gpd_fit, gpd_quantile, loo, LOOResult, problematic,
       waic, loo_compare

# Layer 5: samplers
include("progress.jl")
export ProgressLevel, ProgressReporter, tick!, report!, finish!, format_duration

include("samplers/interface.jl")
export AbstractSampler, sample, init_state, step!, refresh!, finish_warmup!, random_init,
       check_model, ChainFailure, failures, failed

include("samplers/mh.jl")
export RandomWalkMH, AcceptanceRule, MetropolisRule, BarkerRule, accept_prob

include("samplers/gradient_mh.jl")
export MALA, Barker

include("samplers/hamiltonian.jl")
export AbstractMetric, UnitMetric, DiagMetric, DenseMetric, leapfrog, hamiltonian,
       kinetic, velocity, rand_momentum

include("samplers/hmc.jl")
export HMC

include("samplers/nuts.jl")
export NUTS, UTurnCriterion, ClassicUTurn, GeneralizedUTurn, StrictGeneralizedUTurn,
       tree_continues, merge_continues, uturn_ok

include("samplers/advi.jl")
export ADVI, VIResult, VariationalFamily, MeanField, FullRank, elbo, elbo_with_error,
       posterior_samples,
       variational_mean, variational_scale, variational_factor

include("samplers/smc.jl")
export SMC, SMCResult, TemperedModel, AbstractResampler, MultinomialResampling,
       StratifiedResampling, SystematicResampling, ResidualResampling, resample,
       ess_weights, AdaptiveRandomWalk, weighted_mean, weighted_std, weighted_quantile

# Modes and the Gaussian that matches the curvature at one
include("optimise.jl")
export find_mode, ModeResult, laplace, LaplaceResult

# Reading the diagnostics back as advice
include("diagnose.jl")
export diagnose, Diagnosis, Finding, healthy

# Reference problems with closed-form answers
include("conjugate.jl")
export beta_bernoulli, normal_normal, gamma_poisson

# Sum-product networks
include("spn.jl")
export SPNNode, LeafNode, SumNode, ProductNode, sum_node, product_node, scope,
       n_sum_nodes, naive_bayes_spn

# Uniformity testing for calibration ranks
include("ecdf.jl")
export ecdf_simultaneous_band, ecdf_band_coverage, rank_ecdf, rank_uniformity_ecdf,
       uniformity_ecdf, binomial_quantile

# Leave-one-out predictive calibration
include("loo_pit.jl")
export loo_pit, LOOPITResult, pointwise_cdf

# Calibration checks against the joint distribution
include("calibration.jl")
export CalibrationProblem, conjugate_problem, sbc, SBCResult, rank_histogram,
       rank_uniformity_test, geweke, GewekeResult, calibrated

# Checks on the prior rather than on the posterior
include("prior.jl")
export prior_predictive, prior_predictive_check, power_scale, prior_sensitivity,
       SensitivityResult, sensitive

# A dynamical system on the simplex, as a worked non-conjugate example
include("simplex_dynamics.jl")
export simplex_step, simplex_trajectory, replicator_model, replicator_problem

# Compile the common paths at build time rather than on the user's first call
include("precompile.jl")

end # module
