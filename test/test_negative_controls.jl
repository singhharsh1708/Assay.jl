# Deliberately broken samplers. A suite that only ever passes proves nothing:
# each of these installs a specific, realistic bug through a documented
# extension point - a transform, an acceptance rule, a gradient backend, a
# termination criterion - and asserts what the verification suite does about it.
#
# Two of the four are caught loudly. The other two are not caught at all by any
# correctness test, and that is the finding, not a gap: an error in the gradient
# or in the U-turn criterion cannot bias a sampler whose acceptance step uses
# the true log density, because the leapfrog map stays reversible and volume
# preserving whatever force it integrates. Those two show up in the efficiency
# numbers instead, which is where the suite looks for them.

"""
A positive-support transform with the log Jacobian determinant deliberately set
to zero: the single most common bug in hand-written probabilistic programming
languages.
"""
struct NoJacobianPositive <: AS.AbstractTransform
    n::Int
end
AS.udim(t::NoJacobianPositive) = t.n
AS.cdim(t::NoJacobianPositive) = t.n
AS.to_constrained(::NoJacobianPositive, y::AbstractVector) = (exp.(y), zero(eltype(y)))
AS.to_unconstrained(::NoJacobianPositive, x::AbstractVector) = log.(float.(x))
broken_positive() = AS.ScalarT(NoJacobianPositive(1))

broken_gamma_poisson(data; a = 2.0, b = 1.0) =
    AS.Model((lambda = broken_positive(),),
             t -> AS.logpdf(AS.Gamma(a, b), t.lambda) +
                  AS.loglikelihood(AS.Poisson(t.lambda), data))

"""An acceptance rule that never rejects, i.e. no Metropolis correction at all."""
struct AlwaysAccept <: AS.AcceptanceRule end
AS.accept_prob(::AlwaysAccept, logratio::Real) = 1.0

"""A gradient backend that returns a multiple of the true gradient."""
struct ScaledGradient <: AS.ADBackend
    factor::Float64
end
function AS.logdensity_and_gradient(b::ScaledGradient, f, y::AbstractVector)
    v, g = AS.logdensity_and_gradient(AS.ForwardDiffAD(), f, y)
    return v, g .* b.factor
end

"""A termination criterion that never fires: every trajectory runs to the maximum depth."""
struct NeverUTurn <: AS.UTurnCriterion end
AS.tree_continues(::NeverUTurn, metric, t) = true

@testset "negative controls" begin
    rng = Random.Xoshiro(11)
    data = [AS.rand(rng, AS.Poisson(3.0)) for _ in 1:40]
    ref = AS.gamma_poisson(data; a = 2.0, b = 1.0)
    S = sum(data)
    exact_mean = AS.mean(ref.posterior.lambda)

    @testset "dropping the Jacobian targets a different, predictable posterior" begin
        chn = AS.sample(broken_gamma_poisson(data), AS.NUTS(), 20_000;
                        n_warmup = 1000, n_chains = 4, rng = Random.Xoshiro(1))
        x = chn[:lambda]
        se = AS.mcse_mean(x)
        # Omitting log|dlambda/dy| = y divides the density by lambda, which is
        # exactly a shift of one in the Gamma shape. The broken sampler is not
        # merely "off": it is a correct sampler for Gamma(a + S - 1, b + n).
        predicted = AS.Gamma(2.0 + S - 1, 1.0 + length(data))
        @test abs(mean(vec(x)) - AS.mean(predicted)) < 4 * se
        @test abs(mean(vec(x)) - exact_mean) > 10 * se
        @test !check_mean(x, exact_mean)
    end

    @testset "simulation based calibration and Geweke both catch it" begin
        good = AS.conjugate_problem(AS.gamma_poisson, 5; a = 2.0, b = 1.0)
        bad = AS.CalibrationProblem(d -> broken_gamma_poisson(d; a = 2.0, b = 1.0),
                                    good.prior_rand, good.simulate)
        res_good = AS.sbc(Random.Xoshiro(3), good, AS.NUTS(); n_sims = 200, n_draws = 64,
                          thin = 6, n_warmup = 400)
        res_bad = AS.sbc(Random.Xoshiro(3), bad, AS.NUTS(); n_sims = 200, n_draws = 64,
                         thin = 6, n_warmup = 400)
        @test res_good.pvalue[1] > 0.01
        @test res_bad.pvalue[1] < 0.01
        # the histogram slopes rather than spiking: the draws are shifted, not overdispersed
        counts = AS.rank_histogram(res_bad, 1)
        @test sum(counts[(end - 1):end]) > sum(counts[1:2])

        g = AS.geweke(Random.Xoshiro(7), bad, AS.NUTS(); n_marginal = 40_000,
                      n_successive = 20_000)
        @test abs(g.z_mean[1]) > 5
    end

    @testset "removing the Metropolis correction destroys the chain" begin
        chn = AS.sample(ref.model,
                        AS.RandomWalkMH(; rule = AlwaysAccept(), adapt_scale = false, scale = 0.3),
                        20_000; n_warmup = 1000, n_chains = 4, rng = Random.Xoshiro(3))
        x = chn[:lambda]
        # An unconstrained random walk with no correction is not ergodic for any
        # distribution: the draws run away, and both R-hat and the mean say so.
        @test AS.rhat(x) > 1.1
        @test mean(vec(x)) > 100 * exact_mean
    end

    @testset "a wrong gradient costs efficiency, not correctness" begin
        good = AS.sample(ref.model, AS.NUTS(), 20_000; n_warmup = 1000, n_chains = 4,
                         rng = Random.Xoshiro(2))
        mild = AS.sample(ref.model, AS.NUTS(; backend = ScaledGradient(1.1)), 20_000;
                         n_warmup = 1000, n_chains = 4, rng = Random.Xoshiro(2))
        wild = AS.sample(ref.model, AS.NUTS(; backend = ScaledGradient(3.0)), 20_000;
                         n_warmup = 1000, n_chains = 4, rng = Random.Xoshiro(2))

        # The leapfrog map is reversible and volume preserving for any force
        # field, and the acceptance step uses the true log density, so a wrong
        # gradient leaves the invariant distribution untouched. A mild error is
        # therefore undetectable by any correctness test.
        @test check_mean(mild[:lambda], exact_mean)
        @test check_std(mild[:lambda], sqrt(AS.var(ref.posterior.lambda)); nse = 5)

        # A large error is caught, but by the efficiency diagnostics: the
        # trajectory cost explodes and the effective sample size collapses.
        eff(c) = AS.ess_bulk(c[:lambda]) / sum(AS.sampler_stat(c, :n_leapfrog))
        @test eff(wild) < eff(good) / 100
        @test mean(AS.sampler_stat(wild, :n_leapfrog)) > 20 * mean(AS.sampler_stat(good, :n_leapfrog))
        @test AS.rhat(wild[:lambda]) > 1.01
    end

    @testset "a termination criterion that never fires stays correct and gets slow" begin
        good = AS.sample(ref.model, AS.NUTS(; max_treedepth = 6), 20_000;
                         n_warmup = 1000, n_chains = 4, rng = Random.Xoshiro(4))
        never = AS.sample(ref.model, AS.NUTS(; uturn = NeverUTurn(), max_treedepth = 6), 20_000;
                          n_warmup = 1000, n_chains = 4, rng = Random.Xoshiro(4))
        # Still valid: the U-turn rule decides when to stop, not what to sample.
        @test check_mean(never[:lambda], exact_mean)
        @test check_std(never[:lambda], sqrt(AS.var(ref.posterior.lambda)); nse = 5)
        eff(c) = AS.ess_bulk(c[:lambda]) / sum(AS.sampler_stat(c, :n_leapfrog))
        @test eff(never) < eff(good) / 5
        @test mean(AS.sampler_stat(never, :n_leapfrog)) ==
              2^6 - 1 || mean(AS.sampler_stat(never, :n_leapfrog)) > 30
    end
end
