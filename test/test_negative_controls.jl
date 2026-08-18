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
struct NoJacobianPositive <: SB.AbstractTransform
    n::Int
end
SB.udim(t::NoJacobianPositive) = t.n
SB.cdim(t::NoJacobianPositive) = t.n
SB.to_constrained(::NoJacobianPositive, y::AbstractVector) = (exp.(y), zero(eltype(y)))
SB.to_unconstrained(::NoJacobianPositive, x::AbstractVector) = log.(float.(x))
broken_positive() = SB.ScalarT(NoJacobianPositive(1))

broken_gamma_poisson(data; a = 2.0, b = 1.0) =
    SB.Model((lambda = broken_positive(),),
             t -> SB.logpdf(SB.Gamma(a, b), t.lambda) +
                  SB.loglikelihood(SB.Poisson(t.lambda), data))

"""An acceptance rule that never rejects, i.e. no Metropolis correction at all."""
struct AlwaysAccept <: SB.AcceptanceRule end
SB.accept_prob(::AlwaysAccept, logratio::Real) = 1.0

"""A gradient backend that returns a multiple of the true gradient."""
struct ScaledGradient <: SB.ADBackend
    factor::Float64
end
function SB.logdensity_and_gradient(b::ScaledGradient, f, y::AbstractVector)
    v, g = SB.logdensity_and_gradient(SB.ForwardDiffAD(), f, y)
    return v, g .* b.factor
end

"""A termination criterion that never fires: every trajectory runs to the maximum depth."""
struct NeverUTurn <: SB.UTurnCriterion end
SB.tree_continues(::NeverUTurn, metric, t) = true

@testset "negative controls" begin
    rng = Random.Xoshiro(11)
    data = [SB.rand(rng, SB.Poisson(3.0)) for _ in 1:40]
    ref = SB.gamma_poisson(data; a = 2.0, b = 1.0)
    S = sum(data)
    exact_mean = SB.mean(ref.posterior.lambda)

    @testset "dropping the Jacobian targets a different, predictable posterior" begin
        chn = SB.sample(broken_gamma_poisson(data), SB.NUTS(), 20_000;
                        n_warmup = 1000, n_chains = 4, rng = Random.Xoshiro(1))
        x = chn[:lambda]
        se = SB.mcse_mean(x)
        # Omitting log|dlambda/dy| = y divides the density by lambda, which is
        # exactly a shift of one in the Gamma shape. The broken sampler is not
        # merely "off": it is a correct sampler for Gamma(a + S - 1, b + n).
        predicted = SB.Gamma(2.0 + S - 1, 1.0 + length(data))
        @test abs(mean(vec(x)) - SB.mean(predicted)) < 4 * se
        @test abs(mean(vec(x)) - exact_mean) > 10 * se
        @test !check_mean(x, exact_mean)
    end

    @testset "simulation based calibration and Geweke both catch it" begin
        good = SB.conjugate_problem(SB.gamma_poisson, 5; a = 2.0, b = 1.0)
        bad = SB.CalibrationProblem(d -> broken_gamma_poisson(d; a = 2.0, b = 1.0),
                                    good.prior_rand, good.simulate)
        res_good = SB.sbc(Random.Xoshiro(3), good, SB.NUTS(); n_sims = 200, n_draws = 64,
                          thin = 6, n_warmup = 400)
        res_bad = SB.sbc(Random.Xoshiro(3), bad, SB.NUTS(); n_sims = 200, n_draws = 64,
                         thin = 6, n_warmup = 400)
        @test res_good.pvalue[1] > 0.01
        @test res_bad.pvalue[1] < 0.01
        # the histogram slopes rather than spiking: the draws are shifted, not overdispersed
        counts = SB.rank_histogram(res_bad, 1)
        @test sum(counts[(end - 1):end]) > sum(counts[1:2])

        g = SB.geweke(Random.Xoshiro(7), bad, SB.NUTS(); n_marginal = 40_000,
                      n_successive = 20_000)
        @test abs(g.z_mean[1]) > 5
    end

    @testset "removing the Metropolis correction destroys the chain" begin
        chn = SB.sample(ref.model,
                        SB.RandomWalkMH(; rule = AlwaysAccept(), adapt_scale = false, scale = 0.3),
                        20_000; n_warmup = 1000, n_chains = 4, rng = Random.Xoshiro(3))
        x = chn[:lambda]
        # An unconstrained random walk with no correction is not ergodic for any
        # distribution: the draws run away, and both R-hat and the mean say so.
        @test SB.rhat(x) > 1.1
        @test mean(vec(x)) > 100 * exact_mean
    end

    @testset "a wrong gradient costs efficiency, not correctness" begin
        good = SB.sample(ref.model, SB.NUTS(), 20_000; n_warmup = 1000, n_chains = 4,
                         rng = Random.Xoshiro(2))
        mild = SB.sample(ref.model, SB.NUTS(; backend = ScaledGradient(1.1)), 20_000;
                         n_warmup = 1000, n_chains = 4, rng = Random.Xoshiro(2))
        wild = SB.sample(ref.model, SB.NUTS(; backend = ScaledGradient(3.0)), 20_000;
                         n_warmup = 1000, n_chains = 4, rng = Random.Xoshiro(2))

        # The leapfrog map is reversible and volume preserving for any force
        # field, and the acceptance step uses the true log density, so a wrong
        # gradient leaves the invariant distribution untouched. A mild error is
        # therefore undetectable by any correctness test.
        @test check_mean(mild[:lambda], exact_mean)
        @test check_std(mild[:lambda], sqrt(SB.var(ref.posterior.lambda)); nse = 5)

        # A large error is caught, but by the efficiency diagnostics: the
        # trajectory cost explodes and the effective sample size collapses.
        eff(c) = SB.ess_bulk(c[:lambda]) / sum(SB.sampler_stat(c, :n_leapfrog))
        @test eff(wild) < eff(good) / 100
        @test mean(SB.sampler_stat(wild, :n_leapfrog)) > 20 * mean(SB.sampler_stat(good, :n_leapfrog))
        @test SB.rhat(wild[:lambda]) > 1.01
    end

    @testset "a termination criterion that never fires stays correct and gets slow" begin
        good = SB.sample(ref.model, SB.NUTS(; max_treedepth = 6), 20_000;
                         n_warmup = 1000, n_chains = 4, rng = Random.Xoshiro(4))
        never = SB.sample(ref.model, SB.NUTS(; uturn = NeverUTurn(), max_treedepth = 6), 20_000;
                          n_warmup = 1000, n_chains = 4, rng = Random.Xoshiro(4))
        # Still valid: the U-turn rule decides when to stop, not what to sample.
        @test check_mean(never[:lambda], exact_mean)
        @test check_std(never[:lambda], sqrt(SB.var(ref.posterior.lambda)); nse = 5)
        eff(c) = SB.ess_bulk(c[:lambda]) / sum(SB.sampler_stat(c, :n_leapfrog))
        @test eff(never) < eff(good) / 5
        @test mean(SB.sampler_stat(never, :n_leapfrog)) ==
              2^6 - 1 || mean(SB.sampler_stat(never, :n_leapfrog)) > 30
    end
end
