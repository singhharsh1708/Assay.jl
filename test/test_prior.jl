@testset "checks on the prior" begin
    rng = Random.Xoshiro(31415)

    @testset "prior predictive draws come from the prior, not the posterior" begin
        prob = AS.conjugate_problem(AS.gamma_poisson, 20; a = 2.0, b = 1.0)
        pp = AS.prior_predictive(prob, 500; rng = Random.Xoshiro(1))
        @test length(pp.data) == 500
        @test length(pp.theta) == 500
        @test all(d -> length(d) == 20, pp.data)
        # the parameters are prior draws: Gamma(2, 1) has mean 2 and variance 2
        lambdas = [t.lambda for t in pp.theta]
        @test Statistics.mean(lambdas) ≈ 2.0 rtol = 0.15
        @test Statistics.var(lambdas) ≈ 2.0 rtol = 0.25
        @test_throws ArgumentError AS.prior_predictive(prob, 0)
    end

    @testset "a sensible prior passes and an absurd one does not" begin
        # Data generated from Poisson(4). Under a prior that believes the rate
        # is around 2, data like this is unremarkable. Under one that believes
        # it is around 500 it is not, and nothing about that requires fitting
        # anything.
        y = [AS.rand(rng, AS.Poisson(4.0)) for _ in 1:20]

        sensible = AS.conjugate_problem(AS.gamma_poisson, 20; a = 2.0, b = 1.0)
        chk = AS.prior_predictive_check(sensible, y, Statistics.mean, 2000;
                                        rng = Random.Xoshiro(2))
        @test 0.05 < chk.pvalue < 0.95
        @test chk.observed ≈ Statistics.mean(y)

        absurd = AS.conjugate_problem(AS.gamma_poisson, 20; a = 500.0, b = 1.0)
        bad = AS.prior_predictive_check(absurd, y, Statistics.mean, 2000;
                                        rng = Random.Xoshiro(3))
        @test bad.pvalue > 0.999
        @test occursin("in the tail", sprint(show, bad))
    end

    @testset "power scaling changes the prior and leaves the likelihood alone" begin
        y = [AS.rand(rng, AS.Poisson(4.0)) for _ in 1:15]
        ref = AS.gamma_poisson(y; a = 2.0, b = 1.0)
        tm = ref.tempered
        theta = (lambda = 3.0,)
        base = AS.at(tm, 1.0).logjoint(theta)
        prior_part = AS.logpdf(AS.Gamma(2.0, 1.0), theta.lambda)
        @test AS.power_scale(tm, 1.0).logjoint(theta) ≈ base
        @test AS.power_scale(tm, 2.0).logjoint(theta) ≈ base + prior_part
        @test AS.power_scale(tm, 0.0).logjoint(theta) ≈ base - prior_part
        # and it is the prior being scaled, not the likelihood, which `at` does
        @test AS.power_scale(tm, 2.0).logjoint(theta) != AS.at(tm, 2.0).logjoint(theta)
    end

    @testset "the data overwhelms a weak prior and does not overwhelm a strong one" begin
        # Raising a Gamma(a, b) prior to the power alpha gives
        # lambda^(alpha(a-1)) exp(-alpha b lambda), so against a Poisson
        # likelihood the posterior is Gamma(alpha(a-1) + 1 + S, alpha b + n).
        # Not Gamma(alpha a + S, alpha b + n): the shape loses the constant that
        # is not part of the kernel, and for a = 1 the shape does not move at
        # all. Both the summaries and the slope follow from that.
        y = [AS.rand(rng, AS.Poisson(4.0)) for _ in 1:200]
        S, n = sum(y), length(y)
        exact_mean(a, a0, b0) = (a * (a0 - 1) + 1 + S) / (a * b0 + n)
        exact_sd(a0, b0) = sqrt(a0 + S) / (b0 + n)
        function exact_slope(a0, b0)
            x = log.([0.8, 1.0, 1.25])
            m = [exact_mean(a, a0, b0) for a in [0.8, 1.0, 1.25]]
            return sum((x .- Statistics.mean(x)) .* (m .- Statistics.mean(m))) /
                   sum((x .- Statistics.mean(x)) .^ 2) / exact_sd(a0, b0)
        end

        function fit(a0, b0, seed)
            ref = AS.gamma_poisson(y; a = a0, b = b0)
            chn = AS.sample(ref.model, AS.NUTS(), 4000; n_warmup = 1000, n_chains = 4,
                            rng = Random.Xoshiro(seed))
            return AS.prior_sensitivity(ref.tempered, chn)
        end
        weak = fit(1.0, 0.01, 4)          # diffuse against 200 observations
        strong = fit(400.0, 100.0, 5)     # worth 100 prior observations

        @test weak.names == [:lambda]
        @test weak.alphas == [0.8, 1.0, 1.25]
        @test weak.base_index == 2
        @test AS.reliable(weak)
        @test AS.reliable(strong)

        # the reweighted summaries against the closed form at every power
        for (k, a) in pairs(weak.alphas)
            @test weak.means[k, 1] ≈ exact_mean(a, 1.0, 0.01) rtol = 0.005
            @test strong.means[k, 1] ≈ exact_mean(a, 400.0, 100.0) rtol = 0.005
            @test weak.stds[k, 1] ≈ exact_sd(1.0, 0.01) rtol = 0.05
        end

        # and the slope itself, which is what the diagnostic reduces to
        @test weak.sensitivity[1] ≈ exact_slope(1.0, 0.01) rtol = 0.1
        @test strong.sensitivity[1] ≈ exact_slope(400.0, 100.0) rtol = 0.1
        @test weak.sensitivity_se[1] > 0
        @test strong.sensitivity_se[1] > 0

        @test abs(weak.sensitivity[1]) < abs(strong.sensitivity[1])
        @test isempty(AS.sensitive(weak))
        @test AS.sensitive(strong) == [:lambda]
        @test occursin("prior matters", sprint(show, strong))
        @test occursin("data dominates", sprint(show, weak))
        @test occursin("std error", sprint(show, weak))
    end

    @testset "a slope smaller than its own error is not a finding" begin
        # Both bars, checked directly. A slope over the threshold whose standard
        # error is the same size has not been measured, and reporting it as a
        # finding is the failure mode of every diagnostic that gives a point
        # estimate on its own.
        big_and_certain = AS.SensitivityResult([:a], [0.8, 1.0, 1.25], zeros(3, 1), ones(3, 1),
                                               [0.4], [0.02], [0.1], 2)
        big_and_noisy = AS.SensitivityResult([:a], [0.8, 1.0, 1.25], zeros(3, 1), ones(3, 1),
                                             [0.4], [0.3], [0.1], 2)
        small = AS.SensitivityResult([:a], [0.8, 1.0, 1.25], zeros(3, 1), ones(3, 1),
                                     [0.01], [0.001], [0.1], 2)
        @test AS.sensitive(big_and_certain) == [:a]
        @test isempty(AS.sensitive(big_and_noisy))
        @test isempty(AS.sensitive(small))
        @test occursin("not measurable", sprint(show, big_and_noisy))
        @test occursin("data dominates", sprint(show, small))
        @test !AS.reliable(AS.SensitivityResult([:a], [1.0], zeros(1, 1), ones(1, 1),
                                                [0.0], [0.0], [0.9], 1))
    end

    @testset "reweighting agrees with refitting, and costs one fit instead of three" begin
        # The check that the shortcut is a shortcut rather than a different
        # answer. Refitting at alpha = 1.25 targets the same posterior the
        # weights are meant to represent.
        y = [AS.rand(rng, AS.Poisson(4.0)) for _ in 1:50]
        ref = AS.gamma_poisson(y; a = 3.0, b = 2.0)
        chn = AS.sample(ref.model, AS.NUTS(), 4000; n_warmup = 1000, n_chains = 4,
                        rng = Random.Xoshiro(6))
        r = AS.prior_sensitivity(ref.tempered, chn; alphas = [1.0, 1.25])

        refit = AS.sample(AS.power_scale(ref.tempered, 1.25), AS.NUTS(), 4000;
                          n_warmup = 1000, n_chains = 4, rng = Random.Xoshiro(7))
        @test r.means[2, 1] ≈ Statistics.mean(vec(refit[:lambda])) rtol = 0.01
        @test r.stds[2, 1] ≈ Statistics.std(vec(refit[:lambda])) rtol = 0.05
        @test AS.reliable(r)
    end

    @testset "input validation" begin
        y = [AS.rand(rng, AS.Poisson(4.0)) for _ in 1:10]
        tm = AS.gamma_poisson(y; a = 2.0, b = 1.0).tempered
        chn = AS.sample(AS.at(tm, 1.0), AS.NUTS(), 200; n_warmup = 200,
                        rng = Random.Xoshiro(8))
        @test_throws DomainError AS.prior_sensitivity(tm, chn; alphas = [-1.0, 1.0])
        @test_throws ArgumentError AS.prior_sensitivity(tm, chn; alphas = [0.5, 2.0])
    end
end
