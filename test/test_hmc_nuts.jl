@testset "gradient-based samplers" begin
    rng = Random.Xoshiro(777)
    data_b = [rand(rng) < 0.35 ? 1 : 0 for _ in 1:100]
    data_n = [1.5 .+ 0.8 * randn(rng) for _ in 1:50]
    data_p = [SB.rand(rng, SB.Poisson(4.0)) for _ in 1:60]

    @testset "$(nameof(typeof(spl))) on conjugate models" for spl in (SB.HMC(), SB.NUTS())
        b = SB.beta_bernoulli(data_b; a = 2.0, b = 2.0)
        chn = SB.sample(b.model, spl, 4000; n_warmup = 1000, n_chains = 4, rng = Random.Xoshiro(11))
        check_conjugate(chn, :p, b.posterior.p)

        nn = SB.normal_normal(data_n; mu0 = 0.0, tau0 = 5.0, sigma = 0.8)
        chn = SB.sample(nn.model, spl, 4000; n_warmup = 1000, n_chains = 4, rng = Random.Xoshiro(12))
        check_conjugate(chn, :mu, nn.posterior.mu)

        gp = SB.gamma_poisson(data_p; a = 2.0, b = 1.0)
        chn = SB.sample(gp.model, spl, 4000; n_warmup = 1000, n_chains = 4, rng = Random.Xoshiro(13))
        check_conjugate(chn, :lambda, gp.posterior.lambda)
    end

    @testset "step size adaptation tracks the target acceptance rate" begin
        # The achieved rate sits slightly *above* the target, and the gap grows
        # when the metric is being adapted underneath the step size: the
        # sampling phase uses the dual-averaged log step size, and the
        # acceptance statistic is convex and decreasing in the step size, so
        # averaging in log space biases it upwards. The same effect is visible
        # in Stan. What must hold is that the target is tracked monotonically
        # and never undershot.
        gp = SB.gamma_poisson(data_p)
        achieved = Float64[]
        for target in (0.65, 0.8, 0.95)
            chn = SB.sample(gp.model, SB.NUTS(; target_accept = target), 2000;
                            n_warmup = 1000, n_chains = 2, rng = Random.Xoshiro(14))
            push!(achieved, SB.acceptance_rate(chn))
            @test target - 0.03 <= achieved[end] <= target + 0.15
        end
        @test issorted(achieved)
        # With the metric fixed there is nothing to restart the dual averaging,
        # and the target is hit tightly.
        chn = SB.sample(gp.model, SB.NUTS(; target_accept = 0.7, adapt_metric = false), 2000;
                        n_warmup = 1000, n_chains = 2, rng = Random.Xoshiro(21))
        @test abs(SB.acceptance_rate(chn) - 0.7) < 0.05
    end

    @testset "metrics on an ill-conditioned Gaussian" begin
        d = 8
        sds = 10 .^ range(-1, 1, length = d)
        target = SB.MvNormal(zeros(d), Matrix(Diagonal(sds .^ 2)))
        model = SB.Model((x = SB.unconstrained(d),), theta -> SB.logpdf(target, theta.x))
        for metric in (:unit, :diag, :dense)
            chn = SB.sample(model, SB.NUTS(; metric = metric), 2000;
                            n_warmup = 1000, n_chains = 4, rng = Random.Xoshiro(15))
            for i in 1:d
                x = chn[Symbol("x[$i]")]
                @test check_mean(x, 0.0; nse = 5)
                @test check_std(x, sds[i]; nse = 5)
            end
            @test SB.divergences(chn) == 0
        end
    end

    @testset "dense metric recovers correlation" begin
        Sigma = [1.0 0.95; 0.95 1.0]
        target = SB.MvNormal([0.0, 0.0], Sigma)
        model = SB.Model((x = SB.unconstrained(2),), theta -> SB.logpdf(target, theta.x))
        chn = SB.sample(model, SB.NUTS(; metric = :dense), 3000;
                        n_warmup = 1000, n_chains = 4, rng = Random.Xoshiro(16))
        @test cor(vec(chn[Symbol("x[1]")]), vec(chn[Symbol("x[2]")])) ≈ 0.95 atol = 0.01
    end

    @testset "the U-turn criteria agree on an easy target" begin
        nn = SB.normal_normal(data_n; sigma = 0.8)
        for crit in (SB.ClassicUTurn(), SB.GeneralizedUTurn())
            chn = SB.sample(nn.model, SB.NUTS(; uturn = crit), 3000;
                            n_warmup = 1000, n_chains = 4, rng = Random.Xoshiro(17))
            check_conjugate(chn, :mu, nn.posterior.mu)
        end
    end

    @testset "divergences are reported on the funnel" begin
        funnel = SB.Model((v = SB.unconstrained(), x = SB.unconstrained(5)),
                          theta -> SB.logpdf(SB.Normal(0.0, 3.0), theta.v) +
                                   sum(SB.logpdf(SB.Normal(0.0, exp(theta.v / 2)), xi) for xi in theta.x))
        chn = SB.sample(funnel, SB.NUTS(; target_accept = 0.8), 2000;
                        n_warmup = 1000, n_chains = 4, rng = Random.Xoshiro(18))
        @test SB.divergences(chn) > 0
        @test all(isfinite, SB.bfmi(chn))
        # the sampler under-covers the neck: this is the documented failure, and
        # a run that reported no divergences here would be the real problem
        @test std(vec(chn[:v])) < 3.0
    end

    @testset "finite differences and forward mode give the same posterior" begin
        nn = SB.normal_normal(data_n; sigma = 0.8)
        a = SB.sample(nn.model, SB.NUTS(; backend = SB.ForwardDiffAD()), 1000;
                      n_warmup = 500, n_chains = 2, rng = Random.Xoshiro(19))
        b = SB.sample(nn.model, SB.NUTS(; backend = SB.FiniteDiffAD()), 1000;
                      n_warmup = 500, n_chains = 2, rng = Random.Xoshiro(19))
        @test abs(mean(vec(a[:mu])) - mean(vec(b[:mu]))) < 4 * SB.mcse_mean(a[:mu])
    end

    @testset "reported statistics are present and sane" begin
        gp = SB.gamma_poisson(data_p)
        chn = SB.sample(gp.model, SB.NUTS(), 1000; n_warmup = 500, n_chains = 2,
                        rng = Random.Xoshiro(20))
        for k in (:accept_prob, :divergent, :treedepth, :n_leapfrog, :step_size, :energy)
            @test haskey(chn.stats, k)
        end
        @test all(0 .<= SB.sampler_stat(chn, :accept_prob) .<= 1)
        @test all(SB.sampler_stat(chn, :n_leapfrog) .>= 1)
        @test all(SB.sampler_stat(chn, :step_size) .> 0)
        @test 0 < mean(SB.bfmi(chn)) < 2
    end
end
