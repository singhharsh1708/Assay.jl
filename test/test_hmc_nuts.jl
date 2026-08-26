@testset "gradient-based samplers" begin
    rng = Random.Xoshiro(777)
    data_b = [rand(rng) < 0.35 ? 1 : 0 for _ in 1:100]
    data_n = [1.5 .+ 0.8 * randn(rng) for _ in 1:50]
    data_p = [AS.rand(rng, AS.Poisson(4.0)) for _ in 1:60]

    @testset "$(nameof(typeof(spl))) on conjugate models" for spl in (AS.HMC(), AS.NUTS())
        b = AS.beta_bernoulli(data_b; a = 2.0, b = 2.0)
        chn = AS.sample(b.model, spl, 4000; n_warmup = 1000, n_chains = 4, rng = Random.Xoshiro(11))
        check_conjugate(chn, :p, b.posterior.p)

        nn = AS.normal_normal(data_n; mu0 = 0.0, tau0 = 5.0, sigma = 0.8)
        chn = AS.sample(nn.model, spl, 4000; n_warmup = 1000, n_chains = 4, rng = Random.Xoshiro(12))
        check_conjugate(chn, :mu, nn.posterior.mu)

        gp = AS.gamma_poisson(data_p; a = 2.0, b = 1.0)
        chn = AS.sample(gp.model, spl, 4000; n_warmup = 1000, n_chains = 4, rng = Random.Xoshiro(13))
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
        gp = AS.gamma_poisson(data_p)
        achieved = Float64[]
        for target in (0.65, 0.8, 0.95)
            chn = AS.sample(gp.model, AS.NUTS(; target_accept = target), 2000;
                            n_warmup = 1000, n_chains = 2, rng = Random.Xoshiro(14))
            push!(achieved, AS.acceptance_rate(chn))
            @test target - 0.03 <= achieved[end] <= target + 0.15
        end
        @test issorted(achieved)
        # With the metric fixed there is nothing to restart the dual averaging,
        # and the target is hit tightly.
        chn = AS.sample(gp.model, AS.NUTS(; target_accept = 0.7, adapt_metric = false), 2000;
                        n_warmup = 1000, n_chains = 2, rng = Random.Xoshiro(21))
        @test abs(AS.acceptance_rate(chn) - 0.7) < 0.05
    end

    @testset "the averaged step size is the stable one" begin
        # The sampling phase uses the step size averaged in log space, which
        # overshoots the target acceptance rate slightly. The alternative, the
        # last raw iterate, lands closer on average and varies enormously
        # between runs. This asserts the trade rather than the overshoot.
        d = 6
        target = AS.MvNormal(zeros(d), Matrix{Float64}(I, d, d))
        model = AS.Model((x = AS.unconstrained(d),), t -> AS.logpdf(target, t.x))

        function achieved(use_raw, seed)
            spl = AS.NUTS(; target_accept = 0.8)
            y0 = AS.random_init(Random.Xoshiro(seed), model)
            st = AS.init_state(Random.Xoshiro(seed), model, spl, y0; n_warmup = 600)
            rng = Random.Xoshiro(seed + 1000)
            for _ in 1:600
                AS.step!(rng, model, spl, st, true)
            end
            st.step_size = use_raw ? exp(st.da.logeps) : AS.da_final(st.da)
            acc = Float64[]
            for _ in 1:800
                _, stats = AS.step!(rng, model, spl, st, false)
                push!(acc, stats.accept_prob)
            end
            return Statistics.mean(acc)
        end

        averaged = [achieved(false, s) for s in 1:5]
        raw = [achieved(true, s) for s in 1:5]
        @test Statistics.std(averaged) < Statistics.std(raw) / 3
        @test all(a -> 0.75 < a < 0.95, averaged)      # overshoots, but predictably
    end

    @testset "metrics on an ill-conditioned Gaussian" begin
        d = 8
        sds = 10 .^ range(-1, 1, length = d)
        target = AS.MvNormal(zeros(d), Matrix(Diagonal(sds .^ 2)))
        model = AS.Model((x = AS.unconstrained(d),), theta -> AS.logpdf(target, theta.x))
        for metric in (:unit, :diag, :dense)
            chn = AS.sample(model, AS.NUTS(; metric = metric), 2000;
                            n_warmup = 1000, n_chains = 4, rng = Random.Xoshiro(15))
            for i in 1:d
                x = chn[Symbol("x[$i]")]
                @test check_mean(x, 0.0; nse = 5)
                @test check_std(x, sds[i]; nse = 5)
            end
            @test AS.divergences(chn) == 0
        end
    end

    @testset "dense metric recovers correlation" begin
        Sigma = [1.0 0.95; 0.95 1.0]
        target = AS.MvNormal([0.0, 0.0], Sigma)
        model = AS.Model((x = AS.unconstrained(2),), theta -> AS.logpdf(target, theta.x))
        chn = AS.sample(model, AS.NUTS(; metric = :dense), 3000;
                        n_warmup = 1000, n_chains = 4, rng = Random.Xoshiro(16))
        @test cor(vec(chn[Symbol("x[1]")]), vec(chn[Symbol("x[2]")])) ≈ 0.95 atol = 0.01
    end

    @testset "the U-turn criteria agree on an easy target" begin
        nn = AS.normal_normal(data_n; sigma = 0.8)
        for crit in (AS.ClassicUTurn(), AS.GeneralizedUTurn(), AS.StrictGeneralizedUTurn())
            chn = AS.sample(nn.model, AS.NUTS(; uturn = crit), 3000;
                            n_warmup = 1000, n_chains = 4, rng = Random.Xoshiro(17))
            check_conjugate(chn, :mu, nn.posterior.mu)
        end
    end

    @testset "divergences are reported on the funnel" begin
        funnel = AS.Model((v = AS.unconstrained(), x = AS.unconstrained(5)),
                          theta -> AS.logpdf(AS.Normal(0.0, 3.0), theta.v) +
                                   sum(AS.logpdf(AS.Normal(0.0, exp(theta.v / 2)), xi) for xi in theta.x))
        chn = AS.sample(funnel, AS.NUTS(; target_accept = 0.8), 2000;
                        n_warmup = 1000, n_chains = 4, rng = Random.Xoshiro(18))
        @test AS.divergences(chn) > 0
        @test all(isfinite, AS.bfmi(chn))
        # the sampler under-covers the neck: this is the documented failure, and
        # a run that reported no divergences here would be the real problem
        @test std(vec(chn[:v])) < 3.0
    end

    @testset "finite differences and forward mode give the same posterior" begin
        nn = AS.normal_normal(data_n; sigma = 0.8)
        a = AS.sample(nn.model, AS.NUTS(; backend = AS.ForwardDiffAD()), 1000;
                      n_warmup = 500, n_chains = 2, rng = Random.Xoshiro(19))
        b = AS.sample(nn.model, AS.NUTS(; backend = AS.FiniteDiffAD()), 1000;
                      n_warmup = 500, n_chains = 2, rng = Random.Xoshiro(19))
        @test abs(mean(vec(a[:mu])) - mean(vec(b[:mu]))) < 4 * AS.mcse_mean(a[:mu])
    end

    @testset "reported statistics are present and sane" begin
        gp = AS.gamma_poisson(data_p)
        chn = AS.sample(gp.model, AS.NUTS(), 1000; n_warmup = 500, n_chains = 2,
                        rng = Random.Xoshiro(20))
        for k in (:accept_prob, :divergent, :treedepth, :n_leapfrog, :step_size, :energy)
            @test haskey(chn.stats, k)
        end
        @test all(0 .<= AS.sampler_stat(chn, :accept_prob) .<= 1)
        @test all(AS.sampler_stat(chn, :n_leapfrog) .>= 1)
        @test all(AS.sampler_stat(chn, :step_size) .> 0)
        @test 0 < mean(AS.bfmi(chn)) < 2
    end
end
