@testset "gradient-informed random walks" begin
    rng = Random.Xoshiro(24601)

    @testset "both recover the conjugate posteriors" begin
        for spl in (AS.MALA(), AS.Barker())
            for (name, ref) in (("beta-bernoulli",
                                 AS.beta_bernoulli([1, 0, 1, 1, 0, 1, 1, 1, 0, 1];
                                                   a = 2.0, b = 2.0)),
                                ("normal-normal",
                                 AS.normal_normal(2.0 .+ randn(Random.Xoshiro(1), 40);
                                                  mu0 = 0.0, tau0 = 5.0, sigma = 1.0)),
                                ("gamma-poisson",
                                 AS.gamma_poisson([AS.rand(Random.Xoshiro(2), AS.Poisson(4.0))
                                                   for _ in 1:30]; a = 2.0, b = 1.0)))
                chn = AS.sample(ref.model, spl, 20_000; n_warmup = 5000, n_chains = 4,
                                rng = Random.Xoshiro(7))
                pname = only(AS.parameter_names(ref.model))
                post = only(values(ref.posterior))
                @test check_mean(chn[pname], AS.mean(post); nse = 4)
                @test check_std(chn[pname], sqrt(AS.var(post)); nse = 4)
                @test AS.rhat(chn[pname]) < 1.01
            end
        end
    end

    @testset "the Hastings correction is what makes them right" begin
        # Dropping it leaves a chain that still moves and targets the wrong
        # distribution, which is the failure mode that does not announce itself.
        ref = AS.gamma_poisson([AS.rand(rng, AS.Poisson(4.0)) for _ in 1:30]; a = 2.0, b = 1.0)
        for spl in (AS.MALA(), AS.Barker())
            st = AS.init_state(Random.Xoshiro(3), ref.model, spl, [log(4.0)]; n_warmup = 0)
            prop = AS._propose(Random.Xoshiro(4), spl, st, 0.5)
            _, gp = AS.logdensity_and_gradient(ref.model, prop)
            @test AS._log_hastings(spl, st, prop, gp, 0.5) != 0
        end
    end

    @testset "adaptation reaches the acceptance rate it aims at" begin
        y = 2.0 .+ randn(rng, 40)
        ref = AS.normal_normal(y; mu0 = 0.0, tau0 = 5.0, sigma = 1.0)
        for spl in (AS.MALA(), AS.Barker())
            chn = AS.sample(ref.model, spl, 10_000; n_warmup = 5000, n_chains = 2,
                            rng = Random.Xoshiro(8))
            @test AS.acceptance_rate(chn) ≈ spl.target_accept atol = 0.12
        end
    end

    @testset "Barker survives a step size MALA does not" begin
        # The property the Barker proposal exists for, on a standard normal in
        # ten dimensions with adaptation off so the step size stays put.
        # Measured across step sizes, acceptance and bulk effective sample size:
        #
        #   eps    MALA          Barker
        #   1.0    0.70, 10763   0.57, 5799
        #   1.5    0.21,  4271   0.25, 3730
        #   2.0    0.008,  109   0.089, 1369
        #   3.0    0.000,    2   0.012,  148
        #
        # MALA is the better sampler at a step size someone chose well and is
        # gone by three times it, because its drift term grows with the square
        # of the step. Barker's move is bounded by the step it drew, so it
        # degrades towards a random walk instead.
        d = 10
        model = AS.Model((x = AS.unconstrained(d),),
                         t -> sum(AS.logpdf(AS.Normal(0.0, 1.0), xi) for xi in t.x))
        too_big = 3.0
        mala = AS.sample(model, AS.MALA(; step_size = too_big, adapt_step_size = false,
                                        adapt_precond = false), 20_000;
                         n_warmup = 500, n_chains = 2, rng = Random.Xoshiro(9))
        barker = AS.sample(model, AS.Barker(; step_size = too_big, adapt_step_size = false,
                                            adapt_precond = false), 20_000;
                           n_warmup = 500, n_chains = 2, rng = Random.Xoshiro(9))

        @test AS.acceptance_rate(mala) < 0.001
        @test AS.ess_bulk(mala[Symbol("x[1]")]) < 10
        @test AS.acceptance_rate(barker) > 0.005
        @test AS.ess_bulk(barker[Symbol("x[1]")]) > 50

        # And the failure is silent, which is the part that matters. A chain
        # that never moves reports the spread of wherever it started, not the
        # spread of the posterior, and nothing about the numbers says so.
        @test Statistics.std(vec(mala[Symbol("x[1]")])) < 0.6      # true value is 1
        @test check_std(barker[Symbol("x[1]")], 1.0; nse = 5)
    end

    @testset "at a well chosen step size MALA is the faster of the two" begin
        # The other half of the comparison, so the robustness claim above is not
        # mistaken for a claim that Barker is simply better.
        d = 10
        model = AS.Model((x = AS.unconstrained(d),),
                         t -> sum(AS.logpdf(AS.Normal(0.0, 1.0), xi) for xi in t.x))
        mala = AS.sample(model, AS.MALA(), 20_000; n_warmup = 5000, n_chains = 2,
                         rng = Random.Xoshiro(10))
        barker = AS.sample(model, AS.Barker(), 20_000; n_warmup = 5000, n_chains = 2,
                           rng = Random.Xoshiro(10))
        @test AS.ess_bulk(mala[Symbol("x[1]")]) > AS.ess_bulk(barker[Symbol("x[1]")])
        for chn in (mala, barker)
            @test check_mean(chn[Symbol("x[1]")], 0.0; nse = 4)
            @test check_std(chn[Symbol("x[1]")], 1.0; nse = 4)
        end
    end

    @testset "the preconditioner is learned" begin
        # Scales three orders of magnitude apart. Without a preconditioner one
        # step size cannot serve both coordinates.
        model = AS.Model((x = AS.unconstrained(2),),
                         t -> AS.logpdf(AS.Normal(0.0, 100.0), t.x[1]) +
                              AS.logpdf(AS.Normal(0.0, 0.1), t.x[2]))
        for spl in (AS.MALA(), AS.Barker())
            chn = AS.sample(model, spl, 40_000; n_warmup = 10_000, n_chains = 4,
                            rng = Random.Xoshiro(11))
            @test check_std(chn[Symbol("x[1]")], 100.0; nse = 6)
            @test check_std(chn[Symbol("x[2]")], 0.1; nse = 6)
        end
    end

    @testset "calibrated against the joint distribution" begin
        for spl in (AS.MALA(), AS.Barker())
            prob = AS.conjugate_problem(AS.beta_bernoulli, 12; a = 2.0, b = 2.0)
            r = AS.sbc(Random.Xoshiro(12), prob, spl; n_sims = 150, n_draws = 63,
                       n_warmup = 2000)
            @test AS.calibrated(r)
        end
    end
end
