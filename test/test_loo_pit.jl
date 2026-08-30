@testset "leave-one-out predictive calibration" begin
    rng = Random.Xoshiro(6161)

    # One fitting helper, used for both the correct model and the wrong one, so
    # the only difference between the two cases below is the data.
    function fit_normal(y; seed)
        ref = AS.normal_normal(y; mu0 = 0.0, tau0 = 5.0, sigma = 1.0)
        chn = AS.sample(ref.model, AS.NUTS(), 3000; n_warmup = 1000, n_chains = 4,
                        rng = Random.Xoshiro(seed))
        ll = AS.pointwise_log_likelihood(ref.model, chn,
                                         (t, i) -> AS.logpdf(AS.Normal(t.mu, 1.0), y[i]);
                                         n_obs = length(y), thin = 4)
        cd = AS.pointwise_cdf(ref.model, chn,
                              (t, i) -> AS.cdf(AS.Normal(t.mu, 1.0), y[i]);
                              n_obs = length(y), thin = 4)
        return ll, cd
    end

    @testset "a correctly specified model is uniform" begin
        y = 2.0 .+ randn(rng, 120)
        ll, cd = fit_normal(y; seed = 1)
        r = AS.loo_pit(ll, cd)
        @test AS.calibrated(r)
        @test length(r.pit) == length(y)
        @test all(0 .<= r.pit .<= 1)
        @test isempty(AS.problematic(r))
        @test r.max_deviation < 0.2
        @test occursin("uniform", sprint(show, r))
        # the transforms average about a half, as uniform values do
        @test Statistics.mean(r.pit) ≈ 0.5 atol = 0.06
    end

    @testset "heavy tails under a normal model are caught" begin
        # Heavy-tailed data under a normal model with known scale, which is the
        # same failure test_usability.jl checks with a posterior predictive
        # statistic. There, the check passes or fails depending entirely on
        # which statistic is chosen. Here nothing has to be chosen.
        y = 2.0 .+ [AS.rand(rng, AS.StudentT(2.0, 0.0, 1.0)) for _ in 1:120]
        ll, cd = fit_normal(y; seed = 2)
        r = AS.loo_pit(ll, cd)
        @test !AS.calibrated(r)
        @test r.max_deviation > 0.1
        @test occursin("not uniform", sprint(show, r))

        # the comparison that makes the point: on the same fit, a posterior
        # predictive check on the mean sees nothing wrong
        ref = AS.normal_normal(y; mu0 = 0.0, tau0 = 5.0, sigma = 1.0)
        chn = AS.sample(ref.model, AS.NUTS(), 3000; n_warmup = 1000, n_chains = 4,
                        rng = Random.Xoshiro(2))
        yrep = AS.predictive(ref.model, chn,
                             (t, g) -> [AS.rand(g, AS.Normal(t.mu, 1.0)) for _ in eachindex(y)];
                             rng = Random.Xoshiro(22), thin = 20)
        @test 0.05 < AS.predictive_check(y, yrep, Statistics.mean).pvalue < 0.95
    end

    @testset "the transforms are the weighted cdf, not the plain one" begin
        # Against a direct computation with the same weights, so the difference
        # between this and a naive average is on the record.
        y = 1.0 .+ randn(rng, 30)
        ll, cd = fit_normal(y; seed = 3)
        r = AS.loo_pit(ll, cd)
        for i in (1, 7, 30)
            w = exp.(AS.psis(-view(ll, :, i)).log_weights)
            @test r.pit[i] ≈ sum(w .* view(cd, :, i))
        end
        plain = [Statistics.mean(view(cd, :, i)) for i in axes(cd, 2)]
        @test r.pit != plain
    end

    @testset "discrete data needs the randomised transform" begin
        # A Poisson model fitted to Poisson data is correct, and the unrandomised
        # transform still fails, because a discrete cdf lands on a lattice rather
        # than spreading over the interval. Reporting that as a bad model would
        # be blaming the model for the arithmetic.
        y = [AS.rand(rng, AS.Poisson(4.0)) for _ in 1:150]
        ref = AS.gamma_poisson(y; a = 2.0, b = 1.0)
        chn = AS.sample(ref.model, AS.NUTS(), 3000; n_warmup = 1000, n_chains = 4,
                        rng = Random.Xoshiro(4))
        ll = AS.pointwise_log_likelihood(ref.model, chn,
                                         (t, i) -> AS.logpdf(AS.Poisson(t.lambda), y[i]);
                                         n_obs = length(y), thin = 4)
        hi = AS.pointwise_cdf(ref.model, chn, (t, i) -> AS.cdf(AS.Poisson(t.lambda), y[i]);
                              n_obs = length(y), thin = 4)
        lo = AS.pointwise_cdf(ref.model, chn,
                              (t, i) -> y[i] == 0 ? 0.0 : AS.cdf(AS.Poisson(t.lambda), y[i] - 1);
                              n_obs = length(y), thin = 4)

        plain = AS.loo_pit(ll, hi)
        randomised = AS.loo_pit(ll, hi; predictive_cdf_lower = lo, rng = Random.Xoshiro(5))
        @test !AS.calibrated(plain)                    # the arithmetic, not the model
        @test AS.calibrated(randomised)
        @test randomised.max_deviation < plain.max_deviation
        @test all(0 .<= randomised.pit .<= 1)
    end

    @testset "input validation" begin
        ll = randn(rng, 200, 5)
        @test_throws DimensionMismatch AS.loo_pit(ll, rand(rng, 200, 4))
        @test_throws ArgumentError AS.loo_pit(randn(rng, 5, 3), rand(rng, 5, 3))
        @test_throws ArgumentError AS.loo_pit(ll, rand(rng, 200, 5);
                                              predictive_cdf_lower = rand(rng, 200, 5))
        @test_throws DimensionMismatch AS.loo_pit(ll, rand(rng, 200, 5);
                                                  predictive_cdf_lower = rand(rng, 200, 4),
                                                  rng = Random.Xoshiro(1))
    end

    @testset "the uniformity test itself" begin
        # Factored out of the rank version, so it has to agree with it.
        ranks = [3, 7, 0, 12, 5, 9, 1, 14, 6, 8, 2, 11, 4, 13, 10, 15]
        n_draws = 15
        a = AS.rank_uniformity_ecdf(ranks, n_draws)
        b = AS.uniformity_ecdf([(r + 0.5) / (n_draws + 1) for r in ranks])
        @test a.inside == b.inside
        @test a.max_deviation ≈ b.max_deviation
        @test a.ecdf == b.ecdf

        # uniform draws stay inside, a visibly sloped set does not
        @test AS.uniformity_ecdf(rand(Random.Xoshiro(9), 200)).inside
        @test !AS.uniformity_ecdf(rand(Random.Xoshiro(9), 200) .^ 2).inside
    end
end
