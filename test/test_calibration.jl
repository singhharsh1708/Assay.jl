@testset "calibration against the joint distribution" begin
    @testset "simulation based calibration is uniform for a correct sampler" begin
        # Rank statistics of the prior draw among posterior draws must be
        # uniform on 0:n_draws. The chi-square p value is the summary; the
        # histogram is in docs/results.md.
        for (builder, kwargs, n) in ((gamma_poisson, (a = 2.0, b = 1.0), 20),
                                     (beta_bernoulli, (a = 2.0, b = 3.0), 25),
                                     (normal_normal, (mu0 = 0.0, tau0 = 2.0, sigma = 1.0), 15))
            prob = AS.conjugate_problem(builder, n; kwargs...)
            res = AS.sbc(Random.Xoshiro(4), prob, AS.NUTS(); n_sims = 200, n_draws = 64,
                         thin = 6, n_warmup = 400)
            @test all(res.pvalue .> 0.01)
            counts = AS.rank_histogram(res, 1)
            @test sum(counts) == 200
            @test maximum(counts) < 2.5 * (200 / res.n_bins)     # no spike in any bin
        end
    end

    @testset "calibration holds for the random walk too" begin
        prob = AS.conjugate_problem(AS.beta_bernoulli, 20; a = 1.0, b = 1.0)
        # The random walk needs heavier thinning than NUTS to give near-independent
        # draws; too little and the rank histogram is bathtub-shaped even though
        # the sampler is correct.
        res = AS.sbc(Random.Xoshiro(5), prob, AS.RandomWalkMH(); n_sims = 150, n_draws = 64,
                     thin = 60, n_warmup = 1000)
        @test all(res.pvalue .> 0.01)
    end

    @testset "Geweke: the two joint simulators agree" begin
        for (builder, kwargs, n) in ((gamma_poisson, (a = 2.0, b = 1.0), 20),
                                     (beta_bernoulli, (a = 2.0, b = 3.0), 25),
                                     (normal_normal, (mu0 = 0.0, tau0 = 2.0, sigma = 1.0), 15))
            prob = AS.conjugate_problem(builder, n; kwargs...)
            res = AS.geweke(Random.Xoshiro(6), prob, AS.NUTS(); n_marginal = 40_000,
                            n_successive = 20_000, n_steps = 5)
            @test all(abs.(res.z_mean) .< 3)
            @test all(abs.(res.z_second) .< 3)
        end
    end
end
