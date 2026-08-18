@testset "calibration against the joint distribution" begin
    @testset "simulation based calibration is uniform for a correct sampler" begin
        # Rank statistics of the prior draw among posterior draws must be
        # uniform on 0:n_draws. The chi-square p value is the summary; the
        # histogram is in docs/results.md.
        for (builder, kwargs, n) in ((gamma_poisson, (a = 2.0, b = 1.0), 20),
                                     (beta_bernoulli, (a = 2.0, b = 3.0), 25),
                                     (normal_normal, (mu0 = 0.0, tau0 = 2.0, sigma = 1.0), 15))
            prob = SB.conjugate_problem(builder, n; kwargs...)
            res = SB.sbc(Random.Xoshiro(4), prob, SB.NUTS(); n_sims = 200, n_draws = 64,
                         thin = 6, n_warmup = 400)
            @test all(res.pvalue .> 0.01)
            counts = SB.rank_histogram(res, 1)
            @test sum(counts) == 200
            @test maximum(counts) < 2.5 * (200 / res.n_bins)     # no spike in any bin
        end
    end

    @testset "calibration holds for the random walk too" begin
        prob = SB.conjugate_problem(SB.beta_bernoulli, 20; a = 1.0, b = 1.0)
        res = SB.sbc(Random.Xoshiro(5), prob, SB.RandomWalkMH(); n_sims = 150, n_draws = 64,
                     thin = 20, n_warmup = 1000)
        @test all(res.pvalue .> 0.01)
    end

    @testset "Geweke: the two joint simulators agree" begin
        for (builder, kwargs, n) in ((gamma_poisson, (a = 2.0, b = 1.0), 20),
                                     (beta_bernoulli, (a = 2.0, b = 3.0), 25),
                                     (normal_normal, (mu0 = 0.0, tau0 = 2.0, sigma = 1.0), 15))
            prob = SB.conjugate_problem(builder, n; kwargs...)
            res = SB.geweke(Random.Xoshiro(6), prob, SB.NUTS(); n_marginal = 40_000,
                            n_successive = 20_000, n_steps = 5)
            @test all(abs.(res.z_mean) .< 3)
            @test all(abs.(res.z_second) .< 3)
        end
    end
end
