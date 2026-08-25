@testset "uniformity by distribution function" begin
    @testset "the binomial quantile" begin
        @test AS.binomial_quantile(10, 0.5, 0.5) == 5
        @test AS.binomial_quantile(10, 0.5, 0.0) == 0
        @test AS.binomial_quantile(10, 0.5, 1.0) == 10
        # against the definition
        for n in (5, 20), p in (0.2, 0.5), q in (0.1, 0.5, 0.9)
            k = AS.binomial_quantile(n, p, q)
            cdf_at_k = sum(exp(AS.logpdf(AS.Binomial(n, p), j)) for j in 0:k)
            @test cdf_at_k >= q - 1e-9
            if k > 0
                below = sum(exp(AS.logpdf(AS.Binomial(n, p), j)) for j in 0:(k - 1))
                @test below < q
            end
        end
    end

    @testset "the band has the coverage it claims" begin
        # This is the property the whole construction exists for, and it is
        # checked by simulation against the exact computation that produced the
        # band. A pointwise band would sit near 0.6 here, not 0.95.
        L = 64
        for n_sims in (100, 200)
            _, lower, upper = AS.ecdf_simultaneous_band(n_sims; n_grid = 50, confidence = 0.95)
            @test all(lower .<= upper)
            rng = Random.Xoshiro(1)
            hits = 0
            trials = 300
            for _ in 1:trials
                ranks = rand(rng, 0:L, n_sims)
                _, curve = AS.rank_ecdf(ranks, L; n_grid = 50)
                all(lower .<= curve .<= upper) && (hits += 1)
            end
            @test 0.90 <= hits / trials <= 0.99
        end
        @test_throws ArgumentError AS.ecdf_simultaneous_band(5)
        @test_throws DomainError AS.ecdf_simultaneous_band(50; confidence = 1.5)
    end

    @testset "a wider band covers more" begin
        _, lo95, hi95 = AS.ecdf_simultaneous_band(100; n_grid = 30, confidence = 0.95)
        _, lo99, hi99 = AS.ecdf_simultaneous_band(100; n_grid = 30, confidence = 0.99)
        @test all(hi99 .>= hi95)
        @test all(lo99 .<= lo95)
    end

    @testset "miscalibration is caught" begin
        L = 64
        rng = Random.Xoshiro(2)
        shift(r) = clamp.(round.(Int, r .* 0.85 .+ 6), 0, L)
        spread(r) = clamp.(round.(Int, (r .- L / 2) .* 1.6 .+ L / 2), 0, L)
        for distort in (shift, spread)
            caught = 0
            for _ in 1:100
                ranks = distort(rand(rng, 0:L, 200))
                caught += AS.rank_uniformity_ecdf(ranks, L; n_grid = 50).inside ? 0 : 1
            end
            @test caught / 100 > 0.9
        end
        # and a correct sampler is not
        ranks = rand(rng, 0:L, 200)
        r = AS.rank_uniformity_ecdf(ranks, L; n_grid = 50)
        @test r.inside
        @test 0 <= r.max_deviation < 0.2
        @test length(r.grid) == length(r.ecdf) == 50
    end

    @testset "the exact coverage computation agrees with simulation" begin
        # ecdf_band_coverage is a dynamic programme; check it against draws
        n_sims, n_grid = 60, 12
        grid = collect(range(1 / (n_grid + 1), n_grid / (n_grid + 1); length = n_grid))
        lower = [AS.binomial_quantile(n_sims, z, 0.05) - 1 for z in grid]
        upper = [AS.binomial_quantile(n_sims, z, 0.95) for z in grid]
        exact = AS.ecdf_band_coverage(n_sims, grid, lower, upper)
        rng = Random.Xoshiro(3)
        hits = 0
        trials = 4000
        for _ in 1:trials
            u = rand(rng, n_sims)
            counts = [count(<=(z), u) for z in grid]
            all(lower .<= counts .<= upper) && (hits += 1)
        end
        @test exact ≈ hits / trials atol = 0.03
        @test 0 <= exact <= 1
    end

    @testset "calibration reports both tests" begin
        prob = AS.conjugate_problem(AS.gamma_poisson, 20; a = 2.0, b = 1.0)
        res = AS.sbc(Random.Xoshiro(1), prob, AS.NUTS(); n_sims = 200, n_draws = 64,
                     thin = 6, n_warmup = 400)
        @test length(res.ecdf_inside) == length(res.names)
        @test all(res.ecdf_inside)
        @test all(0 .<= res.max_deviation .< 0.2)
        @test AS.calibrated(res)
        @test occursin("in the band", sprint(show, res))
    end
end
