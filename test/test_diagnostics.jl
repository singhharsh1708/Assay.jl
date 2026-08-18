"""
Diagnostics are checked against cases where the answer is known analytically:
an AR(1) chain has effective sample size `n (1-r)/(1+r)`, independent chains
from the same distribution have R-hat 1, and chains from different
distributions do not.
"""
function ar1(rng, n, m, r)
    x = zeros(n, m)
    for c in 1:m
        z = randn(rng) 
        for i in 1:n
            z = r * z + sqrt(1 - r^2) * randn(rng)
            x[i, c] = z
        end
    end
    return x
end

@testset "diagnostics" begin
    rng = Random.Xoshiro(202)

    @testset "effective sample size against AR(1) theory" begin
        n, m = 20_000, 4
        for r in (0.0, 0.5, 0.9, -0.5)
            x = ar1(rng, n, m, r)
            expected = n * m * (1 - r) / (1 + r)
            @test SB.ess(x) ≈ expected rtol = 0.15
        end
        # An antithetic chain carries more information than its length: this is
        # the case a naive "ESS cannot exceed n" clamp would silently corrupt,
        # and it is the regime NUTS actually operates in.
        @test SB.ess(ar1(rng, 20_000, 4, -0.5)) > 20_000 * 4
    end

    @testset "R-hat" begin
        x = randn(rng, 4000, 4)
        @test SB.rhat(x) < 1.01
        y = copy(x)
        y[:, 1] .+= 1.0                                  # one chain in the wrong place
        @test SB.rhat(y) > 1.1
        # split R-hat catches a single drifting chain that unsplit R-hat cannot
        drift = reshape(range(0, 3; length = 4000) .+ 0.1 .* randn(rng, 4000), 4000, 1)
        @test SB.rhat_plain(drift) > 1.1
        # rank normalisation keeps R-hat finite for an infinite-variance chain
        heavy = reshape([SB.rand(rng, SB.Cauchy(0.0, 1.0)) for _ in 1:8000], 2000, 4)
        @test isfinite(SB.rhat(heavy))
    end

    @testset "rank normalisation" begin
        x = exp.(randn(rng, 2000, 4))
        z = SB.rank_normalize(x)
        @test abs(mean(vec(z))) < 0.05
        @test abs(std(vec(z)) - 1) < 0.05
        @test sortperm(vec(x)) == sortperm(vec(z))       # order preserving
    end

    @testset "Monte Carlo standard error covers the truth" begin
        hits = 0
        trials = 200
        for t in 1:trials
            x = ar1(Random.Xoshiro(1000 + t), 2000, 2, 0.5)
            hits += abs(mean(vec(x))) <= 2 * SB.mcse_mean(x) ? 1 : 0
        end
        @test 0.85 <= hits / trials <= 1.0               # nominal coverage is 0.95
    end

    @testset "normal quantile function" begin
        for p in (1e-8, 0.001, 0.025, 0.5, 0.975, 1 - 1e-8)
            @test SB.cdf(SB.Normal(0.0, 1.0), SB.norminvcdf(p)) ≈ p rtol = 1e-8
        end
        @test SB.norminvcdf(0.975) ≈ 1.959963984540054 rtol = 1e-12
    end
end
