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
            @test AS.ess(x) ≈ expected rtol = 0.15
        end
        # An antithetic chain carries more information than its length: this is
        # the case a naive "ESS cannot exceed n" clamp would silently corrupt,
        # and it is the regime NUTS actually operates in.
        @test AS.ess(ar1(rng, 20_000, 4, -0.5)) > 20_000 * 4
    end

    @testset "R-hat" begin
        x = randn(rng, 4000, 4)
        @test AS.rhat(x) < 1.01
        y = copy(x)
        y[:, 1] .+= 2.0                                  # one chain in the wrong place
        @test AS.rhat(y) > 1.1
        @test AS.rhat(y) > AS.rhat(x)
        # split R-hat catches a single drifting chain that unsplit R-hat cannot
        drift = reshape(range(0, 3; length = 4000) .+ 0.1 .* randn(rng, 4000), 4000, 1)
        @test AS.rhat_plain(drift) > 1.1
        # rank normalisation keeps R-hat finite for an infinite-variance chain
        heavy = reshape([AS.rand(rng, AS.Cauchy(0.0, 1.0)) for _ in 1:8000], 2000, 4)
        @test isfinite(AS.rhat(heavy))
    end

    @testset "rank normalisation" begin
        x = exp.(randn(rng, 2000, 4))
        z = AS.rank_normalize(x)
        @test abs(mean(vec(z))) < 0.05
        @test abs(std(vec(z)) - 1) < 0.05
        @test sortperm(vec(x)) == sortperm(vec(z))       # order preserving
    end

    @testset "Monte Carlo standard error covers the truth" begin
        hits = 0
        trials = 200
        for t in 1:trials
            x = ar1(Random.Xoshiro(1000 + t), 2000, 2, 0.5)
            hits += abs(mean(vec(x))) <= 2 * AS.mcse_mean(x) ? 1 : 0
        end
        @test 0.85 <= hits / trials <= 1.0               # nominal coverage is 0.95
    end

    @testset "the fast Fourier transform against a direct one" begin
        # A hand-written FFT is only worth having if it is checked against the
        # definition it is an optimisation of.
        function dft(x)
            n = length(x)
            [sum(x[j + 1] * cis(-2pi * k * j / n) for j in 0:(n - 1)) for k in 0:(n - 1)]
        end
        for n in (2, 8, 64, 256)
            v = ComplexF64.(randn(rng, n), randn(rng, n))
            @test AS.fft!(copy(v)) ≈ dft(v) rtol = 1e-10
            @test AS.ifft!(AS.fft!(copy(v))) ≈ v rtol = 1e-12
        end
        @test_throws ArgumentError AS.fft!(ComplexF64.(randn(rng, 6)))
        @test AS.next_power_of_two(1) == 1
        @test AS.next_power_of_two(5) == 8
        @test AS.next_power_of_two(1024) == 1024
    end

    @testset "autocovariance by FFT equals the direct sum" begin
        for n in (500, 4096)
            v = randn(rng, n)
            @test AS.autocov_fft(v, 50) ≈ AS.autocov(v, 50) atol = 1e-12
        end
        # and it works out to every lag, which is what lets ESS stop truncating
        v = randn(rng, 1024)
        @test length(AS.autocov_fft(v)) == 1024
        @test AS.autocov_fft(v)[1] ≈ Statistics.var(v) * (length(v) - 1) / length(v)
    end

    @testset "Monte Carlo standard errors against closed forms" begin
        # For independent normal draws every one of these has an analytic value.
        n, m = 20_000, 4
        x = reshape(randn(rng, n * m), n, m)
        N = n * m

        # the median of a normal sample: sd * sqrt(pi / 2) / sqrt(N)
        @test AS.mcse_quantile(x, 0.5) ≈ sqrt(pi / 2) / sqrt(N) rtol = 0.2

        # a general quantile: sqrt(p (1-p) / N) / density at that quantile
        for p in (0.1, 0.25, 0.75, 0.9)
            q = AS.quantile(AS.Normal(0.0, 1.0), p)
            analytic = sqrt(p * (1 - p) / N) / exp(AS.logpdf(AS.Normal(0.0, 1.0), q))
            @test AS.mcse_quantile(x, p) ≈ analytic rtol = 0.25
        end

        # the standard deviation of a normal sample: sd / sqrt(2N)
        @test AS.mcse_std(x) ≈ 1 / sqrt(2N) rtol = 0.15

        # and the effective sample size for a quantile is the draw count when
        # the draws are independent
        @test AS.ess_quantile(x, 0.5) ≈ N rtol = 0.1
        @test_throws DomainError AS.mcse_quantile(x, 0.0)
        @test_throws DomainError AS.ess_quantile(x, 1.0)
    end

    @testset "quantile error grows where the density is small" begin
        # The point of estimating quantile error on the probability scale is
        # that a normal approximation on the value scale fails in the tails.
        x = reshape(randn(rng, 40_000), 10_000, 4)
        @test AS.mcse_quantile(x, 0.99) > AS.mcse_quantile(x, 0.5)
        @test AS.mcse_quantile(x, 0.01) > AS.mcse_quantile(x, 0.5)
    end

    @testset "normal quantile function" begin
        for p in (1e-8, 0.001, 0.025, 0.5, 0.975, 1 - 1e-8)
            @test AS.cdf(AS.Normal(0.0, 1.0), AS.norminvcdf(p)) ≈ p rtol = 1e-8
        end
        @test AS.norminvcdf(0.975) ≈ 1.959963984540054 rtol = 1e-12
    end
end
