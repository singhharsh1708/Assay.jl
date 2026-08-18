using Distributions: Distributions
const D = Distributions

@testset "densities against Distributions.jl" begin
    @testset "log densities" begin
        cases = [
            (SB.Normal(0.3, 2.0), D.Normal(0.3, 2.0), [-3.0, 0.0, 1.7]),
            (SB.LogNormal(0.1, 0.7), D.LogNormal(0.1, 0.7), [0.2, 1.0, 4.0]),
            (SB.Cauchy(-1.0, 3.0), D.Cauchy(-1.0, 3.0), [-10.0, 0.0, 8.0]),
            (SB.Uniform(-1.0, 2.0), D.Uniform(-1.0, 2.0), [-0.5, 1.9]),
            (SB.Exponential(2.5), D.Exponential(1 / 2.5), [0.1, 3.0]),
            (SB.Gamma(2.5, 1.7), D.Gamma(2.5, 1 / 1.7), [0.3, 2.0, 9.0]),
            (SB.InverseGamma(3.0, 2.0), D.InverseGamma(3.0, 2.0), [0.4, 1.0, 5.0]),
            (SB.Beta(2.0, 5.0), D.Beta(2.0, 5.0), [0.05, 0.4, 0.95]),
            (SB.Bernoulli(0.3), D.Bernoulli(0.3), [0, 1]),
            (SB.Binomial(10, 0.3), D.Binomial(10, 0.3), [0, 3, 10]),
            (SB.Poisson(4.2), D.Poisson(4.2), [0, 4, 12]),
        ]
        for (mine, theirs, xs) in cases
            for x in xs
                @test SB.logpdf(mine, x) ≈ D.logpdf(theirs, x) rtol = 1e-10
            end
        end
        # Student t is stated in location-scale form here
        for x in (-4.0, 0.0, 2.2)
            @test SB.logpdf(SB.StudentT(5.0, 1.0, 2.0), x) ≈
                  D.logpdf(D.TDist(5.0), (x - 1) / 2) - log(2.0) rtol = 1e-10
        end
        Sig = [2.0 0.6; 0.6 1.0]
        @test SB.logpdf(SB.MvNormal([1.0, -1.0], Sig), [0.3, 0.4]) ≈
              D.logpdf(D.MvNormal([1.0, -1.0], Sig), [0.3, 0.4]) rtol = 1e-10
        al = [1.5, 2.0, 0.7]
        x = [0.2, 0.5, 0.3]
        @test SB.logpdf(SB.Dirichlet(al), x) ≈ D.logpdf(D.Dirichlet(al), x) rtol = 1e-10
    end

    @testset "support" begin
        @test SB.logpdf(SB.Gamma(2.0, 1.0), -0.1) == -Inf
        @test SB.logpdf(SB.Beta(2.0, 2.0), 1.2) == -Inf
        @test SB.logpdf(SB.Poisson(1.0), 1.5) == -Inf
        @test SB.logpdf(SB.Poisson(1.0), -1) == -Inf
        @test SB.logpdf(SB.Normal(0.0, -1.0), 0.0) == -Inf
        @test SB.logpdf(SB.Uniform(0.0, 1.0), 2.0) == -Inf
    end

    @testset "samplers match their own densities" begin
        rng = Random.Xoshiro(20240)
        n = 200_000
        for d in (SB.Normal(1.0, 2.0), SB.Gamma(2.5, 1.7), SB.Beta(2.0, 5.0),
                  SB.Exponential(1.4), SB.Poisson(4.2), SB.Poisson(45.0),
                  SB.Uniform(-1.0, 3.0), SB.LogNormal(0.0, 0.4))
            x = SB.rand(rng, d, n)
            m = SB.mean(d)
            s = sqrt(SB.var(d))
            @test abs(mean(x) - m) < 5 * s / sqrt(n)
            @test abs(std(x) - s) < 6 * s / sqrt(2n)
        end
        # Poisson above the lambda = 30 branch point must still be exact, not
        # normal-approximated: check the pmf of a few atoms directly.
        x = SB.rand(rng, SB.Poisson(45.0), n)
        for k in (30, 45, 60)
            phat = count(==(k), x) / n
            p = exp(SB.logpdf(SB.Poisson(45.0), k))
            @test abs(phat - p) < 5 * sqrt(p * (1 - p) / n)
        end
    end

    @testset "cdf and quantile" begin
        for d in (SB.Normal(0.5, 1.5), SB.Beta(2.0, 3.0), SB.Gamma(3.0, 2.0), SB.Exponential(1.2))
            for p in (0.001, 0.05, 0.5, 0.9, 0.999)
                q = SB.quantile(d, p)
                @test SB.cdf(d, q) ≈ p atol = 1e-8
            end
        end
        @test SB.cdf(SB.Normal(0.0, 1.0), 1.96) ≈ D.cdf(D.Normal(), 1.96) rtol = 1e-12
        @test SB.quantile(SB.Beta(2.0, 3.0), 0.3) ≈ D.quantile(D.Beta(2.0, 3.0), 0.3) rtol = 1e-8
        @test SB.quantile(SB.Gamma(3.0, 2.0), 0.7) ≈ D.quantile(D.Gamma(3.0, 0.5), 0.7) rtol = 1e-8
    end
end
