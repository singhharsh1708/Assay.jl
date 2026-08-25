using Distributions: Distributions
const D = Distributions

@testset "densities against Distributions.jl" begin
    @testset "log densities" begin
        cases = [
            (AS.Normal(0.3, 2.0), D.Normal(0.3, 2.0), [-3.0, 0.0, 1.7]),
            (AS.LogNormal(0.1, 0.7), D.LogNormal(0.1, 0.7), [0.2, 1.0, 4.0]),
            (AS.Cauchy(-1.0, 3.0), D.Cauchy(-1.0, 3.0), [-10.0, 0.0, 8.0]),
            (AS.Uniform(-1.0, 2.0), D.Uniform(-1.0, 2.0), [-0.5, 1.9]),
            (AS.Exponential(2.5), D.Exponential(1 / 2.5), [0.1, 3.0]),
            (AS.Gamma(2.5, 1.7), D.Gamma(2.5, 1 / 1.7), [0.3, 2.0, 9.0]),
            (AS.InverseGamma(3.0, 2.0), D.InverseGamma(3.0, 2.0), [0.4, 1.0, 5.0]),
            (AS.Beta(2.0, 5.0), D.Beta(2.0, 5.0), [0.05, 0.4, 0.95]),
            (AS.Bernoulli(0.3), D.Bernoulli(0.3), [0, 1]),
            (AS.Binomial(10, 0.3), D.Binomial(10, 0.3), [0, 3, 10]),
            (AS.Poisson(4.2), D.Poisson(4.2), [0, 4, 12]),
        ]
        for (mine, theirs, xs) in cases
            for x in xs
                @test AS.logpdf(mine, x) ≈ D.logpdf(theirs, x) rtol = 1e-10
            end
        end
        # Student t is stated in location-scale form here
        for x in (-4.0, 0.0, 2.2)
            @test AS.logpdf(AS.StudentT(5.0, 1.0, 2.0), x) ≈
                  D.logpdf(D.TDist(5.0), (x - 1) / 2) - log(2.0) rtol = 1e-10
        end
        Sig = [2.0 0.6; 0.6 1.0]
        @test AS.logpdf(AS.MvNormal([1.0, -1.0], Sig), [0.3, 0.4]) ≈
              D.logpdf(D.MvNormal([1.0, -1.0], Sig), [0.3, 0.4]) rtol = 1e-10
        al = [1.5, 2.0, 0.7]
        x = [0.2, 0.5, 0.3]
        @test AS.logpdf(AS.Dirichlet(al), x) ≈ D.logpdf(D.Dirichlet(al), x) rtol = 1e-10
    end

    @testset "the LKJ density against Distributions.jl" begin
        for K in (2, 3, 4, 5), eta in (0.8, 1.0, 3.0)
            mine = AS.LKJCholesky(K, eta)
            theirs = D.LKJCholesky(K, eta)
            L = rand(Random.Xoshiro(K * 7 + round(Int, eta * 10)), theirs).L
            @test AS.logpdf(mine, L) ≈
                  D.logpdf(theirs, LinearAlgebra.Cholesky(Matrix(L), :L, 0)) rtol = 1e-10
        end
        # the normalising constant is a constant, which is what makes the
        # density usable for evidence and not only for sampling
        d = AS.LKJCholesky(4, 2.5)
        cs = [AS.logpdf(d, rand(Random.Xoshiro(s), D.LKJCholesky(4, 2.5)).L) -
              sum((4 - i + 2 * 2.5 - 2) * log(rand(Random.Xoshiro(s), D.LKJCholesky(4, 2.5)).L[i, i])
                  for i in 2:4) for s in 1:5]
        @test all(c -> c ≈ cs[1], cs)
        @test cs[1] ≈ AS.lkj_log_constant(4, 2.5)
    end

    @testset "support" begin
        @test AS.logpdf(AS.Gamma(2.0, 1.0), -0.1) == -Inf
        @test AS.logpdf(AS.Beta(2.0, 2.0), 1.2) == -Inf
        @test AS.logpdf(AS.Poisson(1.0), 1.5) == -Inf
        @test AS.logpdf(AS.Poisson(1.0), -1) == -Inf
        @test AS.logpdf(AS.Normal(0.0, -1.0), 0.0) == -Inf
        @test AS.logpdf(AS.Uniform(0.0, 1.0), 2.0) == -Inf
    end

    @testset "samplers match their own densities" begin
        rng = Random.Xoshiro(20240)
        n = 200_000
        for d in (AS.Normal(1.0, 2.0), AS.Gamma(2.5, 1.7), AS.Beta(2.0, 5.0),
                  AS.Exponential(1.4), AS.Poisson(4.2), AS.Poisson(45.0),
                  AS.Uniform(-1.0, 3.0), AS.LogNormal(0.0, 0.4))
            x = AS.rand(rng, d, n)
            m = AS.mean(d)
            s = sqrt(AS.var(d))
            @test abs(mean(x) - m) < 5 * s / sqrt(n)
            @test abs(std(x) - s) < 6 * s / sqrt(2n)
        end
        # Poisson above the lambda = 30 branch point must still be exact, not
        # normal-approximated: check the pmf of a few atoms directly.
        x = AS.rand(rng, AS.Poisson(45.0), n)
        for k in (30, 45, 60)
            phat = count(==(k), x) / n
            p = exp(AS.logpdf(AS.Poisson(45.0), k))
            @test abs(phat - p) < 5 * sqrt(p * (1 - p) / n)
        end
    end

    @testset "cdf and quantile" begin
        for d in (AS.Normal(0.5, 1.5), AS.Beta(2.0, 3.0), AS.Gamma(3.0, 2.0), AS.Exponential(1.2))
            for p in (0.001, 0.05, 0.5, 0.9, 0.999)
                q = AS.quantile(d, p)
                @test AS.cdf(d, q) ≈ p atol = 1e-8
            end
        end
        @test AS.cdf(AS.Normal(0.0, 1.0), 1.96) ≈ D.cdf(D.Normal(), 1.96) rtol = 1e-12
        @test AS.quantile(AS.Beta(2.0, 3.0), 0.3) ≈ D.quantile(D.Beta(2.0, 3.0), 0.3) rtol = 1e-8
        @test AS.quantile(AS.Gamma(3.0, 2.0), 0.7) ≈ D.quantile(D.Gamma(3.0, 0.5), 0.7) rtol = 1e-8
    end
end
