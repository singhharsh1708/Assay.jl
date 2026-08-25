using ForwardDiff, LinearAlgebra

@testset "correlation matrices" begin
    rng = Random.Xoshiro(4242)

    @testset "the factor really is a correlation Cholesky factor" begin
        for K in (2, 3, 5, 8)
            t = AS.corr_cholesky(K)
            @test AS.udim(t) == K * (K - 1) ÷ 2
            @test AS.cdim(t) == K * (K - 1) ÷ 2
            for scale in (0.5, 3.0)
                y = randn(rng, AS.udim(t)) .* scale
                L, _ = AS.to_constrained(t, y)
                R = AS.correlation_matrix(L)
                @test istril(L)
                @test all(i -> L[i, i] > 0, 1:K)              # positive diagonal
                @test all(i -> norm(L[i, :]) ≈ 1, 1:K)        # unit rows
                @test all(i -> R[i, i] ≈ 1, 1:K)              # unit diagonal
                @test issymmetric(round.(R; digits = 12))
                @test eigmin(Symmetric(R)) > 0                # positive definite
                @test all(abs.(R) .<= 1 + 1e-12)
                @test AS.to_unconstrained(t, L) ≈ y rtol = 1e-9
            end
        end
    end

    @testset "log Jacobian determinant against automatic differentiation" begin
        # The free coordinates on the constrained side are the strictly lower
        # entries; the diagonal is determined by the unit-row constraint. An
        # earlier version excluded the last off-diagonal column of each row by
        # mistake and was wrong by 2.5 nats, which this catches.
        for K in (2, 3, 4, 6)
            t = AS.corr_cholesky(K)
            free = z -> begin
                M, _ = AS.to_constrained(t, z)
                [M[i, j] for i in 2:K for j in 1:(i - 1)]
            end
            for seed in (1, 2, 3)
                y = randn(Random.Xoshiro(seed), AS.udim(t)) .* 1.3
                _, lj = AS.to_constrained(t, y)
                @test lj ≈ log(abs(det(ForwardDiff.jacobian(free, y)))) rtol = 1e-9
            end
        end
    end

    @testset "draws are reported as correlations, not as factor entries" begin
        t = AS.corr_cholesky(3)
        @test AS.flat_names(t, :R) == [Symbol("R[2,1]"), Symbol("R[3,1]"), Symbol("R[3,2]")]
        y = randn(rng, 3)
        L, _ = AS.to_constrained(t, y)
        R = AS.correlation_matrix(L)
        @test AS.flatten(t, L) ≈ [R[2, 1], R[3, 1], R[3, 2]]
        @test AS.flatten(t, L) != [L[2, 1], L[3, 1], L[3, 2]]     # the factor is not the answer
    end

    @testset "the LKJ density" begin
        # matched against Distributions.jl in test_densities.jl; here the shape
        # of the prior is checked against what it is supposed to mean
        L = AS.rand(rng, AS.LKJCholesky(4, 1.0))
        @test size(L) == (4, 4)
        @test AS.correlation_matrix(L)[1, 1] ≈ 1
        @test_throws ArgumentError AS.LKJCholesky(1, 1.0)
        @test_throws DomainError AS.LKJCholesky(3, 0.0)
        # eta above one concentrates towards the identity
        spread(eta) = Statistics.std([AS.correlation_matrix(AS.rand(rng, AS.LKJCholesky(4, eta)))[2, 1]
                                      for _ in 1:20_000])
        @test spread(10.0) < spread(2.0) < spread(0.5)
    end

    @testset "the onion sampler against the closed-form marginal" begin
        # Under LKJ(eta) every off-diagonal correlation is marginally Beta on
        # (-1, 1) with both shapes equal to alpha = eta + (K - 2) / 2, so its
        # variance is exactly 1 / (2 alpha + 1).
        for (K, eta) in ((2, 1.0), (3, 2.0), (5, 2.0), (4, 0.8))
            alpha = eta + (K - 2) / 2
            rs = [AS.correlation_matrix(AS.rand(rng, AS.LKJCholesky(K, eta)))[K, 1]
                  for _ in 1:60_000]
            @test abs(Statistics.mean(rs)) < 0.02
            @test Statistics.var(rs) ≈ 1 / (2 * alpha + 1) rtol = 0.05
        end
    end

    @testset "a Cholesky-parameterised normal agrees with the dense one" begin
        K = 4
        A = randn(rng, K, K)
        Sigma = A * A' + K * I
        L = cholesky(Symmetric(Sigma)).L
        mu = randn(rng, K)
        x = randn(rng, K)
        @test AS.logpdf(AS.MvNormalCholesky(mu, L), x) ≈ AS.logpdf(AS.MvNormal(mu, Sigma), x)
        @test AS.covariance(AS.MvNormalCholesky(mu, L)) ≈ Sigma
        # and it samples the covariance it claims. One generator, advanced by
        # each draw: building a fresh Xoshiro(5) inside the comprehension would
        # produce the same vector 200,000 times, and a covariance of zero.
        drng = Random.Xoshiro(5)
        d = AS.MvNormalCholesky(mu, L)
        draws = reduce(hcat, AS.rand(drng, d) for _ in 1:200_000)
        @test Statistics.cov(draws; dims = 2) ≈ Sigma rtol = 0.05
        @test vec(Statistics.mean(draws; dims = 2)) ≈ mu atol = 0.05
    end

    @testset "fitting a covariance: the posterior tracks the data, not the population" begin
        K, n = 3, 400
        Rpop = [1.0 0.7 -0.4; 0.7 1.0 -0.2; -0.4 -0.2 1.0]
        sdpop = [1.0, 2.0, 0.5]
        Lpop = cholesky(Symmetric(Diagonal(sdpop) * Rpop * Diagonal(sdpop))).L
        drng = Random.Xoshiro(700)
        dpop = AS.MvNormalCholesky(zeros(K), Lpop)
        data = [AS.rand(drng, dpop) for _ in 1:n]

        model = AS.Model((R = AS.corr_cholesky(K), sigma = AS.positive(K)),
                         t -> AS.logpdf(AS.LKJCholesky(K, 2.0), t.R) +
                              sum(AS.logpdf(AS.Gamma(2.0, 1.0), s) for s in t.sigma) +
                              sum(AS.logpdf(AS.MvNormalCholesky(zeros(K), Diagonal(t.sigma) * t.R), y)
                                  for y in data))
        chn = AS.sample(model, AS.NUTS(), 2000; n_warmup = 1000, n_chains = 4,
                        rng = Random.Xoshiro(8))
        @test AS.divergences(chn) == 0
        for name in AS.parameter_names(model)
            @test AS.rhat(chn[name]) < 1.01
        end

        # The posterior is centred on the sample correlation, which at n = 400
        # is a visibly different number from the population correlation. Testing
        # against the population value would be testing the sampler for the
        # sampling error of the data.
        M = reduce(hcat, data)
        Remp = Statistics.cor(M; dims = 2)
        for (name, target) in ((Symbol("R[2,1]"), Remp[2, 1]), (Symbol("R[3,1]"), Remp[3, 1]),
                               (Symbol("R[3,2]"), Remp[3, 2]))
            @test abs(Statistics.mean(vec(chn[name])) - target) < 0.03
        end
        for k in 1:K
            @test Statistics.mean(vec(chn[Symbol("sigma[$k]")])) ≈
                  Statistics.std(view(M, k, :)) rtol = 0.05
        end
    end

    @testset "calibrated against the joint distribution" begin
        # The real check on a transform is calibration, not point recovery.
        K, n = 2, 25
        prior_rand = function (rng)
            return (R = AS.rand(rng, AS.LKJCholesky(K, 2.0)),
                    sigma = [AS.rand(rng, AS.Gamma(2.0, 2.0)) for _ in 1:K])
        end
        simulate = function (theta, rng)
            L = Diagonal(theta.sigma) * theta.R
            return [AS.rand(rng, AS.MvNormalCholesky(zeros(K), L)) for _ in 1:n]
        end
        build = data -> AS.Model((R = AS.corr_cholesky(K), sigma = AS.positive(K)),
                                 t -> AS.logpdf(AS.LKJCholesky(K, 2.0), t.R) +
                                      sum(AS.logpdf(AS.Gamma(2.0, 2.0), s) for s in t.sigma) +
                                      sum(AS.logpdf(AS.MvNormalCholesky(zeros(K),
                                                                        Diagonal(t.sigma) * t.R), y)
                                          for y in data))
        prob = AS.CalibrationProblem(build, prior_rand, simulate)
        res = AS.sbc(Random.Xoshiro(11), prob, AS.NUTS(); n_sims = 150, n_draws = 64,
                     thin = 5, n_warmup = 400)
        @test all(res.pvalue .> 0.01)
        @test res.names == [Symbol("R[2,1]"), Symbol("sigma[1]"), Symbol("sigma[2]")]
    end
end
