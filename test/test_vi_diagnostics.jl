@testset "did the variational approximation work" begin
    rng = Random.Xoshiro(9001)

    @testset "the log density of q agrees with the density it samples from" begin
        # Computed from z rather than from the point, so it is worth checking
        # against the direct evaluation.
        y = 2.0 .+ randn(rng, 30)
        ref = AS.normal_normal(y; mu0 = 0.0, tau0 = 5.0, sigma = 1.0)
        for family in (AS.MeanField(), AS.FullRank())
            res = AS.sample(ref.model, AS.ADVI(; family = family); rng = Random.Xoshiro(1))
            d = AS.dimension(ref.model)
            mu = AS.variational_mean(res)
            sd = AS.variational_scale(res)
            for _ in 1:20
                z = randn(rng, d)
                point = AS.transform_sample(res.family, res.params, z, d)
                direct = sum(AS.logpdf(AS.Normal(mu[i], sd[i]), point[i]) for i in 1:d)
                @test AS.log_variational_density(res.family, res.params, z, d) ≈ direct
            end
        end
    end

    @testset "an approximation that is exact reports the best possible k" begin
        # A normal posterior with one parameter is exactly representable by a
        # mean-field Gaussian, so every importance weight is equal.
        y = 2.0 .+ randn(rng, 40)
        ref = AS.normal_normal(y; mu0 = 0.0, tau0 = 5.0, sigma = 1.0)
        res = AS.sample(ref.model, AS.ADVI(; n_iterations = 5000); rng = Random.Xoshiro(2))
        r = AS.psis_check(res; rng = Random.Xoshiro(3))
        @test r isa AS.PSISResult
        @test AS.reliable(r)
        @test r.k < 0.2
    end

    @testset "k rises with the correlation mean field cannot represent" begin
        # The optimal mean-field Gaussian for a bivariate normal has marginal
        # standard deviation sqrt(1 - rho^2), so at rho = 0.99 it is seven times
        # too narrow and the importance weights have no usable variance. That
        # deficit is already measured elsewhere in the suite; here it is the k
        # that reports it.
        ks = Float64[]
        for rho in (0.0, 0.5, 0.9, 0.99)
            model = AS.Model((x = AS.unconstrained(2),),
                             t -> AS.logpdf(AS.MvNormal(zeros(2), [1.0 rho; rho 1.0]), t.x))
            res = AS.sample(model, AS.ADVI(; n_iterations = 5000); rng = Random.Xoshiro(4))
            push!(ks, AS.psis_check(res; n_samples = 8000, rng = Random.Xoshiro(5)).k)
        end
        @test issorted(ks)                       # worse approximation, larger k
        @test ks[1] < 0.2                        # independent: mean field is exact
        @test ks[end] > 0.7                      # rho = 0.99: not to be trusted

        # and a full-rank family, which can represent the correlation, does not
        # have the problem
        model = AS.Model((x = AS.unconstrained(2),),
                         t -> AS.logpdf(AS.MvNormal(zeros(2), [1.0 0.99; 0.99 1.0]), t.x))
        full = AS.sample(model, AS.ADVI(; family = AS.FullRank(), n_iterations = 8000);
                         rng = Random.Xoshiro(6))
        @test AS.psis_check(full; n_samples = 8000, rng = Random.Xoshiro(7)).k < ks[end]
    end

    @testset "calibration separates a width problem from a location problem" begin
        # Beta-Bernoulli, where the posterior is close to Gaussian on the logit
        # scale, so the approximation is nearly right and should come back
        # centred. The uniformity test and the bias test answer different
        # questions and both are reported.
        prob = AS.conjugate_problem(AS.beta_bernoulli, 30; a = 2.0, b = 2.0)
        r = AS.vsbc(Random.Xoshiro(8), prob, AS.ADVI(; n_iterations = 2000); n_sims = 150)
        @test r.n_sims == 150
        @test size(r.p) == (150, 1)
        @test all(0 .<= r.p .<= 1)
        @test AS.unbiased(r)
        @test occursin("bias", sprint(show, r))
        @test occursin("parameter", sprint(show, r))
        @test_throws ArgumentError AS.vsbc(Random.Xoshiro(9), prob; n_sims = 5)
    end

    @testset "a deliberately shifted approximation is caught as a location problem" begin
        # The negative control. Shifting the prior the model is fitted with,
        # while the data still comes from the original, biases the posterior and
        # the calibration probabilities go with it.
        n = 30
        prior_rand = r -> (mu = AS.rand(r, AS.Normal(0.0, 1.0)),)
        simulate = (theta, r) -> [AS.rand(r, AS.Normal(theta.mu, 1.0)) for _ in 1:n]
        shifted = data -> AS.Model((mu = AS.unconstrained(),),
                                   t -> AS.logpdf(AS.Normal(1.5, 1.0), t.mu) +
                                        AS.loglikelihood(AS.Normal(t.mu, 1.0), data))
        prob = AS.CalibrationProblem(shifted, prior_rand, simulate)
        r = AS.vsbc(Random.Xoshiro(10), prob, AS.ADVI(; n_iterations = 2000); n_sims = 150)
        @test !AS.unbiased(r)
        @test !AS.calibrated(r)
        @test r.bias[1] < -3 * r.bias_se[1]      # the truth sits low in q, consistently
    end
end
