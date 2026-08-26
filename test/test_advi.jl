@testset "ADVI" begin
    rng = Random.Xoshiro(4242)

    @testset "entropy and sampling match the family they claim" begin
        d = 3
        params = [0.5, -1.0, 2.0, log(0.3), log(1.5), log(0.7)]
        @test AS.entropy(AS.MeanField(), params, d) ≈
              sum(log, [0.3, 1.5, 0.7]) + d / 2 * (1 + log(2pi))
        zs = [randn(Random.Xoshiro(i), d) for i in 1:20_000]
        xs = reduce(hcat, AS.transform_sample(AS.MeanField(), params, z, d) for z in zs)
        @test vec(mean(xs; dims = 2)) ≈ params[1:d] atol = 0.05
        @test vec(std(xs; dims = 2)) ≈ [0.3, 1.5, 0.7] rtol = 0.05
        # full rank: covariance must be L L'
        L = [0.8 0.0; -0.5 1.2]
        fp = [0.0, 0.0, log(L[1, 1]), L[2, 1], log(L[2, 2])]
        xs = reduce(hcat, AS.transform_sample(AS.FullRank(), fp, randn(Random.Xoshiro(i), 2), 2)
                    for i in 1:40_000)
        @test cov(xs; dims = 2) ≈ L * L' atol = 0.05
        @test AS.entropy(AS.FullRank(), fp, 2) ≈ log(det(L)) + 1 * (1 + log(2pi))
    end

    @testset "the ELBO is a lower bound on the log evidence" begin
        data = [rand(rng) < 0.3 ? 1 : 0 for _ in 1:80]
        ref = AS.beta_bernoulli(data; a = 2.0, b = 2.0)
        res = AS.sample(ref.model, AS.ADVI(); rng = Random.Xoshiro(5))
        # The bound is the definition of the ELBO, so a violation beyond Monte
        # Carlo error means the entropy term or the Jacobian is wrong.
        @test res.elbo_final <= ref.logevidence + 0.05
        @test res.elbo_final > ref.logevidence - 1.0
        @test res.elbo_trace[end] > res.elbo_trace[1]
        @test res.converged
    end

    @testset "conjugate posteriors are approximately recovered" begin
        data = [AS.rand(rng, AS.Poisson(4.0)) for _ in 1:50]
        ref = AS.gamma_poisson(data; a = 2.0, b = 1.0)
        res = AS.sample(ref.model, AS.ADVI(; n_samples = 8); rng = Random.Xoshiro(6))
        chn = AS.posterior_samples(res, 40_000; rng = Random.Xoshiro(7))
        exact = ref.posterior.lambda
        # ADVI is a Gaussian in the unconstrained space, so the constrained mean
        # is biased even at the optimum; 5% is the honest tolerance here, and
        # the bias is reported in docs/results.md rather than tuned away.
        @test mean(vec(chn[:lambda])) ≈ AS.mean(exact) rtol = 0.05
        @test std(vec(chn[:lambda])) ≈ sqrt(AS.var(exact)) rtol = 0.15
    end

    @testset "mean field understates correlated variance by the known factor" begin
        rho = 0.9
        Sigma = [1.0 rho; rho 1.0]
        target = AS.MvNormal([0.0, 0.0], Sigma)
        model = AS.Model((x = AS.unconstrained(2),), theta -> AS.logpdf(target, theta.x))
        res = AS.sample(model, AS.ADVI(; family = AS.MeanField(), n_samples = 16, step_size = 0.02);
                        rng = Random.Xoshiro(8))
        # The optimal mean-field Gaussian for a bivariate normal has marginal
        # standard deviation sqrt(1 - rho^2); this is the closed form the
        # approximation is checked against, not a vague "it is too narrow".
        @test AS.variational_scale(res) ≈ fill(sqrt(1 - rho^2), 2) rtol = 0.1
    end

    @testset "full rank recovers the correlation" begin
        rho = 0.9
        Sigma = [1.0 rho; rho 1.0]
        target = AS.MvNormal([0.0, 0.0], Sigma)
        model = AS.Model((x = AS.unconstrained(2),), theta -> AS.logpdf(target, theta.x))
        res = AS.sample(model, AS.ADVI(; family = AS.FullRank(), n_samples = 16, step_size = 0.02);
                        rng = Random.Xoshiro(9))
        L = AS.variational_factor(res)
        @test L * L' ≈ Sigma atol = 0.15
        chn = AS.posterior_samples(res, 20_000; rng = Random.Xoshiro(10))
        @test cor(vec(chn[Symbol("x[1]")]), vec(chn[Symbol("x[2]")])) ≈ rho atol = 0.05
    end

    @testset "the step size decays, and that is what makes the default work" begin
        # A constant step size has no good default. These four targets each
        # prefer a different one, and the wrong choice is not slightly worse but
        # catastrophically so. Decay removes most of that sensitivity, which is
        # why it is on by default and why the candidate search Stan performs was
        # implemented, measured and then removed.
        cases = Tuple{String,Any,Vector{Float64}}[]
        push!(cases, ("badly scaled",
                      AS.Model((x = AS.unconstrained(2),),
                               t -> AS.logpdf(AS.MvNormal(zeros(2),
                                                          Matrix(Diagonal([1e-3, 1e3] .^ 2))), t.x)),
                      [1e-3, 1e3]))
        push!(cases, ("correlated",
                      AS.Model((x = AS.unconstrained(2),),
                               t -> AS.logpdf(AS.MvNormal(zeros(2), [1.0 0.99; 0.99 1.0]), t.x)),
                      fill(sqrt(1 - 0.99^2), 2)))
        obs = randn(Random.Xoshiro(5), 5000)
        push!(cases, ("many observations",
                      AS.Model((mu = AS.unconstrained(), sigma = AS.positive()),
                               t -> AS.logpdf(AS.Normal(0.0, 10.0), t.mu) +
                                    AS.logpdf(AS.Gamma(2.0, 1.0), t.sigma) +
                                    AS.loglikelihood(AS.Normal(t.mu, t.sigma), obs)),
                      [1 / sqrt(5000), 1 / sqrt(2 * 5000)]))

        for (name, model, truth) in cases
            res = AS.sample(model, AS.ADVI(; n_samples = 8); rng = Random.Xoshiro(1))
            err = maximum(abs.(AS.variational_scale(res) .- truth) ./ truth)
            @test err < 0.1
        end

        # and the decay is actually applied: turning it off changes the answer
        model = cases[end][2]
        with_decay = AS.sample(model, AS.ADVI(; n_samples = 8); rng = Random.Xoshiro(1))
        without = AS.sample(model, AS.ADVI(; n_samples = 8, decay = 0.0, step_size = 0.05);
                            rng = Random.Xoshiro(1))
        truth = cases[end][3]
        err_with = maximum(abs.(AS.variational_scale(with_decay) .- truth) ./ truth)
        err_without = maximum(abs.(AS.variational_scale(without) .- truth) ./ truth)
        @test err_with < err_without
    end

    @testset "constrained parameters stay in their support" begin
        data = [rand(rng) < 0.2 ? 1 : 0 for _ in 1:40]
        ref = AS.beta_bernoulli(data; a = 1.0, b = 1.0)
        res = AS.sample(ref.model, AS.ADVI(); rng = Random.Xoshiro(11))
        chn = AS.posterior_samples(res, 5_000; rng = Random.Xoshiro(12))
        @test all(0 .< vec(chn[:p]) .< 1)
    end
end
