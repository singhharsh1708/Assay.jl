@testset "modes and the Laplace approximation" begin
    rng = Random.Xoshiro(1337)

    @testset "the two modes are different points, and both are exact" begin
        # A Beta(a, b) posterior has its constrained mode at (a-1)/(a+b-2). Push
        # it through the logit and the Jacobian p(1-p) shifts the maximum to
        # a/(a+b), which is the posterior mean. Both closed forms, so both
        # answers can be checked rather than compared to each other.
        data = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1]
        a0, b0 = 2.0, 2.0
        ref = AS.beta_bernoulli(data; a = a0, b = b0)
        a = a0 + sum(data)
        b = b0 + length(data) - sum(data)

        constrained = AS.find_mode(ref.model; space = :constrained, rng = Random.Xoshiro(1))
        @test constrained.converged
        @test constrained.theta.p ≈ (a - 1) / (a + b - 2) rtol = 1e-8
        @test constrained.space === :constrained

        unconstrained = AS.find_mode(ref.model; rng = Random.Xoshiro(1))
        @test unconstrained.converged
        @test unconstrained.theta.p ≈ a / (a + b) rtol = 1e-8
        @test unconstrained.space === :unconstrained

        # not the same point, which is the entire reason `space` exists
        @test !isapprox(constrained.theta.p, unconstrained.theta.p; rtol = 1e-3)
        @test occursin("unconstrained space", sprint(show, unconstrained))
        @test_throws ArgumentError AS.find_mode(ref.model; space = :whatever)
    end

    @testset "the same split on a positive parameter" begin
        # Gamma(alpha, rate) posterior: mode at (alpha-1)/rate on the constrained
        # scale, and at alpha/rate, the mean, through the log transform.
        y = [AS.rand(rng, AS.Poisson(4.0)) for _ in 1:25]
        a0, b0 = 2.0, 1.0
        ref = AS.gamma_poisson(y; a = a0, b = b0)
        alpha = a0 + sum(y)
        rate = b0 + length(y)
        @test AS.find_mode(ref.model; space = :constrained,
                           rng = Random.Xoshiro(2)).theta.lambda ≈ (alpha - 1) / rate rtol = 1e-8
        @test AS.find_mode(ref.model; rng = Random.Xoshiro(2)).theta.lambda ≈ alpha / rate rtol = 1e-8
    end

    @testset "a mode the optimiser has to work for" begin
        # Ten correlated dimensions, condition number 100. The mode is the mean
        # of the normal, known exactly, and the identity-Hessian first step is
        # nowhere near it.
        d = 10
        A = randn(rng, d, d)
        Sigma = A * A' + d * I
        mu = randn(rng, d) .* 3
        model = AS.Model((x = AS.unconstrained(d),),
                         t -> AS.logpdf(AS.MvNormal(mu, Sigma), t.x))
        r = AS.find_mode(model; init = zeros(d))
        @test r.converged
        @test r.theta.x ≈ mu rtol = 1e-6
        @test r.gradient_norm < 1e-8
        @test r.iterations < 100
    end

    @testset "the Laplace covariance is the posterior covariance when it should be" begin
        # A Gaussian posterior is its own Laplace approximation, so this is an
        # equality rather than a tolerance, and the importance weights are all
        # equal, which psis reports as the best possible k rather than as a
        # failure.
        y = 2.0 .+ randn(rng, 40)
        ref = AS.normal_normal(y; mu0 = 0.0, tau0 = 5.0, sigma = 1.0)
        prec = 1 / 25 + length(y)
        r = AS.laplace(ref.model; rng = Random.Xoshiro(3))
        @test r.mode.theta.mu ≈ sum(y) / prec rtol = 1e-8
        @test only(r.covariance) ≈ 1 / prec rtol = 1e-8
        @test r.khat == -Inf                       # exact, so every weight is equal
        @test occursin("good", sprint(show, r))

        # and the Laplace evidence is the exact marginal likelihood here, since
        # a Gaussian integral is what the Laplace estimate assumes it is doing
        @test r.log_evidence ≈ ref.logevidence rtol = 1e-8
    end

    @testset "the covariance in more than one dimension" begin
        d = 4
        A = randn(rng, d, d)
        Sigma = A * A' + d * I
        mu = randn(rng, d)
        model = AS.Model((x = AS.unconstrained(d),),
                         t -> AS.logpdf(AS.MvNormal(mu, Sigma), t.x))
        r = AS.laplace(model; init = zeros(d), rng = Random.Xoshiro(4))
        @test r.covariance ≈ Sigma rtol = 1e-6
        @test r.factor * r.factor' ≈ Sigma rtol = 1e-6
        # Not exactly -Inf here, unlike the one-dimensional case above: the
        # Hessian is inverted numerically, so the weights differ at rounding
        # level rather than not at all, and the fit is to that noise.
        @test 0 <= r.khat < 0.1

        # the draws it produces are ordinary chains
        chn = AS.posterior_samples(r, 40_000; rng = Random.Xoshiro(5))
        @test AS.nchains(chn) == 1
        @test Statistics.cov(reshape(chn.value, 40_000, d); dims = 1) ≈ Sigma rtol = 0.05
        @test isfinite(AS.ess_bulk(chn[Symbol("x[1]")]))
    end

    @testset "the approximation reports when it is bad" begin
        # A skewed posterior is not Gaussian on any scale, and the Pareto shape
        # of the importance weights says so. Without that number the covariance
        # would look just as authoritative as it does above.
        y = [AS.rand(rng, AS.Exponential(0.5)) for _ in 1:8]
        model = AS.Model((rate = AS.positive(),),
                         t -> AS.logpdf(AS.Gamma(1.1, 0.1), t.rate) +
                              sum(AS.logpdf(AS.Exponential(t.rate), yi) for yi in y))
        r = AS.laplace(model; n_check = 4000, rng = Random.Xoshiro(6))
        @test r.mode.converged
        @test isfinite(r.khat)
        @test r.khat > -Inf                        # not exact, unlike the Gaussian case

        # the good case and the bad case are distinguished by the number, which
        # is the point of reporting it
        gauss = AS.Model((x = AS.unconstrained(),), t -> AS.logpdf(AS.Normal(0.0, 1.0), t.x))
        @test AS.laplace(gauss; rng = Random.Xoshiro(7)).khat < r.khat
    end

    @testset "a point that is not a maximum is refused" begin
        # The Hessian at a saddle is indefinite, so there is no Gaussian, and
        # inverting it anyway would produce a covariance with negative variances.
        saddle = AS.Model((x = AS.unconstrained(2),), t -> t.x[1]^2 - t.x[2]^2)
        m = AS.ModeResult(zeros(2), (x = zeros(2),), 0.0, 0.0, 1, true, :unconstrained)
        @test_throws ArgumentError AS.laplace(saddle; mode = m, n_check = 0)

        # and a constrained-space mode is refused, because it is the maximum of
        # a different function from the one being approximated
        ref = AS.beta_bernoulli([1, 0, 1]; a = 2.0, b = 2.0)
        cm = AS.find_mode(ref.model; space = :constrained, rng = Random.Xoshiro(8))
        @test_throws ArgumentError AS.laplace(ref.model; mode = cm, n_check = 0)
    end

    @testset "the mode agrees with where the sampler went" begin
        # The check the mode is most useful for: a cheap answer that says
        # whether an expensive one landed in the right place.
        y = randn(rng, 60) .* 2 .+ 1
        model = AS.Model((mu = AS.unconstrained(), sigma = AS.positive()),
                         t -> AS.logpdf(AS.Normal(0.0, 10.0), t.mu) +
                              AS.logpdf(AS.Gamma(2.0, 1.0), t.sigma) +
                              AS.loglikelihood(AS.Normal(t.mu, t.sigma), y))
        r = AS.laplace(model; rng = Random.Xoshiro(9))
        chn = AS.sample(model, AS.NUTS(), 4000; n_warmup = 1000, n_chains = 4,
                        rng = Random.Xoshiro(10))
        @test r.mode.theta.mu ≈ Statistics.mean(vec(chn[:mu])) atol = 0.1
        # the Laplace standard deviation is in the right ballpark on the
        # unconstrained scale, which is where the approximation is made
        @test sqrt(r.covariance[1, 1]) ≈ Statistics.std(vec(chn[:mu])) rtol = 0.15
        @test r.khat < 0.7
    end
end
