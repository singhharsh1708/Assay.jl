@testset "working with a fitted posterior" begin
    rng = Random.Xoshiro(5150)

    @testset "a log density that is not a function of its argument is refused" begin
        # Data drawn inside the closure is the classic version of this. Without
        # a check the symptom is an acceptance rate near zero and an effective
        # sample size of two, which reads like a badly tuned sampler rather
        # than a broken model.
        r = Random.Xoshiro(1)
        bad = AS.Model((mu = AS.unconstrained(),),
                       t -> AS.loglikelihood(AS.Normal(t.mu, 1.0), randn(r, 30)))
        err = try
            AS.sample(bad, AS.NUTS(), 50; n_warmup = 20, rng = Random.Xoshiro(2))
            nothing
        catch e
            e
        end
        @test err !== nothing
        @test occursin("not deterministic", sprint(showerror, err))

        data = randn(rng, 30)
        good = AS.Model((mu = AS.unconstrained(),),
                        t -> AS.loglikelihood(AS.Normal(t.mu, 1.0), data))
        @test AS.check_model(good, [0.0]) === good
        @test_throws ArgumentError AS.check_model(good, [NaN])
    end

    @testset "parameters reconstruct exactly from a chain" begin
        data = randn(rng, 30)
        model = AS.Model((mu = AS.unconstrained(), sigma = AS.positive(), w = AS.simplex(3)),
                         t -> AS.logpdf(AS.Normal(0.0, 5.0), t.mu) +
                              AS.logpdf(AS.Gamma(2.0, 1.0), t.sigma) +
                              AS.logpdf(AS.Dirichlet([2.0, 2.0, 2.0]), t.w) +
                              AS.loglikelihood(AS.Normal(t.mu, t.sigma), data))
        chn = AS.sample(model, AS.NUTS(), 500; n_warmup = 500, n_chains = 2,
                        rng = Random.Xoshiro(11))
        @test AS.has_unconstrained(chn)
        @test AS.n_parameter_draws(chn) == 1000
        @test AS.n_parameter_draws(chn; thin = 10) == 100

        thetas = collect(AS.parameter_draws(model, chn))
        @test length(thetas) == 1000
        for theta in thetas[1:50]
            @test theta.sigma > 0
            @test sum(theta.w) ≈ 1                    # the simplex constraint survives
            @test all(>(0), theta.w)
        end
        # the scalar columns of the chain agree with the reconstruction
        @test thetas[1].mu ≈ chn.value[1, 1, 1]
        @test thetas[1].sigma ≈ chn.value[1, 2, 1]

        # a chain built without its unconstrained draws says so rather than
        # silently reconstructing something wrong
        bare = AS.Chains(chn.value, chn.names, chn.stats, chn.info)
        @test !AS.has_unconstrained(bare)
        @test_throws ArgumentError AS.parameter_draws(model, bare)
    end

    @testset "posterior predictive draws and checks" begin
        n, sigma = 40, 1.0
        y = 2.0 .+ sigma .* randn(rng, n)
        ref = AS.normal_normal(y; mu0 = 0.0, tau0 = 5.0, sigma = sigma)
        chn = AS.sample(ref.model, AS.NUTS(), 2000; n_warmup = 1000, n_chains = 4,
                        rng = Random.Xoshiro(4))

        yrep = AS.predictive(ref.model, chn,
                             (t, r) -> [AS.rand(r, AS.Normal(t.mu, sigma)) for _ in 1:n];
                             rng = Random.Xoshiro(5), thin = 8)
        @test length(yrep) == AS.n_parameter_draws(chn; thin = 8)
        @test all(r -> length(r) == n, yrep)
        @test eltype(yrep) <: AbstractVector

        # a correctly specified model reproduces the data it was fitted to
        for stat in (Statistics.mean, Statistics.std, maximum)
            chk = AS.predictive_check(y, yrep, stat)
            @test 0.05 < chk.pvalue < 0.95
            @test length(chk.replicated) == length(yrep)
            @test chk.observed ≈ stat(y)
        end
    end

    @testset "a predictive check catches a model that cannot fit the tails" begin
        # Heavy-tailed data under a normal model. The mean is reproduced and
        # everything about the spread is not, which is the argument for
        # choosing the statistic deliberately: a check on the mean alone would
        # have passed this.
        n, sigma = 50, 1.0
        y = 2.0 .+ [AS.rand(rng, AS.StudentT(2.0, 0.0, 1.0)) for _ in 1:n]
        ref = AS.normal_normal(y; mu0 = 0.0, tau0 = 5.0, sigma = sigma)
        chn = AS.sample(ref.model, AS.NUTS(), 2000; n_warmup = 1000, n_chains = 4,
                        rng = Random.Xoshiro(78))
        yrep = AS.predictive(ref.model, chn,
                             (t, r) -> [AS.rand(r, AS.Normal(t.mu, sigma)) for _ in 1:n];
                             rng = Random.Xoshiro(79), thin = 8)
        @test 0.05 < AS.predictive_check(y, yrep, Statistics.mean).pvalue < 0.95
        @test AS.predictive_check(y, yrep, Statistics.std).pvalue < 0.01
        @test AS.predictive_check(y, yrep, maximum).pvalue < 0.01
    end

    @testset "pointwise log likelihood feeds cross validation" begin
        n, sigma = 30, 1.0
        y = 1.0 .+ sigma .* randn(rng, n)
        ref = AS.normal_normal(y; mu0 = 0.0, tau0 = 5.0, sigma = sigma)
        chn = AS.sample(ref.model, AS.NUTS(), 1000; n_warmup = 500, n_chains = 2,
                        rng = Random.Xoshiro(12))
        ll = AS.pointwise_log_likelihood(ref.model, chn,
                                         (t, i) -> AS.logpdf(AS.Normal(t.mu, sigma), y[i]);
                                         n_obs = n, thin = 2)
        @test size(ll) == (AS.n_parameter_draws(chn; thin = 2), n)
        # the same matrix built by hand
        manual = reduce(hcat, [[AS.logpdf(AS.Normal(t.mu, sigma), y[i]) for i in 1:n]
                               for t in AS.parameter_draws(ref.model, chn; thin = 2)])'
        @test ll ≈ manual
        r = AS.loo(ll)
        @test isfinite(r.elpd)
        @test isempty(AS.problematic(r))
        @test_throws ArgumentError AS.pointwise_log_likelihood(ref.model, chn, (t, i) -> 0.0;
                                                              n_obs = 0)
    end

    @testset "subsetting a chain" begin
        data = randn(rng, 20)
        model = AS.Model((mu = AS.unconstrained(), sigma = AS.positive()),
                         t -> AS.logpdf(AS.Normal(0.0, 5.0), t.mu) +
                              AS.logpdf(AS.Gamma(2.0, 1.0), t.sigma) +
                              AS.loglikelihood(AS.Normal(t.mu, t.sigma), data))
        chn = AS.sample(model, AS.NUTS(), 1000; n_warmup = 500, n_chains = 3,
                        rng = Random.Xoshiro(13))

        by_param = AS.subset(chn; params = [:sigma])
        @test by_param.names == [:sigma]
        @test AS.nparams(by_param) == 1
        @test vec(by_param[:sigma]) == vec(chn[:sigma])

        by_draw = AS.subset(chn; draws = 501:1000, chains = 1:2)
        @test AS.ndraws(by_draw) == 500
        @test AS.nchains(by_draw) == 2
        @test AS.has_unconstrained(by_draw)
        @test by_draw.value[1, 1, 1] == chn.value[501, 1, 1]
        @test haskey(by_draw.stats, :accept_prob)
        @test size(by_draw.stats[:accept_prob]) == (500, 2)

        @test_throws KeyError AS.subset(chn; params = [:not_a_parameter])

        # summarising a named subset is the common case for a wide model
        s = AS.summarize(chn, [:mu])
        @test s.names == [:mu]
        @test length(s.mean) == 1
    end
end
