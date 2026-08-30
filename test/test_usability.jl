# A backend no extension will ever define, for the fallback path.
struct UnbackedAD <: AS.ADBackend end

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
        @test err isa AS.NonDeterministicModelError
        @test occursin("not deterministic", sprint(showerror, err))
        # the position is carried, not only described
        @test length(err.position) == 1
        @test err.first != err.second

        data = randn(rng, 30)
        good = AS.Model((mu = AS.unconstrained(),),
                        t -> AS.loglikelihood(AS.Normal(t.mu, 1.0), data))
        @test AS.check_model(good, [0.0]) === good
        @test_throws AS.NonFiniteDensityError AS.check_model(good, [NaN])
    end

    @testset "the failures a caller can act on have types" begin
        # Matching on message text is what these replace. Each one carries the
        # position, which is the part that makes the failure reproducible by
        # hand rather than only describable.
        data = randn(rng, 20)
        good = AS.Model((mu = AS.unconstrained(),),
                        t -> AS.loglikelihood(AS.Normal(t.mu, 1.0), data))

        e = try AS.check_model(good, [NaN]) catch e; e end
        @test e isa AS.NonFiniteDensityError
        @test e isa AS.AssayError
        @test e.what === :logdensity
        @test isnan(e.value)
        @test isnan(e.position[1])
        @test occursin("Position:", sprint(showerror, e))

        # a density that is finite nowhere: the search reports what it tried
        nowhere = AS.Model((mu = AS.unconstrained(),), t -> -Inf)
        e = try AS.random_init(Random.Xoshiro(1), nowhere; tries = 7) catch e; e end
        @test e isa AS.InitialisationError
        @test e.tries == 7
        @test e.dimension == 1
        @test e.last_value == -Inf
        @test occursin("no point with a finite log density found in 7", sprint(showerror, e))

        # A finite log density with a gradient that is not: the value at zero is
        # fine and the slope there is not, which is exactly the case a check on
        # the density alone lets through. The coordinate at fault is named.
        cusp = AS.Model((mu = AS.unconstrained(), nu = AS.unconstrained()),
                        t -> -sqrt(abs(t.mu)) + AS.logpdf(AS.Normal(0.0, 1.0), t.nu))
        @test isfinite(AS.logdensity(cusp, [0.0, 0.5]))
        e = try AS.sample(cusp, AS.NUTS(), 10; n_warmup = 5, init = [0.0, 0.5],
                          rng = Random.Xoshiro(2)) catch e; e end
        @test e isa AS.NonFiniteDensityError
        @test e.what === :gradient
        @test e.coordinate == 1
        @test occursin("coordinate 1 of the gradient", sprint(showerror, e))

        # a backend with nothing behind it, which does not depend on whether an
        # extension happens to be loaded by the time this file runs
        e = try AS.logdensity_and_gradient(UnbackedAD(), sum, [1.0]) catch e; e end
        @test e isa AS.BackendUnavailableError
        @test e.package === nothing
        @test occursin("no gradient method", sprint(showerror, e))
        @test occursin("using ReverseDiff",
                       sprint(showerror, AS.BackendUnavailableError(AS.ReverseDiffAD, :ReverseDiff)))
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

    @testset "one chain dying does not take the run with it" begin
        # A log density that throws for some parameter values, which is the
        # ordinary case: an index out of bounds, a log of something that just
        # went negative, a solver that did not converge. Chain four starts in
        # the region that throws and the other three never reach it.
        data = randn(rng, 20)
        # The band sits between chain four's starting point and the posterior,
        # so that chain has to walk into it and dies partway through warmup
        # with draws already stored. The other three start on the far side of
        # it and stay there.
        fragile = AS.Model((mu = AS.unconstrained(),),
                           function (t)
                               5 < t.mu < 8 && throw(DomainError(t.mu, "outside the table"))
                               return AS.logpdf(AS.Normal(0.0, 3.0), t.mu) +
                                      AS.loglikelihood(AS.Normal(t.mu, 1.0), data)
                           end)

        chn = AS.sample(fragile, AS.RandomWalkMH(; scale = 0.1), 200;
                        n_warmup = 300, n_chains = 4, keep_warmup = true,
                        init = [[0.0], [0.0], [0.0], [9.9]], rng = Random.Xoshiro(404))

        @test AS.failed(chn)
        @test AS.nchains(chn) == 3                       # the survivors came back
        @test length(AS.failures(chn)) == 1
        @test chn.info[:n_chains_requested] == 4

        f = only(AS.failures(chn))
        @test f.chain == 4
        @test f.error isa DomainError
        @test f.phase === :warmup
        @test f.iteration > 0

        # what it managed before dying is kept, not discarded
        @test size(f.value, 1) > 0
        @test size(f.value, 1) == size(f.unconstrained, 1)
        @test all(isfinite, f.value)

        # and the position it died at is carried, so the failure reproduces
        @test length(f.last_position) == 1
        @test isfinite(only(f.last_position))
        @test only(f.last_position) > 8              # it died on the way in
        @test_throws DomainError AS.logdensity(fragile, [6.5])
        # the draws it kept are the ones it took before that, in order
        @test f.unconstrained[end, 1] < 10
        @test size(f.value, 1) <= f.iteration

        # the surviving chains are ordinary chains: every diagnostic still works
        @test all(isfinite, vec(chn[:mu]))
        @test isfinite(AS.rhat(chn[:mu]))
        @test AS.ndraws(chn) == 500                      # warmup kept
        @test size(chn.stats[:accept_prob]) == (500, 3)
        @test occursin("ChainFailure(chain 4", sprint(show, f))

        # a run where every chain dies has no partial result worth returning,
        # so the error itself is what comes back
        e = try AS.sample(fragile, AS.RandomWalkMH(), 50; n_warmup = 20, n_chains = 2,
                          init = [6.5], rng = Random.Xoshiro(5)) catch e; e end
        @test e isa DomainError

        # and a clean run says so, without a failures field getting in the way
        good = AS.sample(fragile, AS.RandomWalkMH(), 200; n_warmup = 200, n_chains = 2,
                         rng = Random.Xoshiro(6))
        @test !AS.failed(good)
        @test isempty(AS.failures(good))
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
