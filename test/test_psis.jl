@testset "Pareto smoothed importance sampling" begin
    rng = Random.Xoshiro(31337)

    @testset "the generalised Pareto fit recovers a known shape" begin
        for (ktrue, sigma) in ((0.3, 1.0), (0.7, 2.0), (-0.2, 1.0), (0.0, 1.5))
            u = rand(rng, 4000)
            x = sort([AS.gpd_quantile(ui, ktrue, sigma) for ui in u])
            k, s = AS.gpd_fit(x)
            @test k ≈ ktrue atol = 0.06
            @test s ≈ sigma rtol = 0.12
        end
        @test_throws ArgumentError AS.gpd_fit([1.0, 2.0])
        @test_throws DomainError AS.gpd_fit([-1.0, 1.0, 2.0, 3.0, 4.0])
    end

    @testset "the Pareto quantile function" begin
        # k -> 0 is the exponential limit
        @test AS.gpd_quantile(0.5, 0.0, 2.0) ≈ -2.0 * log(0.5)
        @test AS.gpd_quantile(0.5, 1e-14, 2.0) ≈ -2.0 * log(0.5) rtol = 1e-6
        # monotone increasing in p, and zero at zero
        qs = [AS.gpd_quantile(p, 0.4, 1.0) for p in 0.05:0.05:0.95]
        @test issorted(qs)
        @test AS.gpd_quantile(0.0, 0.4, 1.0) ≈ 0 atol = 1e-12
    end

    @testset "k rises as the proposal gets worse" begin
        ks = Float64[]
        for propsd in (2.0, 1.5, 0.8, 0.5)
            draws = randn(rng, 20_000) .* propsd
            logr = [AS.logpdf(AS.Normal(0.0, 1.0), d) - AS.logpdf(AS.Normal(0.0, propsd), d)
                    for d in draws]
            r = AS.psis(logr)
            push!(ks, r.k)
            @test sum(exp.(r.log_weights)) ≈ 1        # weights are normalised
            @test length(r.log_weights) == length(logr)
        end
        @test issorted(ks)                            # worse proposal, larger k
        @test ks[1] < 0.5                             # a wider proposal is safe
        @test ks[end] > 0.4                           # a much narrower one is not
    end

    @testset "identical proposal and target is the best case, not a failure" begin
        # Every weight equal is what perfect importance sampling looks like. An
        # implementation that reports "not to be trusted" here is flagging the
        # one situation that never needs flagging.
        logr = zeros(5000)
        r = AS.psis(logr)
        @test r.k == -Inf
        @test AS.reliable(r)
        @test all(w -> w ≈ -log(5000), r.log_weights)
    end

    @testset "degenerate input is reported rather than smoothed over" begin
        r = AS.psis(fill(-Inf, 100))
        @test !isfinite(r.k)
        r = AS.psis(randn(rng, 8))                    # too few draws to fit a tail
        @test r.k == Inf
        @test !AS.reliable(r)
        @test sum(exp.(r.log_weights)) ≈ 1
    end

    @testset "leave-one-out against exact leave-one-out" begin
        # For a normal model with known variance the leave-one-out posterior is
        # still conjugate, so the exact answer is available and PSIS-LOO can be
        # checked against it rather than against another approximation.
        sigma, mu0, tau0 = 1.0, 0.0, 5.0
        y = 2.0 .+ sigma .* randn(rng, 30)

        exact = 0.0
        for i in eachindex(y)
            rest = y[setdiff(eachindex(y), i)]
            prec = 1 / tau0^2 + length(rest) / sigma^2
            m = (mu0 / tau0^2 + sum(rest) / sigma^2) / prec
            exact += AS.logpdf(AS.Normal(m, sqrt(1 / prec + sigma^2)), y[i])
        end

        ref = AS.normal_normal(y; mu0 = mu0, tau0 = tau0, sigma = sigma)
        chn = AS.sample(ref.model, AS.NUTS(), 4000; n_warmup = 1000, n_chains = 4,
                        rng = Random.Xoshiro(22))
        draws = vec(chn[:mu])
        ll = [AS.logpdf(AS.Normal(d, sigma), y[i]) for d in draws, i in eachindex(y)]

        r = AS.loo(ll)
        @test r.elpd ≈ exact atol = 0.15
        @test length(r.pointwise) == length(y)
        @test sum(r.pointwise) ≈ r.elpd
        @test isempty(AS.problematic(r))
        # one parameter is being estimated, so the effective number of them is one
        @test 0.5 < r.p_loo < 1.5
        @test r.se > 0

        w = AS.waic(ll)
        @test w.elpd ≈ r.elpd atol = 0.1        # they agree when nothing is influential
        @test w.p_waic ≈ r.p_loo atol = 0.2
    end

    @testset "an influential observation raises its own k" begin
        sigma = 1.0
        y = vcat(2.0 .+ sigma .* randn(rng, 29), 12.0)      # one gross outlier
        ref = AS.normal_normal(y; mu0 = 0.0, tau0 = 5.0, sigma = sigma)
        chn = AS.sample(ref.model, AS.NUTS(), 4000; n_warmup = 1000, n_chains = 4,
                        rng = Random.Xoshiro(32))
        draws = vec(chn[:mu])
        ll = [AS.logpdf(AS.Normal(d, sigma), y[i]) for d in draws, i in eachindex(y)]
        r = AS.loo(ll)
        # the diagnostic is local: the outlier's own k is the largest of the
        # set, and far above the typical observation. Comparing against the
        # maximum of the others would be comparing against noise, since which
        # ordinary point happens to sit second is seed dependent.
        @test argmax(r.khat) == length(y)
        @test r.khat[end] > 0.4
        @test r.khat[end] > 5 * abs(Statistics.median(r.khat[1:(end - 1)]))
        @test r.pointwise[end] < minimum(r.pointwise[1:(end - 1)])
        # and the effective number of parameters inflates, which is the classic
        # signal that one point is doing the work
        @test r.p_loo > 2
    end

    @testset "model comparison" begin
        sigma = 1.0
        y = 2.0 .+ sigma .* randn(rng, 40)
        ref = AS.normal_normal(y; mu0 = 0.0, tau0 = 5.0, sigma = sigma)
        chn = AS.sample(ref.model, AS.NUTS(), 4000; n_warmup = 1000, n_chains = 4,
                        rng = Random.Xoshiro(33))
        draws = vec(chn[:mu])
        right = AS.loo([AS.logpdf(AS.Normal(d, sigma), y[i]) for d in draws, i in eachindex(y)])
        wrong = AS.loo([AS.logpdf(AS.Normal(d, 4.0), y[i]) for d in draws, i in eachindex(y)])

        cmp = AS.loo_compare(right, wrong)
        @test cmp[1].model == 1                       # the correct scale wins
        @test cmp[1].delta == 0
        @test cmp[2].delta < 0
        # and decisively: the difference is many standard errors of the difference
        @test abs(cmp[2].delta) > 4 * cmp[2].se_delta
        @test_throws ArgumentError AS.loo_compare(right)
        @test_throws DimensionMismatch AS.loo_compare(right, AS.loo(randn(rng, 500, 7)))
    end

    @testset "input validation" begin
        @test_throws ArgumentError AS.loo(randn(rng, 5, 3))
    end
end
