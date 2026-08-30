@testset "cross validation that refits" begin
    rng = Random.Xoshiro(4004)

    # A normal model with known variance stays conjugate when observations are
    # held out, so the exact answer for any fold is available in closed form and
    # the refitting can be checked against it rather than against another
    # approximation.
    sigma, mu0, tau0 = 1.0, 0.0, 5.0
    exact_fold(y, held) = begin
        rest = y[setdiff(eachindex(y), held)]
        prec = 1 / tau0^2 + length(rest) / sigma^2
        m = (mu0 / tau0^2 + sum(rest) / sigma^2) / prec
        sum(AS.logpdf(AS.Normal(m, sqrt(1 / prec + sigma^2)), y[i]) for i in held)
    end

    @testset "folds partition the observations" begin
        for (n, k) in ((30, 5), (30, 30), (17, 4), (11, 2))
            fs = AS.stratified_folds(n, k; rng = Random.Xoshiro(1))
            @test length(fs) == k
            @test sort(reduce(vcat, fs)) == collect(1:n)
            @test all(f -> !isempty(f), fs)
            @test maximum(length, fs) - minimum(length, fs) <= 1
        end
        @test_throws ArgumentError AS.stratified_folds(10, 1)
        @test_throws ArgumentError AS.stratified_folds(10, 11)
    end

    @testset "k-fold agrees with the closed form at every fold" begin
        y = 2.0 .+ sigma .* randn(rng, 24)
        build = train -> AS.normal_normal(y[train]; mu0 = mu0, tau0 = tau0, sigma = sigma).model
        loglik = (theta, i) -> AS.logpdf(AS.Normal(theta.mu, sigma), y[i])

        r = AS.kfold(Random.Xoshiro(2), build, loglik, length(y); k = 6, n_draws = 4000,
                     n_warmup = 1000, n_chains = 2)
        @test r.refits == 6
        @test length(r.pointwise) == length(y)
        @test all(isfinite, r.pointwise)
        @test sum(r.pointwise) ≈ r.elpd
        @test r.se > 0
        @test occursin("6 folds", sprint(show, r))

        exact = sum(exact_fold(y, f) for f in r.folds)
        @test r.elpd ≈ exact atol = 0.1
    end

    @testset "leave-one-out by refitting agrees with exact leave-one-out" begin
        # k equal to n is leave-one-out done the expensive way, which is the
        # thing PSIS is an approximation to. It has to land on the same number.
        y = 2.0 .+ sigma .* randn(rng, 20)
        build = train -> AS.normal_normal(y[train]; mu0 = mu0, tau0 = tau0, sigma = sigma).model
        loglik = (theta, i) -> AS.logpdf(AS.Normal(theta.mu, sigma), y[i])

        r = AS.kfold(Random.Xoshiro(3), build, loglik, length(y); k = length(y),
                     n_draws = 4000, n_warmup = 1000, n_chains = 2)
        exact = sum(exact_fold(y, [i]) for i in eachindex(y))
        @test r.elpd ≈ exact atol = 0.1
        for i in eachindex(y)
            @test r.pointwise[i] ≈ exact_fold(y, [i]) atol = 0.05
        end

        # and against the importance sampling estimate of the same quantity,
        # which is what makes the cheap one worth using when it works
        ref = AS.normal_normal(y; mu0 = mu0, tau0 = tau0, sigma = sigma)
        chn = AS.sample(ref.model, AS.NUTS(), 4000; n_warmup = 1000, n_chains = 4,
                        rng = Random.Xoshiro(4))
        ll = AS.pointwise_log_likelihood(ref.model, chn, loglik; n_obs = length(y))
        psis_r = AS.loo(ll)
        @test isempty(AS.problematic(psis_r))
        @test psis_r.elpd ≈ r.elpd atol = 0.15
    end

    @testset "refitting only what importance sampling could not handle" begin
        # One gross outlier. PSIS reports a high shape for that observation and
        # says its contribution should not be believed; refitting that one
        # observation replaces it with the exact value and leaves the other 29
        # untouched, at the cost of one fit instead of thirty.
        y = vcat(2.0 .+ sigma .* randn(rng, 29), 14.0)
        build = train -> AS.normal_normal(y[train]; mu0 = mu0, tau0 = tau0, sigma = sigma).model
        loglik = (theta, i) -> AS.logpdf(AS.Normal(theta.mu, sigma), y[i])

        ref = AS.normal_normal(y; mu0 = mu0, tau0 = tau0, sigma = sigma)
        chn = AS.sample(ref.model, AS.NUTS(), 4000; n_warmup = 1000, n_chains = 4,
                        rng = Random.Xoshiro(5))
        ll = AS.pointwise_log_likelihood(ref.model, chn, loglik; n_obs = length(y))
        approx = AS.loo(ll)
        @test argmax(approx.khat) == length(y)

        fixed = AS.refit_problematic(Random.Xoshiro(6), approx, build, loglik;
                                     threshold = 0.4, n_draws = 4000, n_warmup = 1000)
        flagged = AS.problematic(approx; threshold = 0.4)
        @test !isempty(flagged)
        @test isempty(AS.problematic(fixed; threshold = 0.4))     # nothing approximated now
        @test all(i -> fixed.khat[i] == 0.0, flagged)

        # The refitted entries match the closed form, and nothing else moved.
        # The tolerance is wider here than elsewhere in this file because the
        # outlier's own contribution is the log of a tiny number, dominated by
        # whichever draws of mu strayed furthest towards it, so it carries much
        # more Monte Carlo error than an ordinary observation does.
        for i in flagged
            @test fixed.pointwise[i] ≈ exact_fold(y, [i]) atol = 0.2
        end
        for i in setdiff(eachindex(y), flagged)
            @test fixed.pointwise[i] == approx.pointwise[i]
        end
        @test sum(fixed.pointwise) ≈ fixed.elpd

        # Both land near the truth, and on this case the approximation happens
        # to land nearer: its bias is smaller than the refit's Monte Carlo
        # error at this number of draws. That is what a high Pareto shape
        # means and does not mean. It is a warning that the estimate cannot be
        # relied on, not a demonstration that it is wrong, and the reason to
        # prefer the refit is that its error shrinks with more draws while the
        # other does not.
        exact = sum(exact_fold(y, [i]) for i in eachindex(y))
        @test abs(fixed.elpd - exact) < 0.3
        @test abs(approx.elpd - exact) < 0.3
    end

    @testset "nothing flagged means nothing refitted" begin
        y = 2.0 .+ sigma .* randn(rng, 20)
        build = train -> AS.normal_normal(y[train]; mu0 = mu0, tau0 = tau0, sigma = sigma).model
        loglik = (theta, i) -> AS.logpdf(AS.Normal(theta.mu, sigma), y[i])
        ref = AS.normal_normal(y; mu0 = mu0, tau0 = tau0, sigma = sigma)
        chn = AS.sample(ref.model, AS.NUTS(), 2000; n_warmup = 1000, n_chains = 2,
                        rng = Random.Xoshiro(7))
        ll = AS.pointwise_log_likelihood(ref.model, chn, loglik; n_obs = length(y))
        approx = AS.loo(ll)
        @test isempty(AS.problematic(approx))
        @test AS.refit_problematic(Random.Xoshiro(8), approx, build, loglik) === approx
    end

    @testset "input validation" begin
        build = train -> AS.normal_normal(randn(Random.Xoshiro(9), max(length(train), 1));
                                          mu0 = mu0, tau0 = tau0, sigma = sigma).model
        loglik = (theta, i) -> 0.0
        @test_throws ArgumentError AS.kfold(Random.Xoshiro(10), build, loglik, 10;
                                            folds = [[1, 2], [2, 3]])
        @test_throws ArgumentError AS.kfold(Random.Xoshiro(11), build, loglik, 10;
                                            folds = Vector{Int}[])
        @test_throws ArgumentError AS.kfold(Random.Xoshiro(12), build, loglik, 3;
                                            folds = [[1, 2, 3]])
    end
end
