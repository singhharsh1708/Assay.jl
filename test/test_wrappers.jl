using Distributions: Distributions

@testset "mixtures, truncation and censoring" begin
    rng = Random.Xoshiro(8080)

    @testset "a mixture agrees with enumerating the assignment" begin
        w = [0.3, 0.45, 0.25]
        comps = [AS.Normal(-2.0, 1.0), AS.Normal(0.5, 2.0), AS.Normal(4.0, 0.7)]
        m = AS.MixtureDensity(w, comps)
        @test AS.n_components(m) == 3
        for x in (-5.0, -1.0, 0.0, 2.5, 7.0)
            direct = log(sum(wi * exp(AS.logpdf(c, x)) for (wi, c) in zip(w, comps)))
            @test AS.logpdf(m, x) ≈ direct
        end
        @test AS.logpdf(m, 0.4) ≈
              Distributions.logpdf(Distributions.MixtureModel(
                                       [Distributions.Normal(-2.0, 1.0),
                                        Distributions.Normal(0.5, 2.0),
                                        Distributions.Normal(4.0, 0.7)], w), 0.4)
    end

    @testset "the case the naive form gets wrong" begin
        # Five hundred standard deviations apart. Every term but one underflows
        # to zero in linear space, so `log(sum(w .* pdf))` returns -Inf and the
        # log-sum-exp form returns the right number.
        m = AS.MixtureDensity([0.5, 0.5], [AS.Normal(0.0, 1.0), AS.Normal(500.0, 1.0)])
        naive(x) = log(sum(0.5 * exp(AS.logpdf(c, x)) for c in m.components))

        # At either component the naive form is fine: the other term underflows
        # to zero and zero is the right thing to add. It breaks between them,
        # where every term underflows and the sum is log of nothing.
        @test naive(500.0) ≈ AS.logpdf(m, 500.0)
        @test naive(250.0) == -Inf                       # the failure, on record
        @test isfinite(AS.logpdf(m, 250.0))
        # 250 is equidistant from both components, so the two halves add back to
        # one whole density rather than to half of one
        @test AS.logpdf(m, 250.0) ≈ AS.logpdf(AS.Normal(0.0, 1.0), 250.0) rtol = 1e-12

        @test AS.logpdf(m, 500.0) ≈ AS.logpdf(AS.Normal(500.0, 1.0), 500.0) + log(0.5)
        @test AS.logpdf(m, 0.0) ≈ AS.logpdf(AS.Normal(0.0, 1.0), 0.0) + log(0.5)
    end

    @testset "a mixture samples the moments it claims" begin
        m = AS.MixtureDensity([0.35, 0.65], [AS.Normal(-3.0, 0.5), AS.Normal(2.0, 1.5)])
        draws = [AS.rand(rng, m) for _ in 1:200_000]
        @test Statistics.mean(draws) ≈ AS.mean(m) atol = 0.02
        # the law of total variance: the spread between the component means is
        # most of it, and averaging the component variances would miss that
        @test Statistics.var(draws) ≈ AS.var(m) rtol = 0.02
        @test AS.var(m) > sum([0.35, 0.65] .* [0.25, 2.25])
        @test AS.cdf(m, AS.mean(m)) ≈ count(<=(AS.mean(m)), draws) / length(draws) atol = 0.01
    end

    @testset "malformed mixtures are refused" begin
        @test_throws DimensionMismatch AS.MixtureDensity([0.5], [AS.Normal(), AS.Normal()])
        @test_throws DomainError AS.MixtureDensity([0.5, 0.7], [AS.Normal(), AS.Normal()])
        @test_throws DomainError AS.MixtureDensity([-0.5, 1.5], [AS.Normal(), AS.Normal()])
        @test_throws ArgumentError AS.MixtureDensity(Float64[], AS.Normal[])
    end

    @testset "a mixture is recovered by sampling" begin
        # Weights inferred on a simplex, component locations known, which is the
        # mixture that stays identified without a label switching argument.
        truth = [0.25, 0.75]
        locs = [-3.0, 3.0]
        y = [AS.rand(rng, AS.MixtureDensity(truth, [AS.Normal(l, 1.0) for l in locs]))
             for _ in 1:400]
        model = AS.Model((w = AS.simplex(2),),
                         t -> AS.logpdf(AS.Dirichlet([1.0, 1.0]), t.w) +
                              AS.loglikelihood(AS.MixtureDensity(t.w,
                                                                 [AS.Normal(l, 1.0) for l in locs]),
                                               y))
        chn = AS.sample(model, AS.NUTS(), 2000; n_warmup = 1000, n_chains = 4,
                        rng = Random.Xoshiro(11))
        @test AS.divergences(chn) == 0
        @test AS.rhat(chn[Symbol("w[1]")]) < 1.01
        # against the exact posterior: with known locations the assignment can
        # be enumerated, so this is a closed-form check rather than a second
        # approximation
        grid = range(0.001, 0.999; length = 2000)
        logpost = [sum(log(p * exp(AS.logpdf(AS.Normal(locs[1], 1.0), yi)) +
                           (1 - p) * exp(AS.logpdf(AS.Normal(locs[2], 1.0), yi))) for yi in y)
                   for p in grid]
        wts = exp.(logpost .- AS.logsumexp(logpost))
        exact_mean = sum(grid .* wts)
        @test Statistics.mean(vec(chn[Symbol("w[1]")])) ≈ exact_mean atol = 0.01
    end

    @testset "a truncated density integrates to one over its support" begin
        for t in (AS.truncated(AS.Normal(0.0, 1.0), -1.5, 2.0),
                  AS.truncated(AS.Normal(1.0, 2.0), 0.0, Inf),
                  AS.truncated(AS.Normal(0.0, 1.0), -Inf, 0.5),
                  AS.truncated(AS.Gamma(2.0, 1.0), 0.5, 4.0),
                  AS.truncated(AS.Exponential(1.5), 1.0, Inf))
            lo, hi = AS.quantile(t, 1e-9), AS.quantile(t, 1 - 1e-9)
            n = 20_000
            xs = range(lo, hi; length = n)
            # trapezoid on the density itself
            ys = [exp(AS.logpdf(t, x)) for x in xs]
            integral = sum((ys[1:(end - 1)] .+ ys[2:end]) ./ 2) * (xs[2] - xs[1])
            @test integral ≈ 1 rtol = 1e-4
            @test AS.logpdf(t, t.lower - 1) == -Inf
            @test AS.logpdf(t, t.upper + 1) == -Inf
        end
    end

    @testset "truncation against Distributions.jl" begin
        for (d, dd, l, u) in ((AS.Normal(0.5, 2.0), Distributions.Normal(0.5, 2.0), -1.0, 3.0),
                              (AS.Normal(0.0, 1.0), Distributions.Normal(0.0, 1.0), 1.0, Inf),
                              (AS.Gamma(3.0, 2.0), Distributions.Gamma(3.0, 0.5), 0.2, 2.0),
                              (AS.Exponential(0.7), Distributions.Exponential(1 / 0.7), 0.5, Inf))
            t = AS.truncated(d, l, u)
            td = Distributions.truncated(dd, l == -Inf ? nothing : l, u == Inf ? nothing : u)
            for x in range(max(l, -4), min(u, 6); length = 9)
                @test AS.logpdf(t, x) ≈ Distributions.logpdf(td, x) rtol = 1e-10
                @test AS.cdf(t, x) ≈ Distributions.cdf(td, x) atol = 1e-10
            end
        end
    end

    @testset "the far tail is where the naive normalisation loses digits" begin
        # Truncation at six standard deviations. The mass is 1e-9, and computing
        # it as cdf(Inf) - cdf(6) is a subtraction of two numbers that agree to
        # nine digits; from the complement it is exact.
        t = AS.truncated(AS.Normal(0.0, 1.0), 6.0, Inf)
        naive_mass = 1 - AS.cdf(AS.Normal(0.0, 1.0), 6.0)
        @test exp(t.logmass) ≈ AS.ccdf(AS.Normal(0.0, 1.0), 6.0) rtol = 1e-12
        @test exp(t.logmass) ≈ naive_mass rtol = 1e-6      # agrees, to fewer digits
        @test AS.logpdf(t, 6.5) ≈
              Distributions.logpdf(Distributions.truncated(Distributions.Normal(), 6.0, nothing),
                                   6.5) rtol = 1e-9
        # and an interval with no mass at all is refused rather than returning
        # infinities later
        @test_throws ArgumentError AS.truncated(AS.Normal(0.0, 1.0), 100.0, 200.0)
        @test_throws ArgumentError AS.truncated(AS.Normal(), 2.0, 1.0)
    end

    @testset "truncated sampling stays inside" begin
        t = AS.truncated(AS.Normal(0.0, 1.0), 0.5, 1.5)
        draws = [AS.rand(rng, t) for _ in 1:50_000]
        @test all(x -> 0.5 <= x <= 1.5, draws)
        @test Statistics.mean(draws) ≈
              Distributions.mean(Distributions.truncated(Distributions.Normal(), 0.5, 1.5)) atol = 0.01
        @test Statistics.quantile(draws, 0.5) ≈ AS.quantile(t, 0.5) atol = 0.02
    end

    @testset "censoring is not truncation" begin
        d = AS.Normal(1.0, 2.0)
        c = AS.censored(d, -Inf, 3.0)
        # below the limit the density is untouched
        @test AS.logpdf(c, 0.0) ≈ AS.logpdf(d, 0.0)
        # at the limit it is a probability, not a density
        @test AS.logpdf(c, 3.0) ≈ log(AS.ccdf(d, 3.0))
        @test AS.logpdf(c, 3.0) ≈ Distributions.logccdf(Distributions.Normal(1.0, 2.0), 3.0)
        @test AS.logpdf(c, 4.0) == -Inf
        # and the two wrappers disagree, which is the point
        t = AS.truncated(d, -Inf, 3.0)
        @test AS.logpdf(t, 0.0) != AS.logpdf(c, 0.0)

        lower = AS.censored(d, -1.0, Inf)
        @test AS.logpdf(lower, -1.0) ≈ log(AS.cdf(d, -1.0))
        @test AS.logpdf(lower, -2.0) == -Inf
        @test_throws ArgumentError AS.censored(d, 3.0, 1.0)

        draws = [AS.rand(rng, c) for _ in 1:20_000]
        @test maximum(draws) == 3.0
        @test count(==(3.0), draws) / length(draws) ≈ AS.ccdf(d, 3.0) atol = 0.01
    end

    @testset "the survival function keeps its digits in the tail" begin
        # log(1 - cdf) against the complement computed directly. By eight
        # standard deviations the subtraction has no digits left at all.
        d = AS.Normal(0.0, 1.0)
        for x in (2.0, 4.0, 6.0, 7.5)
            @test AS.logccdf(d, x) ≈ Distributions.logccdf(Distributions.Normal(), x) rtol = 1e-9
        end
        @test log(1 - AS.cdf(d, 8.5)) == -Inf              # the failure, on record
        @test isfinite(AS.logccdf(d, 8.5))
        @test AS.logccdf(AS.Exponential(2.0), 30.0) ≈ -60.0
    end

    @testset "a censored model is calibrated" begin
        # The check that matters: a wrong normalising constant biases the
        # posterior without raising anything, and only calibration sees it.
        limit = 1.5
        n = 30
        prior_rand = rng -> (mu = AS.rand(rng, AS.Normal(0.0, 1.0)),)
        simulate = (theta, rng) -> [min(AS.rand(rng, AS.Normal(theta.mu, 1.0)), limit)
                                    for _ in 1:n]
        build = data -> AS.Model((mu = AS.unconstrained(),),
                                 t -> AS.logpdf(AS.Normal(0.0, 1.0), t.mu) +
                                      AS.loglikelihood(AS.censored(AS.Normal(t.mu, 1.0),
                                                                   -Inf, limit), data))
        prob = AS.CalibrationProblem(build, prior_rand, simulate)
        r = AS.sbc(Random.Xoshiro(21), prob, AS.NUTS(); n_sims = 128, n_draws = 255,
                   n_warmup = 400)
        @test AS.calibrated(r)

        # and the same model with the censoring ignored is not, which is what
        # says the check has teeth
        wrong = data -> AS.Model((mu = AS.unconstrained(),),
                                 t -> AS.logpdf(AS.Normal(0.0, 1.0), t.mu) +
                                      AS.loglikelihood(AS.Normal(t.mu, 1.0), data))
        rw = AS.sbc(Random.Xoshiro(22), AS.CalibrationProblem(wrong, prior_rand, simulate),
                    AS.NUTS(); n_sims = 128, n_draws = 255, n_warmup = 400)
        @test !AS.calibrated(rw)
    end
end
