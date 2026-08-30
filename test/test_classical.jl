using MCMCDiagnosticTools: MCMCDiagnosticTools

@testset "the classical convergence diagnostics" begin
    rng = Random.Xoshiro(1234)

    # One set of chains used throughout, from a model with a known answer, so
    # that a diagnostic disagreeing with the reference can be attributed.
    y = 2.0 .+ randn(rng, 40)
    ref = AS.normal_normal(y; mu0 = 0.0, tau0 = 5.0, sigma = 1.0)
    good = AS.sample(ref.model, AS.NUTS(), 2000; n_warmup = 1000, n_chains = 4,
                     rng = Random.Xoshiro(1))[:mu]
    # A deliberately bad set: a short random walk with a tiny step, started far
    # apart, which has not converged and is known not to have.
    bad = AS.sample(ref.model, AS.RandomWalkMH(; scale = 0.005, adapt_scale = false), 2000;
                    n_warmup = 10, n_chains = 4, init = [[-8.0], [-3.0], [3.0], [8.0]],
                    rng = Random.Xoshiro(2))[:mu]

    @testset "Gelman-Rubin against MCMCDiagnosticTools" begin
        # Their gelmandiag wants draws x params x chains
        for x in (good, bad)
            # their layout is draws x chains x parameters
            theirs = MCMCDiagnosticTools.gelmandiag(reshape(x, size(x, 1), size(x, 2), 1))
            mine = AS.gelman_rubin(x)
            @test mine.psrf ≈ only(theirs.psrf) rtol = 1e-8
            @test mine.upper >= mine.psrf
        end
        @test AS.gelman_rubin(good).psrf < 1.01
        @test AS.gelman_rubin(bad).psrf > 1.5

        # and it is not rank-normalised split R-hat, which is the point of the
        # docstring: on a converged chain they agree, and they need not
        @test AS.gelman_rubin(good).psrf ≈ AS.rhat(good) atol = 0.02
        @test_throws ArgumentError AS.gelman_rubin(reshape(good[:, 1], :, 1))
    end

    @testset "Gelman-Rubin is blind to a variance difference" begin
        # Four chains with the same mean and wildly different spread. The plain
        # potential scale reduction factor is a ratio of variances that this
        # leaves alone, and reports 1.000. Rank-normalised split R-hat sees it
        # at 1.32. Gelman-Rubin lands in between, at 1.20, and not because the
        # ratio moved: its degrees of freedom correction reacts to the spread
        # of the within-chain variances, so it half sees the problem for a
        # reason unrelated to what the problem is.
        x = hcat(randn(rng, 2000), randn(rng, 2000) .* 3, randn(rng, 2000) .* 0.3,
                 randn(rng, 2000))
        @test AS.rhat_plain(x) ≈ 1.0 atol = 0.01
        @test AS.rhat(x) > 1.3
        @test 1.1 < AS.gelman_rubin(x).psrf < 1.3
    end

    @testset "Geweke's 1992 z score" begin
        for x in (good, bad)
            theirs = [MCMCDiagnosticTools.gewekediag(collect(view(x, :, j))).zscore
                      for j in 1:size(x, 2)]
            mine = AS.geweke_z(x)
            @test length(mine) == size(x, 2)
            # Both compare the same two windows; they differ in how the spectral
            # density at zero is estimated, theirs by an autoregressive fit and
            # this one from the effective sample size, so the scores agree in
            # sign and magnitude rather than to many digits.
            @test all(sign.(mine) .== sign.(theirs))
            @test all(abs.(mine .- theirs) .< 0.6 .* max.(abs.(theirs), 1.0))
        end
        @test maximum(abs, AS.geweke_z(good)) < 3
        @test maximum(abs, AS.geweke_z(bad)) > 3
        @test_throws ArgumentError AS.geweke_z(view(good, :, 1); first = 0.6, last = 0.6)
    end

    @testset "the two Geweke tests are different tests" begin
        # The naming collision, asserted rather than left in a docstring. The
        # 1992 diagnostic compares one chain against itself over time; the 2004
        # test compares a sampler against the model that generated its data.
        prob = AS.conjugate_problem(AS.beta_bernoulli, 12; a = 2.0, b = 2.0)
        joint = AS.geweke(Random.Xoshiro(3), prob, AS.NUTS(); n_marginal = 300,
                          n_successive = 300)
        @test joint isa AS.GewekeResult
        @test AS.geweke_z(view(good, :, 1)) isa Float64
    end

    @testset "Heidelberger-Welch against MCMCDiagnosticTools" begin
        # Both halves of this test rest on an estimate of the spectral density
        # at zero and the two packages estimate it differently, so comparing the
        # implementations directly would only measure that choice. Given the
        # same estimate, the window search, the Cramer-von Mises statistic and
        # the halfwidth all have to agree exactly, and that is what is checked.
        theirs_s0 = y -> length(y) *
                         first(MCMCDiagnosticTools.mcse(reshape(collect(y), :, 1, 1);
                                                        split_chains = 1))^2
        for x in (good, bad)
            for j in 1:size(x, 2)
                chain = collect(view(x, :, j))
                theirs = MCMCDiagnosticTools.heideldiag(chain)
                mine = AS.heidelberger_welch(chain; spectrum0 = theirs_s0)
                @test mine.stationary == only(theirs.stationarity)
                @test mine.halfwidth_passed == only(theirs.test)
                @test mine.burn_in == only(theirs.burnin)
                @test mine.mean ≈ only(theirs.mean) rtol = 1e-10
                @test mine.pvalue ≈ only(theirs.pvalue) rtol = 1e-8
                @test mine.halfwidth ≈ only(theirs.halfwidth) rtol = 1e-10
            end
        end

        # And with the default autoregressive estimate, the verdicts on chains
        # that are not borderline still come out right.
        for j in 1:size(good, 2)
            @test AS.heidelberger_welch(collect(view(good, :, j))).stationary
        end

        r = AS.heidelberger_welch(collect(view(good, :, 1)))
        @test 0 <= r.pvalue <= 1
        @test r.halfwidth > 0
        @test occursin("stationary", sprint(show, r))
        @test_throws ArgumentError AS.heidelberger_welch(randn(rng, 10))
    end

    @testset "the halfwidth test is the mcse question in other clothing" begin
        # It passes when the Monte Carlo error is small relative to the mean,
        # which is what mcse_mean reports directly and without a threshold.
        chain = collect(view(good, :, 1))
        r = AS.heidelberger_welch(chain; eps = 0.1)
        se = sqrt(AS.spectrum0_ar(chain) / length(chain))
        @test r.halfwidth ≈ 1.959963984540054 * se rtol = 1e-6
        # and on a converged chain that is the same number mcse_mean reports,
        # which is the point: the halfwidth test asks the mcse question
        @test se ≈ AS.mcse_mean(reshape(chain, :, 1)) rtol = 0.15
        @test r.halfwidth_passed == (r.halfwidth <= 0.1 * abs(r.mean))
    end

    @testset "the spectral density at zero, two ways" begin
        # An AR(1) with coefficient phi has spectral density at zero equal to
        # sigma^2 / (1 - phi)^2, which is the closed form both estimators are
        # aiming at.
        for phi in (0.0, 0.5, 0.8)
            e = randn(Random.Xoshiro(77), 40_000)
            z = similar(e)
            z[1] = e[1] / sqrt(1 - phi^2)
            for t in 2:length(z)
                z[t] = phi * z[t - 1] + e[t]
            end
            exact = 1 / (1 - phi)^2
            @test AS.spectrum0_ar(z) ≈ exact rtol = 0.15
            @test length(z) * AS.mcse_mean(reshape(z, :, 1))^2 ≈ exact rtol = 0.2
        end
    end

    @testset "Raftery-Lewis against MCMCDiagnosticTools" begin
        # r = 0.01 rather than the 0.005 default, so that 2000 draws clears the
        # minimum of 937 and the reference returns a number instead of NaN
        chain = collect(view(good, :, 1))
        theirs = MCMCDiagnosticTools.rafterydiag(chain; r = 0.01)
        mine = AS.raftery_lewis(chain; r = 0.01)
        @test mine.minimum == Int(only(theirs.nmin))
        @test mine.total > mine.burn_in
        # burn-in and total depend on the thinning search, which stops at the
        # first lag where a first-order model beats a second-order one on BIC,
        # so they agree in order of magnitude rather than exactly
        @test mine.dependence ≈ only(theirs.dependencefactor) rtol = 0.5
        @test occursin("dependence factor", sprint(show, mine))
        @test_throws DomainError AS.raftery_lewis(chain; q = 1.5)
        @test_throws ArgumentError AS.raftery_lewis(randn(rng, 50))
    end

    @testset "a chain that never crosses the quantile is refused" begin
        # The stuck chain goes below the 2.5% quantile and never comes back, so
        # one of the two transition probabilities is zero. The formula then
        # reports a dependence factor below one, which would read as "better
        # than independent draws" for the worst chain in this file.
        e = try
            AS.raftery_lewis(collect(view(bad, :, 1)); r = 0.01)
        catch e
            e
        end
        @test e isa ArgumentError
        @test occursin("only one direction", sprint(showerror, e))
    end

    @testset "the Cramer-von Mises distribution" begin
        # Monotone, and bracketed by the published critical values: the
        # asymptotic 5% point of the statistic is 0.461 and the 1% point 0.743.
        @test AS._cvm_pvalue(0.0) == 1.0
        ps = [AS._cvm_pvalue(w) for w in 0.05:0.05:1.5]
        @test issorted(ps; rev = true)
        @test AS._cvm_pvalue(0.461) ≈ 0.05 atol = 0.01
        @test AS._cvm_pvalue(0.743) ≈ 0.01 atol = 0.005
    end

    @testset "why none of these is the one to reach for" begin
        # The comparison the docstring claims, made here rather than asserted.
        # On chains that have not converged, every diagnostic fires; the
        # difference is what they see on chains that look fine and are not.
        @test AS.gelman_rubin(bad).psrf > 1.5
        @test maximum(abs, AS.geweke_z(bad)) > 3
        @test AS.rhat(bad) > 1.5
        @test AS.ess_bulk(bad) < 50
    end
end
