isdefined(@__MODULE__, :funnel_model) ||
    include(joinpath(@__DIR__, "test_geometries_models.jl"))

@testset "reading the diagnostics back as advice" begin
    rng = Random.Xoshiro(2718)

    @testset "a healthy fit produces nothing" begin
        # The check that keeps this from being a horoscope. A well behaved
        # conjugate model has to come back silent, or every finding below is
        # worthless.
        y = 2.0 .+ randn(rng, 40)
        ref = AS.normal_normal(y; mu0 = 0.0, tau0 = 5.0, sigma = 1.0)
        chn = AS.sample(ref.model, AS.NUTS(), 4000; n_warmup = 1000, n_chains = 4,
                        rng = Random.Xoshiro(1))
        d = AS.diagnose(chn)
        @test isempty(d)
        @test AS.healthy(d)
        @test length(d.checks_run) >= 6
        out = sprint(show, d)
        @test occursin("No findings", out)
        # and it does not claim more than it can see
        @test occursin("not that the answer is right", out)
    end

    @testset "the centred funnel is told to reparameterise" begin
        chn = AS.sample(funnel_model(9), AS.NUTS(), 2000; n_warmup = 1000, n_chains = 4,
                        rng = Random.Xoshiro(31))
        d = AS.diagnose(chn)
        @test !AS.healthy(d)
        checks = [f.check for f in d.findings]
        @test :divergences in checks
        text = sprint(show, d)
        @test occursin("target_accept", text)
        @test occursin("non-centred", text)
        # the divergences are located, not just counted: in this model they sit
        # in the neck, where v is small
        div = only(f for f in d.findings if f.check === :divergences)
        @test occursin("concentrated where v is smaller", div.observation)
        # serious findings come first
        @test d.findings[1].severity === :serious

        # the same posterior written non-centred comes back clean, which is the
        # advice being right rather than merely present
        clean = AS.sample(funnel_noncentred(9), AS.NUTS(), 2000; n_warmup = 1000, n_chains = 4,
                          rng = Random.Xoshiro(32))
        @test AS.healthy(AS.diagnose(clean))
    end

    @testset "a handful of divergences is flagged, not excused" begin
        # The banana at target acceptance 0.95 is the case in docs/results.md:
        # between 1 and 22 divergences, a run that looks clean, and a standard
        # deviation of x2 wrong by 5 to 9 percent. A rule that ignored a small
        # count would pass exactly this fit.
        chn = AS.sample(banana_model(0.03), AS.NUTS(; target_accept = 0.95), 4000;
                        n_warmup = 1000, n_chains = 4, rng = Random.Xoshiro(77))
        nd = AS.divergences(chn)
        @test 0 < nd < 0.01 * AS.ndraws(chn) * AS.nchains(chn)   # small, as documented
        d = AS.diagnose(chn)
        f = only(x for x in d.findings if x.check === :divergences)
        @test f.severity === :warning                            # small count, still reported
        @test f.threshold == "any divergence at all"
        @test occursin("not reassurance", f.advice)
        @test occursin("5 to 9 percent", f.advice)
    end

    @testset "too few draws to say anything" begin
        y = randn(rng, 20)
        ref = AS.normal_normal(y; mu0 = 0.0, tau0 = 5.0, sigma = 1.0)
        chn = AS.sample(ref.model, AS.NUTS(), 60; n_warmup = 200, n_chains = 2,
                        rng = Random.Xoshiro(3))
        d = AS.diagnose(chn)
        checks = [f.check for f in d.findings]
        @test :ess_bulk in checks
        f = only(x for x in d.findings if x.check === :ess_bulk)
        @test f.severity === :serious
        @test occursin("R-hat is not reliable", f.advice)
    end

    @testset "a random walk skips the checks it cannot answer" begin
        y = randn(rng, 30)
        ref = AS.normal_normal(y; mu0 = 0.0, tau0 = 5.0, sigma = 1.0)
        chn = AS.sample(ref.model, AS.RandomWalkMH(), 4000; n_warmup = 2000, n_chains = 4,
                        rng = Random.Xoshiro(4))
        d = AS.diagnose(chn)
        # no Hamiltonian, so no divergences, no energy, no tree depth. Those are
        # reported as not checked rather than quietly passed.
        @test :divergences in d.checks_skipped
        @test :bfmi in d.checks_skipped
        @test :treedepth in d.checks_skipped
        @test :rhat in d.checks_run
        @test occursin("Not checked", sprint(show, d))
        @test all(f -> f.check != :divergences, d.findings)
    end

    @testset "a lost chain is the first thing said" begin
        data = randn(rng, 20)
        fragile = AS.Model((mu = AS.unconstrained(),),
                           function (t)
                               5 < t.mu < 8 && throw(DomainError(t.mu, "no"))
                               return AS.logpdf(AS.Normal(0.0, 3.0), t.mu) +
                                      AS.loglikelihood(AS.Normal(t.mu, 1.0), data)
                           end)
        chn = AS.sample(fragile, AS.RandomWalkMH(; scale = 0.1), 500;
                        n_warmup = 300, n_chains = 4, keep_warmup = true,
                        init = [[0.0], [0.0], [0.0], [9.9]], rng = Random.Xoshiro(404))
        d = AS.diagnose(chn)
        @test !AS.healthy(d)
        @test d.findings[1].check === :chain_failures
        @test occursin("not a random subset", d.findings[1].advice)
    end

    @testset "every finding names a number and a threshold" begin
        # The rule that keeps this honest. A finding with no measurement in it
        # is an opinion.
        chn = AS.sample(funnel_model(5), AS.NUTS(), 1000; n_warmup = 500, n_chains = 4,
                        rng = Random.Xoshiro(5))
        d = AS.diagnose(chn)
        @test !isempty(d)
        for f in d.findings
            @test f.severity in (:serious, :warning, :note)
            @test !isempty(f.threshold)
            @test occursin(r"\d", f.observation)          # a number, not an adjective
            @test length(f.advice) > 40
            @test occursin(uppercase(String(f.severity)), sprint(show, f))
        end
    end
end
