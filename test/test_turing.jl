# Cross-check against Turing.jl.
#
# Turing is a test dependency only: nothing in `src/` knows it exists. It is
# used here as an independent oracle on models that have no closed form, which
# is the one thing the conjugate comparisons cannot cover. Where a closed form
# does exist, both implementations are checked against it rather than against
# each other, so agreement on a shared error is not mistaken for correctness.
using Turing: Turing
using MCMCChains: MCMCChains

"""
    agree(x, y; nse = 4)

Do two `draws x chains` matrices have the same mean, to within the combined
Monte Carlo standard error of the two estimates?
"""
function agree(x::AbstractMatrix, y::AbstractMatrix; nse::Real = 4)
    dx = mean(vec(x)) - mean(vec(y))
    se = sqrt(SB.mcse_mean(x)^2 + SB.mcse_mean(y)^2)
    ok = abs(dx) <= nse * se
    ok || @info "cross-check mismatch" ours=mean(vec(x)) turing=mean(vec(y)) se=se z=dx / se
    return ok
end

Turing.@model function turing_bernoulli(y)
    p ~ Turing.Beta(2, 2)
    for i in eachindex(y)
        y[i] ~ Turing.Bernoulli(p)
    end
end

Turing.@model function turing_regression(x, y)
    a ~ Turing.Normal(0, 5)
    b ~ Turing.Normal(0, 5)
    sigma ~ Turing.Exponential(1.0)
    for i in eachindex(y)
        y[i] ~ Turing.Normal(a + b * x[i], sigma)
    end
end

Turing.@model function turing_schools(y, s)
    mu ~ Turing.Normal(0, 5)
    tau ~ Turing.truncated(Turing.Cauchy(0, 5); lower = 0)
    theta_raw ~ Turing.filldist(Turing.Normal(0, 1), length(y))
    for i in eachindex(y)
        y[i] ~ Turing.Normal(mu + tau * theta_raw[i], s[i])
    end
end

turing_run(model, seed; n = 2000, chains = 4) =
    Turing.sample(Random.Xoshiro(seed), model, Turing.NUTS(1000, 0.8), Turing.MCMCThreads(),
                  n, chains; progress = false, chain_type = MCMCChains.Chains)

@testset "cross-check against Turing" begin
    rng = Random.Xoshiro(99)

    @testset "Beta-Bernoulli: both against the closed form, then against each other" begin
        data = [rand(rng) < 0.35 ? 1 : 0 for _ in 1:60]
        ref = SB.beta_bernoulli(data; a = 2.0, b = 2.0)
        ours = SB.sample(ref.model, SB.NUTS(), 2000; n_warmup = 1000, n_chains = 4,
                         rng = Random.Xoshiro(1))
        theirs = Array(turing_run(turing_bernoulli(data), 1)[:p])
        @test check_mean(ours[:p], SB.mean(ref.posterior.p))
        @test check_mean(theirs, SB.mean(ref.posterior.p))
        @test agree(ours[:p], theirs)
        @test std(vec(ours[:p])) ≈ std(vec(theirs)) rtol = 0.05
    end

    @testset "linear regression with a positive scale" begin
        x = randn(rng, 40)
        y = 1.0 .+ 2.0 .* x .+ 0.5 .* randn(rng, 40)
        model = SB.Model((a = SB.unconstrained(), b = SB.unconstrained(), sigma = SB.positive()),
                         t -> SB.logpdf(SB.Normal(0.0, 5.0), t.a) +
                              SB.logpdf(SB.Normal(0.0, 5.0), t.b) +
                              SB.logpdf(SB.Exponential(1.0), t.sigma) +
                              sum(SB.logpdf(SB.Normal(t.a + t.b * x[i], t.sigma), y[i])
                                  for i in eachindex(y)))
        ours = SB.sample(model, SB.NUTS(), 2000; n_warmup = 1000, n_chains = 4,
                         rng = Random.Xoshiro(2))
        theirs = turing_run(turing_regression(x, y), 2)
        for name in (:a, :b, :sigma)
            @test agree(ours[name], Array(theirs[name]))
            @test std(vec(ours[name])) ≈ std(vec(Array(theirs[name]))) rtol = 0.1
        end
        # the scale parameter is where a missing Jacobian would show up first
        @test all(vec(ours[:sigma]) .> 0)
    end

    @testset "eight schools, non-centred" begin
        y = [28.0, 8, -3, 7, -1, 1, 18, 12]
        s = [15.0, 10, 16, 11, 9, 11, 10, 18]
        model = SB.Model((mu = SB.unconstrained(), tau = SB.positive(),
                          theta_raw = SB.unconstrained(8)),
                         t -> SB.logpdf(SB.Normal(0.0, 5.0), t.mu) +
                              SB.logpdf(SB.Cauchy(0.0, 5.0), t.tau) + log(2) +
                              sum(SB.logpdf(SB.Normal(0.0, 1.0), r) for r in t.theta_raw) +
                              sum(SB.logpdf(SB.Normal(t.mu + t.tau * t.theta_raw[i], s[i]), y[i])
                                  for i in eachindex(y)))
        ours = SB.sample(model, SB.NUTS(), 4000; n_warmup = 1000, n_chains = 4,
                         rng = Random.Xoshiro(3))
        theirs = turing_run(turing_schools(y, s), 3; n = 4000)
        @test agree(ours[:mu], Array(theirs[:mu]); nse = 4)
        @test agree(ours[:tau], Array(theirs[:tau]); nse = 4)
        @test std(vec(ours[:tau])) ≈ std(vec(Array(theirs[:tau]))) rtol = 0.15
        for i in 1:3
            @test agree(ours[Symbol("theta_raw[$i]")], Array(theirs[Symbol("theta_raw[$i]")]))
        end
    end

    @testset "our diagnostics agree with MCMCChains" begin
        data = [rand(rng) < 0.35 ? 1 : 0 for _ in 1:60]
        chn = turing_run(turing_bernoulli(data), 4)
        x = Array(chn[:p])
        their_ess = MCMCChains.ess(chn)[:p][1]
        their_rhat = MCMCChains.rhat(chn)[:p][1]
        # Same estimators (rank-normalised split R-hat, Geyer-truncated ESS),
        # independently implemented: they should land within a few percent.
        @test SB.ess_bulk(x) ≈ their_ess rtol = 0.1
        @test SB.rhat(x) ≈ their_rhat rtol = 0.02
    end
end
