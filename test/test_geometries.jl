# Targets chosen because they are hard in different ways: strong correlation
# (wrong metric), varying scale (the funnel), and curvature (the banana). Every
# assertion here is against a closed form, so "NUTS handles it" is a measurement
# rather than an impression.

"""Neal's funnel: `v ~ N(0, 3)`, `x_i | v ~ N(0, exp(v/2))`. The marginal of `v`
is exactly the prior, which is what makes it a test."""
function funnel_model(k::Int)
    return AS.Model((v = AS.unconstrained(), x = AS.unconstrained(k)),
                    t -> AS.logpdf(AS.Normal(0.0, 3.0), t.v) +
                         sum(AS.logpdf(AS.Normal(0.0, exp(t.v / 2)), xi) for xi in t.x))
end

"""The same funnel written non-centred: `x = exp(v/2) * xtilde` with
`xtilde ~ N(0, 1)`. Identical posterior, completely different geometry."""
function funnel_noncentred(k::Int)
    return AS.Model((v = AS.unconstrained(), xt = AS.unconstrained(k)),
                    t -> AS.logpdf(AS.Normal(0.0, 3.0), t.v) +
                         sum(AS.logpdf(AS.Normal(0.0, 1.0), xi) for xi in t.xt))
end

"""Haario's banana: `x1 ~ N(0, 10^2)`, `x2 | x1 ~ N(b (x1^2 - 100), 1)`.
Marginally `E[x2] = 0` and `Var[x2] = 1 + 2 b^2 100^2`."""
function banana_model(b::Float64)
    return AS.Model((x = AS.unconstrained(2),),
                    t -> AS.logpdf(AS.Normal(0.0, 10.0), t.x[1]) +
                         AS.logpdf(AS.Normal(b * (t.x[1]^2 - 100), 1.0), t.x[2]))
end

@testset "hard geometries" begin
    @testset "strongly correlated Gaussian" begin
        rho = 0.95
        target = AS.MvNormal([0.0, 0.0], [1.0 rho; rho 1.0])
        model = AS.Model((x = AS.unconstrained(2),), t -> AS.logpdf(target, t.x))

        # Regression test: an inconsistent termination rule understates this
        # standard deviation by a few percent while every other diagnostic looks
        # healthy, so all three criteria are checked against the analytic value.
        for crit in (AS.StrictGeneralizedUTurn(), AS.GeneralizedUTurn(), AS.ClassicUTurn())
            chn = AS.sample(model, AS.NUTS(; uturn = crit, metric = :unit, max_treedepth = 8),
                            25_000; n_warmup = 1000, n_chains = 4, rng = Random.Xoshiro(21))
            x1 = chn[Symbol("x[1]")]
            @test check_mean(x1, 0.0; nse = 4)
            @test check_std(x1, 1.0; nse = 4)
            @test cor(vec(chn[Symbol("x[1]")]), vec(chn[Symbol("x[2]")])) ≈ rho atol = 0.01
        end

        # The random walk is the control: with an isotropic proposal it needs
        # far more draws per effective draw than NUTS needs gradients.
        rw = AS.sample(model, AS.RandomWalkMH(), 25_000; n_warmup = 2000, n_chains = 4,
                       rng = Random.Xoshiro(22))
        nuts = AS.sample(model, AS.NUTS(; metric = :dense), 5_000; n_warmup = 1000,
                         n_chains = 4, rng = Random.Xoshiro(23))
        @test AS.ess_bulk(rw[Symbol("x[1]")]) / length(rw[Symbol("x[1]")]) <
              AS.ess_bulk(nuts[Symbol("x[1]")]) / length(nuts[Symbol("x[1]")]) / 5
        # and adapting the proposal covariance recovers most of the gap
        rwa = AS.sample(model, AS.RandomWalkMH(; adapt_cov = true), 25_000; n_warmup = 5000,
                        n_chains = 4, rng = Random.Xoshiro(24))
        @test AS.ess_bulk(rwa[Symbol("x[1]")]) > 3 * AS.ess_bulk(rw[Symbol("x[1]")])
    end

    @testset "Neal's funnel" begin
        centred = AS.sample(funnel_model(9), AS.NUTS(), 4_000; n_warmup = 1000, n_chains = 4,
                            rng = Random.Xoshiro(31))
        # The documented failure: divergences appear and the neck is under-explored.
        @test AS.divergences(centred) > 0
        @test std(vec(centred[:v])) < 2.9                     # true value is 3
        @test any(AS.bfmi(centred) .< 0.9)

        noncentred = AS.sample(funnel_noncentred(9), AS.NUTS(), 4_000; n_warmup = 1000,
                               n_chains = 4, rng = Random.Xoshiro(32))
        # Same posterior, written differently: the pathology is a property of the
        # parameterisation, not of the sampler.
        @test AS.divergences(noncentred) == 0
        @test check_mean(noncentred[:v], 0.0; nse = 4)
        @test check_std(noncentred[:v], 3.0; nse = 4)
        @test AS.ess_bulk(noncentred[:v]) > 5 * AS.ess_bulk(centred[:v])
    end

    @testset "banana" begin
        b = 0.03
        model = banana_model(b)
        sd2 = sqrt(1 + 2 * b^2 * 100^2)

        # At the default target acceptance rate this is a documented failure,
        # not a success: the step size adapted for the flat tails is too large
        # for the curved ridge, so the sampler diverges there and never sees the
        # ends of the banana.
        chn = AS.sample(model, AS.NUTS(), 10_000; n_warmup = 1000, n_chains = 4,
                        rng = Random.Xoshiro(41))
        @test AS.divergences(chn) > 100
        @test std(vec(chn[Symbol("x[2]")])) < 0.9 * sd2

        # A dense metric does not rescue it: the problem is local curvature, and
        # a single global covariance cannot represent that.
        dense = AS.sample(model, AS.NUTS(; metric = :dense), 10_000; n_warmup = 1000,
                          n_chains = 4, rng = Random.Xoshiro(43))
        @test AS.divergences(dense) > 50

        # A smaller step size does: at target 0.95 the same sampler recovers both
        # closed-form moments.
        fine = AS.sample(model, AS.NUTS(; target_accept = 0.95), 10_000; n_warmup = 1000,
                         n_chains = 4, rng = Random.Xoshiro(41))
        @test AS.divergences(fine) < 50
        @test check_mean(fine[Symbol("x[1]")], 0.0; nse = 4)
        @test check_std(fine[Symbol("x[1]")], 10.0; nse = 4)
        @test check_std(fine[Symbol("x[2]")], sd2; nse = 4)

        # And so does removing the curvature by reparameterisation, which is the
        # same lesson as the funnel: geometry is a property of the model as
        # written, not of the algorithm.
        flat = AS.Model((x1 = AS.unconstrained(), z = AS.unconstrained()),
                        t -> AS.logpdf(AS.Normal(0.0, 10.0), t.x1) +
                             AS.logpdf(AS.Normal(0.0, 1.0), t.z))
        rep = AS.sample(flat, AS.NUTS(), 10_000; n_warmup = 1000, n_chains = 4,
                        rng = Random.Xoshiro(44))
        x1 = vec(rep[:x1])
        x2 = vec(rep[:z]) .+ b .* (x1 .^ 2 .- 100)
        @test AS.divergences(rep) == 0
        @test std(x1) ≈ 10.0 rtol = 0.05
        @test std(x2) ≈ sd2 rtol = 0.05

        rw = AS.sample(model, AS.RandomWalkMH(), 10_000; n_warmup = 2000, n_chains = 4,
                       rng = Random.Xoshiro(42))
        @test AS.ess_bulk(rw[Symbol("x[1]")]) < AS.ess_bulk(fine[Symbol("x[1]")]) / 3
    end
end
