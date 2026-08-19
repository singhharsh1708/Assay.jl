using ReverseDiff: ReverseDiff        # loads the package extension

@testset "gradient backends" begin
    data = [0.4, 1.2, -0.3, 2.0]
    model = SB.Model((mu = SB.unconstrained(), sigma = SB.positive(), w = SB.simplex(3)),
                     t -> SB.logpdf(SB.Normal(0.0, 5.0), t.mu) +
                          SB.logpdf(SB.Gamma(2.0, 1.0), t.sigma) +
                          SB.logpdf(SB.Dirichlet([2.0, 2.0, 2.0]), t.w) +
                          SB.loglikelihood(SB.Normal(t.mu, t.sigma), data))

    @testset "every backend gives the same gradient" begin
        for y in ([0.3, -0.7, 0.2, 1.1], [-1.5, 1.0, -0.4, 0.0])
            vf, gf = SB.logdensity_and_gradient(model, y; backend = SB.ForwardDiffAD())
            vr, gr = SB.logdensity_and_gradient(model, y; backend = SB.ReverseDiffAD())
            vc, gc = SB.logdensity_and_gradient(model, y; backend = SB.ReverseDiffAD(; compile = true))
            vn, gn = SB.logdensity_and_gradient(model, y; backend = SB.FiniteDiffAD())
            @test vr ≈ vf && vc ≈ vf && vn ≈ vf
            @test gr ≈ gf
            @test gc ≈ gf
            @test gn ≈ gf rtol = 1e-5
        end
    end

    @testset "a compiled tape is reused rather than rebuilt" begin
        backend = SB.ReverseDiffAD(; compile = true)
        @test backend.tape === nothing
        SB.logdensity_and_gradient(model, [0.3, -0.7, 0.2, 1.1]; backend = backend)
        tape = backend.tape
        @test tape !== nothing
        _, g = SB.logdensity_and_gradient(model, [0.1, 0.4, -0.2, 0.6]; backend = backend)
        @test backend.tape === tape                       # same tape, new point
        _, gf = SB.logdensity_and_gradient(model, [0.1, 0.4, -0.2, 0.6])
        @test g ≈ gf
    end

    @testset "reverse mode samples the same posterior" begin
        rng = Random.Xoshiro(3)
        ref = SB.gamma_poisson([SB.rand(rng, SB.Poisson(4.0)) for _ in 1:40])
        # Kept small: loading ReverseDiff invalidates compiled methods, so every
        # gradient in this testset is paid for at compile time as well as at run
        # time. Two chains of 2000 still give an effective sample size in the
        # thousands, which is what the conjugate check needs.
        chn = SB.sample(ref.model, SB.NUTS(; backend = SB.ReverseDiffAD()), 2000;
                        n_warmup = 1000, n_chains = 2, rng = Random.Xoshiro(2))
        check_conjugate(chn, :lambda, ref.posterior.lambda)
    end
end
