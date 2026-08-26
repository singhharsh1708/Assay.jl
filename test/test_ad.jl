using ReverseDiff: ReverseDiff        # each of these loads a package extension
using Zygote: Zygote
using Enzyme: Enzyme

@testset "gradient backends" begin
    data = [0.4, 1.2, -0.3, 2.0]
    model = AS.Model((mu = AS.unconstrained(), sigma = AS.positive(), w = AS.simplex(3)),
                     t -> AS.logpdf(AS.Normal(0.0, 5.0), t.mu) +
                          AS.logpdf(AS.Gamma(2.0, 1.0), t.sigma) +
                          AS.logpdf(AS.Dirichlet([2.0, 2.0, 2.0]), t.w) +
                          AS.loglikelihood(AS.Normal(t.mu, t.sigma), data))

    @testset "every backend gives the same gradient" begin
        for y in ([0.3, -0.7, 0.2, 1.1], [-1.5, 1.0, -0.4, 0.0])
            vf, gf = AS.logdensity_and_gradient(model, y; backend = AS.ForwardDiffAD())
            vr, gr = AS.logdensity_and_gradient(model, y; backend = AS.ReverseDiffAD())
            vc, gc = AS.logdensity_and_gradient(model, y; backend = AS.ReverseDiffAD(; compile = true))
            vn, gn = AS.logdensity_and_gradient(model, y; backend = AS.FiniteDiffAD())
            @test vr ≈ vf && vc ≈ vf && vn ≈ vf
            @test gr ≈ gf
            @test gc ≈ gf
            @test gn ≈ gf rtol = 1e-5
        end
    end

    @testset "a compiled tape is reused rather than rebuilt" begin
        backend = AS.ReverseDiffAD(; compile = true)
        @test backend.tape === nothing
        AS.logdensity_and_gradient(model, [0.3, -0.7, 0.2, 1.1]; backend = backend)
        tape = backend.tape
        @test tape !== nothing
        _, g = AS.logdensity_and_gradient(model, [0.1, 0.4, -0.2, 0.6]; backend = backend)
        @test backend.tape === tape                       # same tape, new point
        _, gf = AS.logdensity_and_gradient(model, [0.1, 0.4, -0.2, 0.6])
        @test g ≈ gf
    end

    @testset "Enzyme handles every transform, including the mutating ones" begin
        for y in ([0.3, -0.7, 0.2, 1.1], [-1.5, 1.0, -0.4, 0.0])
            vf, gf = AS.logdensity_and_gradient(model, y; backend = AS.ForwardDiffAD())
            ve, ge = AS.logdensity_and_gradient(model, y; backend = AS.EnzymeAD())
            @test ve ≈ vf
            @test ge ≈ gf rtol = 1e-8
        end
    end

    @testset "Zygote works where nothing is mutated, and says so where it is" begin
        # The elementwise transforms are broadcast, so Zygote differentiates
        # them. simplex, ordered and corr_cholesky build their output by
        # writing into an array, which Zygote cannot do.
        data = randn(Random.Xoshiro(4), 20)
        plain = AS.Model((mu = AS.unconstrained(), sigma = AS.positive()),
                         t -> AS.logpdf(AS.Normal(0.0, 5.0), t.mu) +
                              AS.logpdf(AS.Gamma(2.0, 1.0), t.sigma) +
                              AS.loglikelihood(AS.Normal(t.mu, t.sigma), data))
        y = [0.2, -0.3]
        vf, gf = AS.logdensity_and_gradient(plain, y)
        vz, gz = AS.logdensity_and_gradient(plain, y; backend = AS.ZygoteAD())
        @test vz ≈ vf
        @test gz ≈ gf rtol = 1e-8

        # and the model with a simplex fails with an explanation rather than a
        # stack trace about setindex!
        err = try
            AS.logdensity_and_gradient(model, [0.3, -0.7, 0.2, 1.1]; backend = AS.ZygoteAD())
            nothing
        catch e
            e
        end
        @test err !== nothing
        msg = sprint(showerror, err)
        @test occursin("mutates arrays", msg)
        @test occursin("EnzymeAD", msg)
    end

    @testset "reverse mode samples the same posterior" begin
        rng = Random.Xoshiro(3)
        ref = AS.gamma_poisson([AS.rand(rng, AS.Poisson(4.0)) for _ in 1:40])
        # Kept small: loading ReverseDiff invalidates compiled methods, so every
        # gradient in this testset is paid for at compile time as well as at run
        # time. Two chains of 2000 still give an effective sample size in the
        # thousands, which is what the conjugate check needs.
        chn = AS.sample(ref.model, AS.NUTS(; backend = AS.ReverseDiffAD()), 2000;
                        n_warmup = 1000, n_chains = 2, rng = Random.Xoshiro(2))
        check_conjugate(chn, :lambda, ref.posterior.lambda)
    end
end
