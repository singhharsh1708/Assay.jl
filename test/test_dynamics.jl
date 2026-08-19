@testset "dynamical system on the simplex" begin
    @testset "the map keeps the state on the simplex" begin
        rng = Random.Xoshiro(1)
        for _ in 1:50
            x = SB.rand(rng, SB.Dirichlet(5, 1.0))
            f = randn(rng, 5) .* 3
            xn = SB.simplex_step(x, f)
            @test sum(xn) ≈ 1
            @test all(>(0), xn)
        end
        # extreme fitness values must not produce NaN through overflow
        x = fill(0.25, 4)
        @test sum(SB.simplex_step(x, [800.0, -800.0, 0.0, 5.0])) ≈ 1
        @test all(isfinite, SB.simplex_step(x, [800.0, -800.0, 0.0, 5.0]))
    end

    @testset "known behaviour of the replicator map" begin
        x0 = [0.5, 0.3, 0.2]
        # zero fitness differences: a fixed point
        @test SB.simplex_step(x0, zeros(3)) ≈ x0
        # shift invariance, which is why one component is pinned in the model
        @test SB.simplex_step(x0, [1.0, 0.5, -0.2]) ≈ SB.simplex_step(x0, [1.0, 0.5, -0.2] .+ 7.3)
        # selection: the fittest component takes over
        traj = SB.simplex_trajectory(x0, [0.0, 1.0, 0.5], 200)
        @test traj[1] ≈ x0
        @test length(traj) == 200
        @test traj[end][2] > 0.999
        # exact solution: after T steps x_T proportional to x_0 .* exp(T f)
        T = 12
        f = [0.0, 0.3, -0.4]
        expected = x0 .* exp.((T - 1) .* f)
        expected ./= sum(expected)
        @test SB.simplex_trajectory(x0, f, T)[end] ≈ expected rtol = 1e-10
    end

    @testset "posterior recovers the fitness vector" begin
        rng = Random.Xoshiro(5)
        truth = (x0 = [0.6, 0.3, 0.1], f = [0.8, -0.5])
        states = SB.simplex_trajectory(truth.x0, vcat(0.0, truth.f), 6)
        counts = [SB.rand(rng, SB.Multinomial(200, x)) for x in states]
        chn = SB.sample(SB.replicator_model(counts, 200), SB.NUTS(), 4000;
                        n_warmup = 1000, n_chains = 4, rng = Random.Xoshiro(6))
        for (name, t) in ((Symbol("f[1]"), 0.8), (Symbol("f[2]"), -0.5),
                          (Symbol("x0[1]"), 0.6))
            v = vec(chn[name])
            @test quantile(v, 0.005) < t < quantile(v, 0.995)
        end
        @test SB.divergences(chn) == 0
        @test all(SB.rhat(chn[n]) < 1.01 for n in SB.parameter_names(SB.replicator_model(counts, 200)))
        # the simplex parameter is a simplex in every draw
        sums = sum(chn.value[:, 1:3, :]; dims = 2)
        @test all(x -> isapprox(x, 1; atol = 1e-10), sums)
    end

    @testset "calibrated against the joint distribution" begin
        # No closed form exists for this posterior, so simulation based
        # calibration is the verification: it exercises the simplex transform,
        # its Jacobian and the sampler together.
        prob = SB.replicator_problem(3, 6, 40; alpha = 1.0, fitness_scale = 1.0)
        res = SB.sbc(Random.Xoshiro(11), prob, SB.NUTS(); n_sims = 150, n_draws = 64,
                     thin = 5, n_warmup = 400)
        @test all(res.pvalue .> 0.01)
        @test res.names == [Symbol("x0[1]"), Symbol("x0[2]"), Symbol("x0[3]"),
                            Symbol("f[1]"), Symbol("f[2]")]
    end
end
