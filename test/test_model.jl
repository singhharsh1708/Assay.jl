@testset "model interface" begin
    data = [0.4, 1.2, -0.3, 2.0]
    model = AS.Model((mu = AS.unconstrained(), sigma = AS.positive(), w = AS.simplex(3)),
                     theta -> AS.logpdf(AS.Normal(0.0, 5.0), theta.mu) +
                              AS.logpdf(AS.Gamma(2.0, 1.0), theta.sigma) +
                              AS.logpdf(AS.Dirichlet([2.0, 2.0, 2.0]), theta.w) +
                              AS.loglikelihood(AS.Normal(theta.mu, theta.sigma), data))

    @testset "shape" begin
        @test AS.dimension(model) == 1 + 1 + 2
        @test AS.flat_dimension(model) == 1 + 1 + 3
        @test AS.parameter_names(model) == [:mu, :sigma, Symbol("w[1]"), Symbol("w[2]"), Symbol("w[3]")]
    end

    @testset "constrain / unconstrain round trip" begin
        y = [0.3, -0.7, 0.2, 1.1]
        theta, lj = AS.constrain(model, y)
        @test theta.mu ≈ 0.3
        @test theta.sigma ≈ exp(-0.7)
        @test sum(theta.w) ≈ 1
        @test AS.unconstrain(model, theta) ≈ y rtol = 1e-10
        @test isfinite(lj)
        @test AS.flatten_draw(model, y) ≈ [theta.mu, theta.sigma, theta.w...]
        @test_throws DimensionMismatch AS.logdensity(model, [0.0, 0.0])
    end

    @testset "log density equals hand-written expression" begin
        y = [0.3, -0.7, 0.2, 1.1]
        theta, lj = AS.constrain(model, y)
        manual = AS.logpdf(AS.Normal(0.0, 5.0), theta.mu) +
                 AS.logpdf(AS.Gamma(2.0, 1.0), theta.sigma) +
                 AS.logpdf(AS.Dirichlet([2.0, 2.0, 2.0]), theta.w) +
                 sum(AS.logpdf(AS.Normal(theta.mu, theta.sigma), d) for d in data)
        @test AS.logdensity(model, y; jacobian = false) ≈ manual
        @test AS.logdensity(model, y) ≈ manual + lj
        @test AS.logdensity(model, y) != AS.logdensity(model, y; jacobian = false)
    end

    @testset "gradients agree across backends" begin
        for y in ([0.3, -0.7, 0.2, 1.1], [-1.5, 1.0, -0.4, 0.0], [2.0, 0.1, 1.5, -1.5])
            lp_f, g_f = AS.logdensity_and_gradient(model, y; backend = AS.ForwardDiffAD())
            lp_n, g_n = AS.logdensity_and_gradient(model, y; backend = AS.FiniteDiffAD())
            @test lp_f ≈ lp_n
            @test lp_f ≈ AS.logdensity(model, y)
            @test g_f ≈ g_n rtol = 1e-5
        end
    end

    @testset "gradient of the Jacobian term is not forgotten" begin
        y = [0.3, -0.7, 0.2, 1.1]
        _, g_with = AS.logdensity_and_gradient(model, y)
        _, g_without = AS.logdensity_and_gradient(model, y; jacobian = false)
        @test !isapprox(g_with, g_without; rtol = 1e-6)
    end

    @testset "errors early on a bad specification" begin
        @test_throws ArgumentError AS.Model((mu = 1.0,), theta -> 0.0)
    end
end
