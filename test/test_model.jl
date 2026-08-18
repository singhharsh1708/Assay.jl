@testset "model interface" begin
    data = [0.4, 1.2, -0.3, 2.0]
    model = SB.Model((mu = SB.unconstrained(), sigma = SB.positive(), w = SB.simplex(3)),
                     theta -> SB.logpdf(SB.Normal(0.0, 5.0), theta.mu) +
                              SB.logpdf(SB.Gamma(2.0, 1.0), theta.sigma) +
                              SB.logpdf(SB.Dirichlet([2.0, 2.0, 2.0]), theta.w) +
                              SB.loglikelihood(SB.Normal(theta.mu, theta.sigma), data))

    @testset "shape" begin
        @test SB.dimension(model) == 1 + 1 + 2
        @test SB.flat_dimension(model) == 1 + 1 + 3
        @test SB.parameter_names(model) == [:mu, :sigma, Symbol("w[1]"), Symbol("w[2]"), Symbol("w[3]")]
    end

    @testset "constrain / unconstrain round trip" begin
        y = [0.3, -0.7, 0.2, 1.1]
        theta, lj = SB.constrain(model, y)
        @test theta.mu ≈ 0.3
        @test theta.sigma ≈ exp(-0.7)
        @test sum(theta.w) ≈ 1
        @test SB.unconstrain(model, theta) ≈ y rtol = 1e-10
        @test isfinite(lj)
        @test SB.flatten_draw(model, y) ≈ [theta.mu, theta.sigma, theta.w...]
        @test_throws DimensionMismatch SB.logdensity(model, [0.0, 0.0])
    end

    @testset "log density equals hand-written expression" begin
        y = [0.3, -0.7, 0.2, 1.1]
        theta, lj = SB.constrain(model, y)
        manual = SB.logpdf(SB.Normal(0.0, 5.0), theta.mu) +
                 SB.logpdf(SB.Gamma(2.0, 1.0), theta.sigma) +
                 SB.logpdf(SB.Dirichlet([2.0, 2.0, 2.0]), theta.w) +
                 sum(SB.logpdf(SB.Normal(theta.mu, theta.sigma), d) for d in data)
        @test SB.logdensity(model, y; jacobian = false) ≈ manual
        @test SB.logdensity(model, y) ≈ manual + lj
        @test SB.logdensity(model, y) != SB.logdensity(model, y; jacobian = false)
    end

    @testset "gradients agree across backends" begin
        for y in ([0.3, -0.7, 0.2, 1.1], [-1.5, 1.0, -0.4, 0.0], [2.0, 0.1, 1.5, -1.5])
            lp_f, g_f = SB.logdensity_and_gradient(model, y; backend = SB.ForwardDiffAD())
            lp_n, g_n = SB.logdensity_and_gradient(model, y; backend = SB.FiniteDiffAD())
            @test lp_f ≈ lp_n
            @test lp_f ≈ SB.logdensity(model, y)
            @test g_f ≈ g_n rtol = 1e-5
        end
    end

    @testset "gradient of the Jacobian term is not forgotten" begin
        y = [0.3, -0.7, 0.2, 1.1]
        _, g_with = SB.logdensity_and_gradient(model, y)
        _, g_without = SB.logdensity_and_gradient(model, y; jacobian = false)
        @test !isapprox(g_with, g_without; rtol = 1e-6)
    end

    @testset "errors early on a bad specification" begin
        @test_throws ArgumentError SB.Model((mu = 1.0,), theta -> 0.0)
    end
end
