using ForwardDiff, LinearAlgebra

"""
Numerical log absolute Jacobian determinant of a transform, by differentiating
the map itself. For the simplex the map is onto `K-1` free coordinates (the last
is determined), so the redundant coordinate is dropped before taking the
determinant.
"""
function numeric_logjac(t, y)
    f = function (z)
        x, _ = SB.to_constrained(t, z)
        v = x isa Real ? [x] : collect(x)
        return length(v) == SB.udim(t) ? v : v[1:SB.udim(t)]
    end
    J = ForwardDiff.jacobian(f, collect(float.(y)))
    return log(abs(det(J)))
end

"""
Simpson quadrature of `f` on `[a, b]`; used to confirm that a density pushed
through a transform still integrates to one, which is the property the Jacobian
term exists to preserve.
"""
function simpson(f, a, b, n)
    n = iseven(n) ? n : n + 1
    h = (b - a) / n
    s = f(a) + f(b)
    for i in 1:(n - 1)
        s += (isodd(i) ? 4 : 2) * f(a + i * h)
    end
    return s * h / 3
end

@testset "transforms" begin
    @testset "round trip" begin
        cases = [
            (SB.positive(), [0.7]),
            (SB.positive(3), [-1.0, 0.4, 2.0]),
            (SB.unit(), [-0.3]),
            (SB.unit(2), [1.2, -2.0]),
            (SB.interval(-2.0, 5.0), [0.8]),
            (SB.interval(-2.0, 5.0, 2), [0.0, -1.1]),
            (SB.lower(3.0), [0.2]),
            (SB.upper(-1.0), [-0.4]),
            (SB.unconstrained(), [1.3]),
            (SB.unconstrained(4), [1.0, -2.0, 0.0, 3.0]),
            (SB.simplex(3), [0.4, -0.9]),
            (SB.simplex(6), [0.4, -0.9, 1.2, 0.0, -2.0]),
            (SB.ordered(4), [-1.0, 0.3, -0.5, 1.1]),
        ]
        for (t, y) in cases
            x, _ = SB.to_constrained(t, y)
            y2 = SB.to_unconstrained(t, x)
            @test collect(y2) ≈ y rtol = 1e-10
            @test SB.udim(t) == length(y)
        end
    end

    @testset "log Jacobian determinant matches automatic differentiation" begin
        cases = [
            (SB.positive(), [0.7]), (SB.positive(3), [-1.0, 0.4, 2.0]),
            (SB.unit(), [-0.3]), (SB.unit(2), [1.2, -2.0]),
            (SB.interval(-2.0, 5.0), [0.8]), (SB.interval(0.0, 1.0, 2), [0.0, -1.1]),
            (SB.lower(3.0), [0.2]), (SB.upper(-1.0), [-0.4]),
            (SB.unconstrained(2), [1.0, -2.0]),
            (SB.simplex(3), [0.4, -0.9]), (SB.simplex(5), [0.4, -0.9, 1.2, 0.1]),
            (SB.ordered(3), [-1.0, 0.3, -0.5]),
        ]
        for (t, y) in cases
            _, lj = SB.to_constrained(t, y)
            @test lj ≈ numeric_logjac(t, y) rtol = 1e-8
        end
    end

    @testset "constraints are respected" begin
        @test SB.to_constrained(SB.positive(), [-30.0])[1] > 0
        @test 0 < SB.to_constrained(SB.unit(), [-40.0])[1] < 1
        x, _ = SB.to_constrained(SB.interval(2.0, 3.0, 3), [-8.0, 0.0, 8.0])
        @test all(2 .< x .< 3)
        # At |y| beyond about 37 the constrained value rounds onto the boundary
        # in Float64. The map is still monotone and the log density still
        # evaluates; the strict inequality is what is lost, and the honest
        # statement is that these transforms are open maps only up to rounding.
        xe, _ = SB.to_constrained(SB.interval(2.0, 3.0, 2), [-50.0, 50.0])
        @test all(2 .<= xe .<= 3)
        @test SB.to_constrained(SB.positive(), [-800.0])[1] == 0.0
        for K in (2, 3, 7)
            y = randn(Random.Xoshiro(K), K - 1)
            x, _ = SB.to_constrained(SB.simplex(K), y)
            @test length(x) == K
            @test all(>(0), x)
            @test sum(x) ≈ 1 rtol = 1e-12
        end
        # the offset in the stick-breaking map puts the origin at the centre
        @test SB.to_constrained(SB.simplex(5), zeros(4))[1] ≈ fill(0.2, 5) rtol = 1e-12
        xo, _ = SB.to_constrained(SB.ordered(5), [0.0, -1.0, 0.5, -3.0, 2.0])
        @test issorted(xo) && allunique(xo)
    end

    @testset "pushforward densities still integrate to one" begin
        # positive support: Gamma through log
        f = y -> exp(SB.logpdf(SB.Gamma(2.5, 1.7), exp(y)) + y)
        @test simpson(f, -30.0, 20.0, 40_000) ≈ 1 atol = 1e-6
        # unit interval: Beta through logit
        g = function (y)
            x, lj = SB.to_constrained(SB.unit(), [y])
            return exp(SB.logpdf(SB.Beta(2.0, 5.0), x) + lj)
        end
        @test simpson(g, -40.0, 40.0, 40_000) ≈ 1 atol = 1e-6
        # bounded interval
        h = function (y)
            x, lj = SB.to_constrained(SB.interval(-1.0, 4.0), [y])
            return exp(SB.logpdf(SB.Uniform(-1.0, 4.0), x) + lj)
        end
        @test simpson(h, -60.0, 60.0, 60_000) ≈ 1 atol = 1e-6
        # without the Jacobian term the same integral is not one: this is the
        # bug the correction exists to prevent
        gbad = y -> exp(SB.logpdf(SB.Beta(2.0, 5.0), SB.to_constrained(SB.unit(), [y])[1]))
        @test !isapprox(simpson(gbad, -40.0, 40.0, 40_000), 1; atol = 1e-3)
    end

    @testset "domain errors" begin
        @test_throws DomainError SB.to_unconstrained(SB.positive(), [-1.0])
        @test_throws DomainError SB.to_unconstrained(SB.unit(), [1.5])
        @test_throws DomainError SB.to_unconstrained(SB.simplex(3), [0.5, 0.6, 0.1])
        @test_throws ArgumentError SB.simplex(1)
        @test_throws ArgumentError SB.interval(2.0, 1.0, 1)
    end
end
