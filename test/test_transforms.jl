using ForwardDiff, LinearAlgebra

"""
Numerical log absolute Jacobian determinant of a transform, by differentiating
the map itself. For the simplex the map is onto `K-1` free coordinates (the last
is determined), so the redundant coordinate is dropped before taking the
determinant.
"""
function numeric_logjac(t, y)
    f = function (z)
        x, _ = AS.to_constrained(t, z)
        v = x isa Real ? [x] : collect(x)
        return length(v) == AS.udim(t) ? v : v[1:AS.udim(t)]
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
            (AS.positive(), [0.7]),
            (AS.positive(3), [-1.0, 0.4, 2.0]),
            (AS.unit(), [-0.3]),
            (AS.unit(2), [1.2, -2.0]),
            (AS.interval(-2.0, 5.0), [0.8]),
            (AS.interval(-2.0, 5.0, 2), [0.0, -1.1]),
            (AS.lower(3.0), [0.2]),
            (AS.upper(-1.0), [-0.4]),
            (AS.unconstrained(), [1.3]),
            (AS.unconstrained(4), [1.0, -2.0, 0.0, 3.0]),
            (AS.simplex(3), [0.4, -0.9]),
            (AS.simplex(6), [0.4, -0.9, 1.2, 0.0, -2.0]),
            (AS.ordered(4), [-1.0, 0.3, -0.5, 1.1]),
        ]
        for (t, y) in cases
            x, _ = AS.to_constrained(t, y)
            y2 = AS.to_unconstrained(t, x)
            @test collect(y2) ≈ y rtol = 1e-10
            @test AS.udim(t) == length(y)
        end
    end

    @testset "log Jacobian determinant matches automatic differentiation" begin
        cases = [
            (AS.positive(), [0.7]), (AS.positive(3), [-1.0, 0.4, 2.0]),
            (AS.unit(), [-0.3]), (AS.unit(2), [1.2, -2.0]),
            (AS.interval(-2.0, 5.0), [0.8]), (AS.interval(0.0, 1.0, 2), [0.0, -1.1]),
            (AS.lower(3.0), [0.2]), (AS.upper(-1.0), [-0.4]),
            (AS.unconstrained(2), [1.0, -2.0]),
            (AS.simplex(3), [0.4, -0.9]), (AS.simplex(5), [0.4, -0.9, 1.2, 0.1]),
            (AS.ordered(3), [-1.0, 0.3, -0.5]),
        ]
        for (t, y) in cases
            _, lj = AS.to_constrained(t, y)
            @test lj ≈ numeric_logjac(t, y) rtol = 1e-8
        end
    end

    @testset "constraints are respected" begin
        @test AS.to_constrained(AS.positive(), [-30.0])[1] > 0
        @test 0 < AS.to_constrained(AS.unit(), [-40.0])[1] < 1
        x, _ = AS.to_constrained(AS.interval(2.0, 3.0, 3), [-8.0, 0.0, 8.0])
        @test all(2 .< x .< 3)
        # At |y| beyond about 37 the constrained value rounds onto the boundary
        # in Float64. The map is still monotone and the log density still
        # evaluates; the strict inequality is what is lost, and the honest
        # statement is that these transforms are open maps only up to rounding.
        xe, _ = AS.to_constrained(AS.interval(2.0, 3.0, 2), [-50.0, 50.0])
        @test all(2 .<= xe .<= 3)
        @test AS.to_constrained(AS.positive(), [-800.0])[1] == 0.0
        for K in (2, 3, 7)
            y = randn(Random.Xoshiro(K), K - 1)
            x, _ = AS.to_constrained(AS.simplex(K), y)
            @test length(x) == K
            @test all(>(0), x)
            @test sum(x) ≈ 1 rtol = 1e-12
        end
        # the offset in the stick-breaking map puts the origin at the centre
        @test AS.to_constrained(AS.simplex(5), zeros(4))[1] ≈ fill(0.2, 5) rtol = 1e-12
        xo, _ = AS.to_constrained(AS.ordered(5), [0.0, -1.0, 0.5, -3.0, 2.0])
        @test issorted(xo) && allunique(xo)
    end

    @testset "pushforward densities still integrate to one" begin
        # positive support: Gamma through log
        f = y -> exp(AS.logpdf(AS.Gamma(2.5, 1.7), exp(y)) + y)
        @test simpson(f, -30.0, 20.0, 40_000) ≈ 1 atol = 1e-6
        # unit interval: Beta through logit
        g = function (y)
            x, lj = AS.to_constrained(AS.unit(), [y])
            return exp(AS.logpdf(AS.Beta(2.0, 5.0), x) + lj)
        end
        @test simpson(g, -40.0, 40.0, 40_000) ≈ 1 atol = 1e-6
        # bounded interval
        h = function (y)
            x, lj = AS.to_constrained(AS.interval(-1.0, 4.0), [y])
            return exp(AS.logpdf(AS.Uniform(-1.0, 4.0), x) + lj)
        end
        @test simpson(h, -60.0, 60.0, 60_000) ≈ 1 atol = 1e-6
        # without the Jacobian term the same integral is not one: this is the
        # bug the correction exists to prevent
        gbad = y -> exp(AS.logpdf(AS.Beta(2.0, 5.0), AS.to_constrained(AS.unit(), [y])[1]))
        @test !isapprox(simpson(gbad, -40.0, 40.0, 40_000), 1; atol = 1e-3)
    end

    @testset "domain errors" begin
        @test_throws DomainError AS.to_unconstrained(AS.positive(), [-1.0])
        @test_throws DomainError AS.to_unconstrained(AS.unit(), [1.5])
        @test_throws DomainError AS.to_unconstrained(AS.simplex(3), [0.5, 0.6, 0.1])
        @test_throws ArgumentError AS.simplex(1)
        @test_throws ArgumentError AS.interval(2.0, 1.0, 1)
    end
end
