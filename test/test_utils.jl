@testset "numerics" begin
    @testset "logistic / logit round trip" begin
        for y in (-800.0, -37.5, -1.0, 0.0, 0.25, 12.0, 700.0)
            s = AS.logistic(y)
            @test 0 <= s <= 1
            if 0 < s < 1
                @test AS.logit(s) ≈ y atol = 1e-8
            end
        end
        @test AS.logistic(-800.0) == 0.0 || AS.logistic(-800.0) < 1e-300
        @test AS.logistic(800.0) == 1.0
    end

    @testset "log1pexp is stable in both tails" begin
        for y in (-1000.0, -50.0, -1.0, 0.0, 1.0, 20.0, 40.0, 1e5)
            v = AS.log1pexp(y)
            @test isfinite(v)
            @test v >= max(y, 0)
            if abs(y) < 30                       # where the naive form is safe
                @test v ≈ log(1 + exp(y)) rtol = 1e-12
            end
        end
        @test AS.log1pexp(1e5) ≈ 1e5
        @test AS.loglogistic(-40.0) ≈ -40.0 rtol = 1e-12
    end

    @testset "logsumexp" begin
        x = [-1000.0, -1001.0, -1002.0]
        @test AS.logsumexp(x) ≈ -1000 + log(1 + exp(-1.0) + exp(-2.0)) rtol = 1e-12
        @test AS.logsumexp([-Inf, -Inf]) == -Inf
        @test AS.logsumexp([0.0, -Inf]) == 0.0
        w = zeros(3)
        lse = AS.softmax!(w, [1.0, 2.0, 3.0])
        @test sum(w) ≈ 1
        @test lse ≈ AS.logsumexp([1.0, 2.0, 3.0])
        # all weights collapsed: fall back to uniform rather than NaN
        AS.softmax!(w, fill(-Inf, 3))
        @test all(w .≈ 1 / 3)
    end
end
