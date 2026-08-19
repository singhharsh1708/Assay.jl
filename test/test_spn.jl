@testset "sum-product networks" begin
    rng = Random.Xoshiro(808)

    comp1 = [SB.Normal(-2.0, 1.0), SB.Normal(0.0, 1.0)]
    comp2 = [SB.Normal(2.0, 0.5), SB.Normal(1.0, 2.0)]
    spn = SB.naive_bayes_spn([0.3, 0.7], [comp1, comp2])

    @testset "structure is validated at construction" begin
        @test SB.scope(spn) == [1, 2]
        @test SB.n_sum_nodes(spn) == 1
        # incomplete sum node: children with different scopes
        @test_throws ArgumentError SB.sum_node(SB.SPNNode[SB.LeafNode(1, SB.Normal(0.0, 1.0)),
                                                          SB.LeafNode(2, SB.Normal(0.0, 1.0))],
                                               [0.5, 0.5])
        # non-decomposable product node: children sharing a variable
        @test_throws ArgumentError SB.product_node(SB.SPNNode[SB.LeafNode(1, SB.Normal(0.0, 1.0)),
                                                              SB.LeafNode(1, SB.Normal(0.0, 1.0))])
        @test_throws ArgumentError SB.sum_node(SB.SPNNode[SB.LeafNode(1, SB.Normal(0.0, 1.0)),
                                                          SB.LeafNode(1, SB.Normal(1.0, 1.0))],
                                               [0.5, 0.6])
    end

    @testset "evaluation matches the mixture it encodes" begin
        for x in ([0.5, 1.0], [-3.0, 2.0], [0.0, 0.0])
            manual = log(0.3 * exp(SB.logpdf(comp1[1], x[1]) + SB.logpdf(comp1[2], x[2])) +
                         0.7 * exp(SB.logpdf(comp2[1], x[1]) + SB.logpdf(comp2[2], x[2])))
            @test SB.logpdf(spn, x) ≈ manual rtol = 1e-12
        end
    end

    @testset "marginalisation is exact, not approximate" begin
        # Setting a leaf to `missing` must give the same answer as integrating
        # that variable out numerically.
        for x1 in (-1.0, 0.5, 2.5)
            numeric = 0.0
            h = 0.002
            for x2 in -30:h:30
                numeric += exp(SB.logpdf(spn, [x1, x2])) * h
            end
            @test exp(SB.logpdf(spn, [x1, missing])) ≈ numeric rtol = 1e-6
        end
        # and both variables missing is the normalisation constant
        @test SB.logpdf(spn, [missing, missing]) ≈ 0 atol = 1e-12
    end

    @testset "the network integrates to one" begin
        h = 0.02
        total = 0.0
        for x1 in -8:h:8, x2 in -12:h:12
            total += exp(SB.logpdf(spn, [x1, x2])) * h^2
        end
        @test total ≈ 1 rtol = 1e-4
    end

    @testset "discrete leaves: exhaustive enumeration" begin
        # With categorical leaves the whole distribution can be enumerated, so
        # the structural claims are checked exactly rather than numerically.
        a = SB.product_node(SB.SPNNode[SB.LeafNode(1, SB.Categorical([0.2, 0.8])),
                                       SB.LeafNode(2, SB.Categorical([0.6, 0.4]))])
        b = SB.product_node(SB.SPNNode[SB.LeafNode(1, SB.Categorical([0.7, 0.3])),
                                       SB.LeafNode(2, SB.Categorical([0.1, 0.9]))])
        net = SB.sum_node(SB.SPNNode[a, b], [0.25, 0.75])
        total = sum(exp(SB.logpdf(net, [i, j])) for i in 1:2, j in 1:2)
        @test total ≈ 1
        for i in 1:2
            marg = sum(exp(SB.logpdf(net, [i, j])) for j in 1:2)
            @test exp(SB.logpdf(net, [i, missing])) ≈ marg
        end
        @test exp(SB.logpdf(net, [1, 1])) ≈ 0.25 * 0.2 * 0.6 + 0.75 * 0.7 * 0.1
    end

    @testset "ancestral sampling matches the density" begin
        n = 100_000
        xs = [SB.rand(Random.Xoshiro(i), spn, 2) for i in 1:n]
        m1 = mean(x[1] for x in xs)
        m2 = mean(x[2] for x in xs)
        e1 = 0.3 * (-2.0) + 0.7 * 2.0
        e2 = 0.3 * 0.0 + 0.7 * 1.0
        # mixture variances, for the standard errors
        v1 = 0.3 * (1.0 + 4.0) + 0.7 * (0.25 + 4.0) - e1^2
        v2 = 0.3 * (1.0 + 0.0) + 0.7 * (4.0 + 1.0) - e2^2
        @test abs(m1 - e1) < 4 * sqrt(v1 / n)
        @test abs(m2 - e2) < 4 * sqrt(v2 / n)
    end

    @testset "inferring the weights is an ordinary model on the simplex" begin
        # The sum weights live on a simplex, which is exactly what the transform
        # layer provides. There is no closed-form posterior here, so calibration
        # is the check.
        leaves = [[SB.Normal(-2.0, 1.0)], [SB.Normal(2.0, 1.0)]]
        nobs = 30
        build = function (data)
            SB.Model((w = SB.simplex(2),),
                     t -> SB.logpdf(SB.Dirichlet([2.0, 2.0]), t.w) +
                          sum(SB.logpdf(SB.naive_bayes_spn(t.w, leaves), [d]) for d in data))
        end
        prior_rand = rng -> (w = SB.rand(rng, SB.Dirichlet([2.0, 2.0])),)
        simulate = (theta, rng) -> [SB.rand(rng, SB.naive_bayes_spn(theta.w, leaves), 1)[1]
                                    for _ in 1:nobs]
        prob = SB.CalibrationProblem(build, prior_rand, simulate)
        res = SB.sbc(Random.Xoshiro(12), prob, SB.NUTS(); n_sims = 150, n_draws = 64,
                     thin = 5, n_warmup = 400)
        @test all(res.pvalue .> 0.01)
    end
end
