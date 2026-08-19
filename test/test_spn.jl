@testset "sum-product networks" begin
    rng = Random.Xoshiro(808)

    comp1 = [AS.Normal(-2.0, 1.0), AS.Normal(0.0, 1.0)]
    comp2 = [AS.Normal(2.0, 0.5), AS.Normal(1.0, 2.0)]
    spn = AS.naive_bayes_spn([0.3, 0.7], [comp1, comp2])

    @testset "structure is validated at construction" begin
        @test AS.scope(spn) == [1, 2]
        @test AS.n_sum_nodes(spn) == 1
        # incomplete sum node: children with different scopes
        @test_throws ArgumentError AS.sum_node(AS.SPNNode[AS.LeafNode(1, AS.Normal(0.0, 1.0)),
                                                          AS.LeafNode(2, AS.Normal(0.0, 1.0))],
                                               [0.5, 0.5])
        # non-decomposable product node: children sharing a variable
        @test_throws ArgumentError AS.product_node(AS.SPNNode[AS.LeafNode(1, AS.Normal(0.0, 1.0)),
                                                              AS.LeafNode(1, AS.Normal(0.0, 1.0))])
        @test_throws ArgumentError AS.sum_node(AS.SPNNode[AS.LeafNode(1, AS.Normal(0.0, 1.0)),
                                                          AS.LeafNode(1, AS.Normal(1.0, 1.0))],
                                               [0.5, 0.6])
    end

    @testset "evaluation matches the mixture it encodes" begin
        for x in ([0.5, 1.0], [-3.0, 2.0], [0.0, 0.0])
            manual = log(0.3 * exp(AS.logpdf(comp1[1], x[1]) + AS.logpdf(comp1[2], x[2])) +
                         0.7 * exp(AS.logpdf(comp2[1], x[1]) + AS.logpdf(comp2[2], x[2])))
            @test AS.logpdf(spn, x) ≈ manual rtol = 1e-12
        end
    end

    @testset "marginalisation is exact, not approximate" begin
        # Setting a leaf to `missing` must give the same answer as integrating
        # that variable out numerically.
        for x1 in (-1.0, 0.5, 2.5)
            numeric = 0.0
            h = 0.002
            for x2 in -30:h:30
                numeric += exp(AS.logpdf(spn, [x1, x2])) * h
            end
            @test exp(AS.logpdf(spn, [x1, missing])) ≈ numeric rtol = 1e-6
        end
        # and both variables missing is the normalisation constant
        @test AS.logpdf(spn, [missing, missing]) ≈ 0 atol = 1e-12
    end

    @testset "the network integrates to one" begin
        h = 0.02
        total = 0.0
        for x1 in -8:h:8, x2 in -12:h:12
            total += exp(AS.logpdf(spn, [x1, x2])) * h^2
        end
        @test total ≈ 1 rtol = 1e-4
    end

    @testset "discrete leaves: exhaustive enumeration" begin
        # With categorical leaves the whole distribution can be enumerated, so
        # the structural claims are checked exactly rather than numerically.
        a = AS.product_node(AS.SPNNode[AS.LeafNode(1, AS.Categorical([0.2, 0.8])),
                                       AS.LeafNode(2, AS.Categorical([0.6, 0.4]))])
        b = AS.product_node(AS.SPNNode[AS.LeafNode(1, AS.Categorical([0.7, 0.3])),
                                       AS.LeafNode(2, AS.Categorical([0.1, 0.9]))])
        net = AS.sum_node(AS.SPNNode[a, b], [0.25, 0.75])
        total = sum(exp(AS.logpdf(net, [i, j])) for i in 1:2, j in 1:2)
        @test total ≈ 1
        for i in 1:2
            marg = sum(exp(AS.logpdf(net, [i, j])) for j in 1:2)
            @test exp(AS.logpdf(net, [i, missing])) ≈ marg
        end
        @test exp(AS.logpdf(net, [1, 1])) ≈ 0.25 * 0.2 * 0.6 + 0.75 * 0.7 * 0.1
    end

    @testset "ancestral sampling matches the density" begin
        n = 100_000
        xs = [AS.rand(Random.Xoshiro(i), spn, 2) for i in 1:n]
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
        leaves = [[AS.Normal(-2.0, 1.0)], [AS.Normal(2.0, 1.0)]]
        nobs = 30
        build = function (data)
            AS.Model((w = AS.simplex(2),),
                     t -> AS.logpdf(AS.Dirichlet([2.0, 2.0]), t.w) +
                          sum(AS.logpdf(AS.naive_bayes_spn(t.w, leaves), [d]) for d in data))
        end
        prior_rand = rng -> (w = AS.rand(rng, AS.Dirichlet([2.0, 2.0])),)
        simulate = (theta, rng) -> [AS.rand(rng, AS.naive_bayes_spn(theta.w, leaves), 1)[1]
                                    for _ in 1:nobs]
        prob = AS.CalibrationProblem(build, prior_rand, simulate)
        res = AS.sbc(Random.Xoshiro(12), prob, AS.NUTS(); n_sims = 150, n_draws = 64,
                     thin = 5, n_warmup = 400)
        @test all(res.pvalue .> 0.01)
    end
end
