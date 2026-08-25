using LogDensityProblems: LogDensityProblems

# A model written against LogDensityProblems and nothing else, the way a user
# arriving from another package would already have it.
struct BananaProblem
    b::Float64
end
LogDensityProblems.capabilities(::Type{BananaProblem}) = LogDensityProblems.LogDensityOrder{0}()
LogDensityProblems.dimension(::BananaProblem) = 2
function LogDensityProblems.logdensity(p::BananaProblem, y::AbstractVector)
    return AS.logpdf(AS.Normal(0.0, 10.0), y[1]) +
           AS.logpdf(AS.Normal(p.b * (y[1]^2 - 100), 1.0), y[2])
end

# The same thing, but advertising its own exact gradient.
struct GaussianProblem
    n::Int
end
LogDensityProblems.capabilities(::Type{GaussianProblem}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(p::GaussianProblem) = p.n
LogDensityProblems.logdensity(p::GaussianProblem, y) = -0.5 * sum(abs2, y) - p.n / 2 * log(2pi)
LogDensityProblems.logdensity_and_gradient(p::GaussianProblem, y) =
    (LogDensityProblems.logdensity(p, y), -collect(y))

@testset "sampling models this package did not define" begin
    @testset "a plain function is a model" begin
        model = AS.LogDensityModel(y -> -0.5 * sum(abs2, y) - 1.5 * log(2pi), 3)
        @test AS.dimension(model) == 3
        @test AS.flat_dimension(model) == 3
        @test AS.parameter_names(model) == [Symbol("x[$i]") for i in 1:3]
        @test AS.logdensity(model, zeros(3)) ≈ -1.5 * log(2pi)
        @test_throws DimensionMismatch AS.logdensity(model, zeros(2))

        chn = AS.sample(model, AS.NUTS(), 4000; n_warmup = 1000, n_chains = 4,
                        rng = Random.Xoshiro(1))
        for i in 1:3
            x = chn[Symbol("x[$i]")]
            @test check_mean(x, 0.0)
            @test check_std(x, 1.0)
        end
        @test AS.divergences(chn) == 0
    end

    @testset "names can be supplied, and are checked" begin
        model = AS.LogDensityModel(y -> -0.5 * sum(abs2, y), 2; names = [:alpha, :beta])
        @test AS.parameter_names(model) == [:alpha, :beta]
        chn = AS.sample(model, AS.RandomWalkMH(), 1000; n_warmup = 500, rng = Random.Xoshiro(2))
        @test Set(chn.names) == Set([:alpha, :beta])
        @test_throws DimensionMismatch AS.LogDensityModel(identity, 2; names = [:only_one])
        @test_throws ArgumentError AS.LogDensityModel(identity, 0)
    end

    @testset "every sampler accepts a wrapped model" begin
        model = AS.LogDensityModel(y -> AS.logpdf(AS.Normal(1.0, 2.0), y[1]), 1; names = [:mu])
        for spl in (AS.RandomWalkMH(), AS.HMC(), AS.NUTS())
            chn = AS.sample(model, spl, 4000; n_warmup = 1000, n_chains = 4,
                            rng = Random.Xoshiro(3))
            @test check_mean(chn[:mu], 1.0; nse = 5)
            @test check_std(chn[:mu], 2.0; nse = 5)
        end
        res = AS.sample(model, AS.ADVI(); rng = Random.Xoshiro(4))
        vc = AS.posterior_samples(res, 20_000; rng = Random.Xoshiro(5))
        @test mean(vec(vc[:mu])) ≈ 1.0 atol = 0.1
        @test std(vec(vc[:mu])) ≈ 2.0 rtol = 0.1
    end

    @testset "a LogDensityProblems object without a gradient" begin
        # capability order 0: the wrapper has to supply the gradient itself
        model = AS.LogDensityModel(BananaProblem(0.03))
        @test AS.dimension(model) == 2
        chn = AS.sample(model, AS.NUTS(; target_accept = 0.99), 10_000; n_warmup = 2000,
                        n_chains = 4, rng = Random.Xoshiro(6))
        @test check_std(chn[Symbol("x[1]")], 10.0; nse = 5)
        @test check_std(chn[Symbol("x[2]")], sqrt(1 + 2 * 0.03^2 * 100^2); nse = 5)
    end

    @testset "a LogDensityProblems object carrying its own gradient" begin
        problem = GaussianProblem(4)
        model = AS.LogDensityModel(problem)
        y = [0.3, -1.2, 0.7, 2.0]
        # the wrapper must use the problem's gradient, not differentiate again
        v, g = AS.logdensity_and_gradient(model, y)
        @test v ≈ LogDensityProblems.logdensity(problem, y)
        @test g ≈ -y
        chn = AS.sample(model, AS.NUTS(), 4000; n_warmup = 1000, n_chains = 4,
                        rng = Random.Xoshiro(7))
        for i in 1:4
            @test check_mean(chn[Symbol("x[$i]")], 0.0; nse = 5)
            @test check_std(chn[Symbol("x[$i]")], 1.0; nse = 5)
        end
    end

    @testset "an Assay model satisfies the interface in the other direction" begin
        data = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1]
        ref = AS.beta_bernoulli(data; a = 2.0, b = 2.0)
        model = ref.model
        @test LogDensityProblems.dimension(model) == 1
        @test LogDensityProblems.capabilities(typeof(model)) ==
              LogDensityProblems.LogDensityOrder{1}()
        y = [0.4]
        @test LogDensityProblems.logdensity(model, y) ≈ AS.logdensity(model, y)
        v, g = LogDensityProblems.logdensity_and_gradient(model, y)
        @test v ≈ AS.logdensity(model, y)
        @test g ≈ AS.logdensity_and_gradient(model, y)[2]
        # and the Jacobian correction travels with it, which is the whole point
        @test v != ref.model.logjoint((p = AS.logistic(y[1]),))
    end

    @testset "sequential Monte Carlo over a wrapped density" begin
        # TemperedModel has to rebuild the target at each temperature, which for
        # a wrapped density means assembling another wrapped density.
        data = [AS.rand(Random.Xoshiro(8), AS.Poisson(4.0)) for _ in 1:40]
        prior = AS.LogDensityModel(y -> AS.logpdf(AS.Normal(0.0, 2.0), y[1]), 1; names = [:mu])
        tm = AS.TemperedModel(prior,
                              theta -> sum(AS.logpdf(AS.Normal(theta.x[1], 1.0), d) for d in data);
                              prior_rand = rng -> (x = [rand(rng, AS.Normal(0.0, 2.0))],))
        res = AS.sample(tm, AS.SMC(; n_particles = 2000); rng = Random.Xoshiro(9))
        # conjugate normal-normal in disguise: posterior mean and sd are exact
        n = length(data); sy = sum(data)
        prec = 1 / 4 + n
        @test AS.weighted_mean(res, :mu) ≈ sy / prec rtol = 0.02
        @test AS.weighted_std(res, :mu) ≈ sqrt(1 / prec) rtol = 0.1
        @test isfinite(res.logZ)
    end

    @testset "wrapping does not smuggle in a transform" begin
        # A wrapped log density has no declared constraints, so `constrain` must
        # report a zero Jacobian rather than inventing one.
        model = AS.LogDensityModel(y -> -0.5 * sum(abs2, y), 2)
        theta, logjac = AS.constrain(model, [0.5, -0.5])
        @test theta.x == [0.5, -0.5]
        @test logjac == 0.0
        @test AS.unconstrain(model, theta) == [0.5, -0.5]
        @test AS.flatten_draw(model, [0.5, -0.5]) == [0.5, -0.5]
    end
end
