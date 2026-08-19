@testset "random walk Metropolis on conjugate models" begin
    rng = Random.Xoshiro(4321)

    @testset "Beta-Bernoulli" begin
        data = [rand(rng) < 0.3 ? 1 : 0 for _ in 1:80]
        ref = AS.beta_bernoulli(data; a = 2.0, b = 2.0)
        chn = AS.sample(ref.model, AS.RandomWalkMH(), 20_000;
                        n_warmup = 2_000, n_chains = 4, rng = Random.Xoshiro(1))
        check_conjugate(chn, :p, ref.posterior.p)
    end

    @testset "Normal-Normal with known variance" begin
        data = [2.0 + 1.5 * randn(rng) for _ in 1:40]
        ref = AS.normal_normal(data; mu0 = 0.0, tau0 = 5.0, sigma = 1.5)
        chn = AS.sample(ref.model, AS.RandomWalkMH(), 20_000;
                        n_warmup = 2_000, n_chains = 4, rng = Random.Xoshiro(2))
        check_conjugate(chn, :mu, ref.posterior.mu)
    end

    @testset "Gamma-Poisson" begin
        data = [AS.rand(rng, AS.Poisson(3.5)) for _ in 1:60]
        ref = AS.gamma_poisson(data; a = 2.0, b = 1.0)
        chn = AS.sample(ref.model, AS.RandomWalkMH(), 20_000;
                        n_warmup = 2_000, n_chains = 4, rng = Random.Xoshiro(3))
        check_conjugate(chn, :lambda, ref.posterior.lambda)
    end

    @testset "Barker's rule targets the same posterior" begin
        data = [rand(rng) < 0.6 ? 1 : 0 for _ in 1:50]
        ref = AS.beta_bernoulli(data; a = 1.0, b = 1.0)
        chn = AS.sample(ref.model, AS.RandomWalkMH(; rule = AS.BarkerRule(), target_accept = 0.15),
                        20_000; n_warmup = 2_000, n_chains = 4, rng = Random.Xoshiro(4))
        check_conjugate(chn, :p, ref.posterior.p)
    end

    @testset "adapted covariance on a correlated Gaussian" begin
        Sigma = [1.0 0.9; 0.9 1.0]
        target = AS.MvNormal([0.5, -0.5], Sigma)
        model = AS.Model((x = AS.unconstrained(2),), theta -> AS.logpdf(target, theta.x))
        chn = AS.sample(model, AS.RandomWalkMH(; adapt_cov = true), 20_000;
                        n_warmup = 5_000, n_chains = 4, rng = Random.Xoshiro(5))
        x1 = chn[Symbol("x[1]")]
        x2 = chn[Symbol("x[2]")]
        @test check_mean(x1, 0.5)
        @test check_mean(x2, -0.5)
        @test check_std(x1, 1.0; nse = 5)
        @test cor(vec(x1), vec(x2)) ≈ 0.9 atol = 0.02
    end

    @testset "reproducibility" begin
        ref = AS.beta_bernoulli([1, 0, 1, 1, 0]; a = 1.0, b = 1.0)
        a = AS.sample(ref.model, AS.RandomWalkMH(), 500; n_warmup = 200, rng = Random.Xoshiro(99))
        b = AS.sample(ref.model, AS.RandomWalkMH(), 500; n_warmup = 200, rng = Random.Xoshiro(99))
        @test a.value == b.value
    end
end
