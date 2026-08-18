@testset "sequential Monte Carlo" begin
    rng = Random.Xoshiro(31337)

    @testset "resampling schemes are unbiased" begin
        # A resampler maps N weights to N ancestor indices, so unbiasedness is
        # the statement E[number of copies of i] = N * w_i.
        w = [0.5, 0.3, 0.15, 0.04, 0.01]
        N = length(w)
        reps = 4000
        for scheme in (SB.MultinomialResampling(), SB.StratifiedResampling(),
                       SB.SystematicResampling(), SB.ResidualResampling())
            counts = zeros(N)
            for r in 1:reps
                idx = SB.resample(Random.Xoshiro(r), scheme, w)
                r == 1 && (@test length(idx) == N && all(1 .<= idx .<= N))
                for i in idx
                    counts[i] += 1
                end
            end
            expected = reps * N .* w
            se = sqrt.(reps * N .* w .* (1 .- w))
            @test all(abs.(counts .- expected) .< 5 .* se)
        end
    end

    @testset "systematic resampling has lower variance than multinomial" begin
        w = fill(1 / 50, 50)
        varm = 0.0
        vars = 0.0
        for r in 1:400
            cm = zeros(50); cs = zeros(50)
            for i in SB.resample(Random.Xoshiro(r), SB.MultinomialResampling(), w)
                cm[i] += 1
            end
            for i in SB.resample(Random.Xoshiro(r), SB.SystematicResampling(), w)
                cs[i] += 1
            end
            varm += sum((cm .- 1) .^ 2)
            vars += sum((cs .- 1) .^ 2)
        end
        @test vars < varm
        @test vars == 0                       # equal weights: exactly one copy each
    end

    @testset "effective sample size of weights" begin
        @test SB.ess_weights(zeros(100)) ≈ 100
        logw = fill(-Inf, 100); logw[1] = 0.0
        @test SB.ess_weights(logw) ≈ 1
        @test 1 < SB.ess_weights(randn(1000)) < 1000
    end

    @testset "tempered model interpolates prior and posterior" begin
        data = [1, 0, 1, 1, 1, 0, 1]
        ref = SB.beta_bernoulli(data; a = 2.0, b = 3.0)
        tm = ref.tempered
        y = [0.4]
        @test SB.logdensity(SB.at(tm, 0.0), y) ≈ SB.logdensity(tm.prior, y)
        @test SB.logdensity(SB.at(tm, 1.0), y) ≈ SB.logdensity(ref.model, y)
        @test SB.logdensity(SB.at(tm, 0.5), y) ≈
              SB.logdensity(tm.prior, y) + 0.5 * SB.loglik_at(tm, y)
    end

    @testset "temperature schedule" begin
        loglik = randn(500) .* 5
        b = SB.next_beta(loglik, zeros(500), 0.0, 0.5)
        @test 0 < b <= 1
        if b < 1
            @test SB.ess_weights(b .* loglik) ≈ 250 rtol = 0.05
        end
        # an uninformative likelihood should be absorbed in a single step
        @test SB.next_beta(zeros(500), zeros(500), 0.0, 0.5) == 1.0
    end

    @testset "posterior and evidence against closed form" begin
        data_b = [rand(rng) < 0.4 ? 1 : 0 for _ in 1:60]
        data_p = [SB.rand(rng, SB.Poisson(3.0)) for _ in 1:40]
        data_n = [0.7 .+ randn(rng) for _ in 1:30]
        cases = [(SB.beta_bernoulli(data_b; a = 2.0, b = 2.0), :p),
                 (SB.gamma_poisson(data_p; a = 2.0, b = 1.0), :lambda),
                 (SB.normal_normal(data_n; mu0 = 0.0, tau0 = 5.0, sigma = 1.0), :mu)]
        for (ref, name) in cases
            exact = getproperty(ref.posterior, name)
            # Repeat the whole run: the spread across independent runs is the
            # only honest error bar for a particle method.
            means = Float64[]
            logZs = Float64[]
            for r in 1:8
                res = SB.sample(ref.tempered, SB.SMC(; n_particles = 2000);
                                rng = Random.Xoshiro(100 + r))
                push!(means, SB.weighted_mean(res, name))
                push!(logZs, res.logZ)
            end
            @test abs(mean(means) - SB.mean(exact)) < 4 * std(means) / sqrt(length(means))
            @test abs(mean(logZs) - ref.logevidence) < 4 * std(logZs) / sqrt(length(logZs))
            # the log evidence estimator is consistent, and biased low in log
            # space by Jensen's inequality; it should be within a nat here
            @test abs(mean(logZs) - ref.logevidence) < 1.0
        end
    end

    @testset "quantiles and standard deviation" begin
        data = [rand(rng) < 0.3 ? 1 : 0 for _ in 1:80]
        ref = SB.beta_bernoulli(data; a = 1.0, b = 1.0)
        res = SB.sample(ref.tempered, SB.SMC(; n_particles = 4000); rng = Random.Xoshiro(55))
        exact = ref.posterior.p
        @test SB.weighted_std(res, :p) ≈ sqrt(SB.var(exact)) rtol = 0.1
        for p in (0.1, 0.5, 0.9)
            @test SB.weighted_quantile(res, :p, p) ≈ SB.quantile(exact, p) atol = 0.02
        end
    end

    @testset "an MCMC sampler can be used as the rejuvenation kernel" begin
        data = [SB.rand(rng, SB.Poisson(5.0)) for _ in 1:40]
        ref = SB.gamma_poisson(data; a = 2.0, b = 1.0)
        res = SB.sample(ref.tempered, SB.SMC(; n_particles = 500, kernel = SB.NUTS());
                        rng = Random.Xoshiro(77))
        @test SB.weighted_mean(res, :lambda) ≈ SB.mean(ref.posterior.lambda) rtol = 0.05
        @test abs(res.logZ - ref.logevidence) < 1.0
    end

    @testset "diagnostics of the run are recorded" begin
        data = [rand(rng) < 0.5 ? 1 : 0 for _ in 1:50]
        ref = SB.beta_bernoulli(data)
        res = SB.sample(ref.tempered, SB.SMC(; n_particles = 1000); rng = Random.Xoshiro(88))
        @test res.betas[1] == 0.0 && res.betas[end] == 1.0
        @test issorted(res.betas)
        @test all(0 .< res.ess_trace .<= 1000)
        @test all(0 .<= res.accept_trace .<= 1)
        @test sum(res.weights) ≈ 1
    end
end
