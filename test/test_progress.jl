using Logging

# A logger that keeps what it was told rather than printing it, which is the
# only way to test progress reporting without deciding what a display looks
# like.
struct CollectingLogger <: AbstractLogger
    records::Vector{NamedTuple}
end
CollectingLogger() = CollectingLogger(NamedTuple[])
Logging.min_enabled_level(::CollectingLogger) = AS.ProgressLevel
Logging.shouldlog(::CollectingLogger, args...) = true
Logging.catch_exceptions(::CollectingLogger) = false
function Logging.handle_message(l::CollectingLogger, level, message, _mod, group, id,
                                file, line; kwargs...)
    push!(l.records, (level = level, message = string(message), kwargs = Dict(kwargs)))
    return nothing
end

progress_records(f) = (l = CollectingLogger(); Logging.with_logger(f, l); l.records)
fractions(rs) = [r.kwargs[:progress] for r in rs
                 if r.level == AS.ProgressLevel && haskey(r.kwargs, :progress)]

@testset "progress reporting" begin
    rng = Random.Xoshiro(909)
    data = randn(rng, 30)
    model = AS.Model((mu = AS.unconstrained(),),
                     t -> AS.logpdf(AS.Normal(0.0, 5.0), t.mu) +
                          AS.loglikelihood(AS.Normal(t.mu, 1.0), data))

    @testset "silent unless asked" begin
        rs = progress_records() do
            AS.sample(model, AS.NUTS(), 200; n_warmup = 200, rng = Random.Xoshiro(1))
        end
        @test isempty(fractions(rs))
        @test isempty(rs)
    end

    @testset "a fraction per update, and a closing message" begin
        # interval zero means every tick reports, which is what makes the
        # sequence checkable rather than dependent on how fast the machine is
        rs = progress_records() do
            AS.sample(model, AS.NUTS(), 100; n_warmup = 100, progress = true,
                      progress_interval = 0.0, rng = Random.Xoshiro(2))
        end
        fs = fractions(rs)
        @test length(fs) >= 3
        @test first(fs) == 0.0
        @test fs[end] == "done"                        # the bar is closed
        numeric = Float64[f for f in fs if f isa Real]
        @test issorted(numeric)                        # progress does not go backwards
        @test all(0 .<= numeric .<= 1)
        @test numeric[end] ≈ 1.0

        # and something visible at Info, since the default logger filters the
        # level a progress bar listens on and the point is not printing nothing
        info_lines = [r.message for r in rs if r.level == Logging.Info]
        @test !isempty(info_lines)
        @test any(m -> occursin("100%", m), info_lines)
        @test any(m -> occursin("finished 200 iterations", m), info_lines)
    end

    @testset "the throttle holds the line count down" begin
        # a ten second interval on a run that takes well under a second: the
        # start, and the close, and nothing in between
        rs = progress_records() do
            AS.sample(model, AS.NUTS(), 100; n_warmup = 100, progress = true,
                      progress_interval = 10.0, rng = Random.Xoshiro(3))
        end
        @test length(fractions(rs)) <= 3
        @test fractions(rs)[end] == "done"
    end

    @testset "every chain counts towards one total" begin
        rs = progress_records() do
            AS.sample(model, AS.NUTS(), 50; n_warmup = 50, n_chains = 4, progress = true,
                      progress_interval = 0.0, rng = Random.Xoshiro(4))
        end
        @test any(m -> occursin("of 400 iterations", m),
                  [r.message for r in rs if r.level == Logging.Info])
        @test any(m -> occursin("finished 400 iterations", m),
                  [r.message for r in rs if r.level == Logging.Info])
    end

    @testset "the variational optimiser reports too" begin
        rs = progress_records() do
            AS.sample(model, AS.ADVI(; n_iterations = 300); progress = true,
                      progress_interval = 0.0, rng = Random.Xoshiro(5))
        end
        fs = fractions(rs)
        @test !isempty(fs)
        @test fs[end] == "done"
        @test any(m -> occursin("variational", m), [r.message for r in rs])
    end

    @testset "sequential Monte Carlo reports the temperature" begin
        # There is no denominator here: the number of steps to reach beta = 1 is
        # chosen adaptively. The temperature is the progress.
        tm = AS.TemperedModel(AS.Model((mu = AS.unconstrained(),),
                                       t -> AS.logpdf(AS.Normal(0.0, 5.0), t.mu)),
                              theta -> AS.loglikelihood(AS.Normal(theta.mu, 1.0), data);
                              prior_rand = r -> (mu = 5 * randn(r),))
        rs = progress_records() do
            AS.sample(tm, AS.SMC(; n_particles = 200); progress = true,
                      progress_interval = 0.0, rng = Random.Xoshiro(6))
        end
        numeric = Float64[f for f in fractions(rs) if f isa Real]
        @test issorted(numeric)
        @test numeric[end] ≈ 1.0                       # tempering finished
        @test fractions(rs)[end] == "done"
    end

    @testset "durations read as durations" begin
        @test AS.format_duration(0.42) == "0.4s"
        @test AS.format_duration(12) == "12s"
        @test AS.format_duration(125) == "2m 05s"
        @test AS.format_duration(7325) == "2h 02m"
        @test AS.format_duration(NaN) == "unknown"
    end
end
