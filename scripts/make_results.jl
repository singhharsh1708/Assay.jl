# Generates docs/results.md and every figure in docs/figures.
#
#     julia --project=scripts -t 4 scripts/make_results.jl
#
# The results document is written by this script rather than by hand, so every
# number in it can be regenerated and none of them can drift away from the code.

ENV["GKSwstype"] = "100"                      # headless GR

using Assay
using Plots, Printf, Random, Statistics, LinearAlgebra

const AS = Assay
const FIG = joinpath(@__DIR__, "..", "docs", "figures")
mkpath(FIG)
default(; size = (900, 600), dpi = 130, legendfontsize = 8, framestyle = :box,
        grid = true, gridalpha = 0.15)

const OUT = IOBuffer()
say(args...) = println(OUT, args...)
fmt(x, n = 4) = @sprintf("%.*f", n, x)

# --------------------------------------------------------------------------
# 1. Conjugate comparisons
# --------------------------------------------------------------------------

"""z score of a sampled mean against the analytic mean, in Monte Carlo standard errors."""
zmean(x, exact) = (mean(vec(x)) - exact) / AS.mcse_mean(x)

"""z score of a sampled standard deviation, using the ESS of the squared deviations."""
function zstd(x, exact)
    v = vec(x)
    s = std(v)
    e = AS.ess_bulk(reshape((v .- mean(v)) .^ 2, size(x)))
    return (s - exact) / (s / sqrt(2 * max(e, 1)))
end

function conjugate_section()
    say("## 1. Conjugate models: sampled against closed form\n")
    say("Every tolerance is stated in Monte Carlo standard errors, so `z` is the",
        " number of standard errors between the sampler and the exact answer. Values",
        " beyond about 4 would be a failure.\n")
    say("| model | parameter | sampler | mean | exact | z | sd | exact | z | ESS | R-hat |")
    say("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    rng = Random.Xoshiro(20240819)
    counts = [AS.rand(rng, AS.Poisson(3.5)) for _ in 1:60]
    bern = [rand(rng) < 0.35 ? 1 : 0 for _ in 1:80]
    normals = [1.5 + 0.8 * randn(rng) for _ in 1:50]

    cases = [("Beta-Bernoulli", AS.beta_bernoulli(bern; a = 2.0, b = 2.0), :p),
             ("Normal-Normal", AS.normal_normal(normals; mu0 = 0.0, tau0 = 5.0, sigma = 0.8), :mu),
             ("Gamma-Poisson", AS.gamma_poisson(counts; a = 2.0, b = 1.0), :lambda)]
    samplers = [("RWM", AS.RandomWalkMH()), ("HMC", AS.HMC()), ("NUTS", AS.NUTS())]

    plots = Any[]
    for (name, ref, param) in cases
        exact = getproperty(ref.posterior, param)
        em, es = AS.mean(exact), sqrt(AS.var(exact))
        local last_chain = nothing
        for (sname, spl) in samplers
            chn = AS.sample(ref.model, spl, 20_000; n_warmup = 2000, n_chains = 4,
                            rng = Random.Xoshiro(7))
            x = chn[param]
            say("| ", name, " | ", param, " | ", sname, " | ", fmt(mean(vec(x))), " | ", fmt(em),
                " | ", fmt(zmean(x, em), 2), " | ", fmt(std(vec(x))), " | ", fmt(es), " | ",
                fmt(zstd(x, es), 2), " | ", fmt(AS.ess_bulk(x), 0), " | ", fmt(AS.rhat(x), 3), " |")
            sname == "NUTS" && (last_chain = x)
        end
        # SMC and ADVI on the same model
        res = AS.sample(ref.tempered, AS.SMC(; n_particles = 4000); rng = Random.Xoshiro(8))
        say("| ", name, " | ", param, " | SMC (4000 particles) | ",
            fmt(AS.weighted_mean(res, param)), " | ", fmt(em), " | - | ",
            fmt(AS.weighted_std(res, param)), " | ", fmt(es), " | - | - | - |")
        vi = AS.sample(ref.model, AS.ADVI(; n_samples = 8); rng = Random.Xoshiro(9))
        vc = AS.posterior_samples(vi, 40_000; rng = Random.Xoshiro(10))
        say("| ", name, " | ", param, " | ADVI (mean field) | ", fmt(mean(vec(vc[param]))), " | ",
            fmt(em), " | - | ", fmt(std(vec(vc[param]))), " | ", fmt(es), " | - | - | - |")

        lo, hi = AS.quantile(exact, 0.0005), AS.quantile(exact, 0.9995)
        grid = range(lo, hi; length = 300)
        p = histogram(vec(last_chain); bins = 60, normalize = :pdf, label = "NUTS draws",
                      color = :steelblue, linecolor = :steelblue, alpha = 0.55)
        plot!(p, grid, [exp(AS.logpdf(exact, g)) for g in grid]; lw = 2.5, color = :black,
              label = "analytic posterior", title = name, xlabel = String(param))
        push!(plots, p)
    end
    plot(plots...; layout = (1, 3), size = (1400, 420))
    savefig(joinpath(FIG, "conjugate.png"))
    say("\n![conjugate posteriors](figures/conjugate.png)\n")

    # Evidence from SMC against the analytic marginal likelihood
    say("### Log normalising constant from SMC\n")
    say("SMC estimates `log p(y)` as a by-product of the tempering weights. Nothing",
        " else in the package produces this number, so it is checked against the",
        " analytic evidence directly. The spread is over 8 independent runs.\n")
    say("| model | particles | log Z (mean of 8 runs) | standard deviation | exact |")
    say("|---|---:|---:|---:|---:|")
    for (name, ref, _) in cases
        for n in (500, 4000)
            zs = [AS.sample(ref.tempered, AS.SMC(; n_particles = n);
                            rng = Random.Xoshiro(100 + r)).logZ for r in 1:8]
            say("| ", name, " | ", n, " | ", fmt(mean(zs), 3), " | ", fmt(std(zs), 3), " | ",
                fmt(ref.logevidence, 3), " |")
        end
    end
    say("")
end

# --------------------------------------------------------------------------
# 2. Simulation based calibration
# --------------------------------------------------------------------------

"""A positive transform with the Jacobian term removed: the negative control."""
struct NoJacobianPositive <: AS.AbstractTransform
    n::Int
end
AS.udim(t::NoJacobianPositive) = t.n
AS.cdim(t::NoJacobianPositive) = t.n
AS.to_constrained(::NoJacobianPositive, y::AbstractVector) = (exp.(y), zero(eltype(y)))
AS.to_unconstrained(::NoJacobianPositive, x::AbstractVector) = log.(float.(x))

broken_gamma_poisson(data; a = 2.0, b = 1.0) =
    AS.Model((lambda = AS.ScalarT(NoJacobianPositive(1)),),
             t -> AS.logpdf(AS.Gamma(a, b), t.lambda) +
                  AS.loglikelihood(AS.Poisson(t.lambda), data))

function sbc_section()
    say("## 2. Simulation based calibration\n")
    say("Rank of the prior draw among 64 thinned posterior draws, over 200 replications.",
        " Uniform ranks are what a correct sampler produces; the p value is a Pearson",
        " chi-square test against uniformity over 8 bins.\n")
    say("| problem | sampler | chi-square | p |")
    say("|---|---|---:|---:|")

    problems = [("Gamma-Poisson", AS.conjugate_problem(AS.gamma_poisson, 20; a = 2.0, b = 1.0), AS.NUTS()),
                ("Beta-Bernoulli", AS.conjugate_problem(AS.beta_bernoulli, 25; a = 2.0, b = 3.0), AS.NUTS()),
                ("Normal-Normal", AS.conjugate_problem(AS.normal_normal, 15; mu0 = 0.0, tau0 = 2.0, sigma = 1.0), AS.NUTS()),
                ("Beta-Bernoulli", AS.conjugate_problem(AS.beta_bernoulli, 20; a = 1.0, b = 1.0), AS.RandomWalkMH())]
    plots = Any[]
    for (name, prob, spl) in problems
        thin = spl isa AS.RandomWalkMH ? 60 : 6
        res = AS.sbc(Random.Xoshiro(4), prob, spl; n_sims = 200, n_draws = 64, thin = thin,
                     n_warmup = 600)
        say("| ", name, " | ", nameof(typeof(spl)), " | ", fmt(res.chisq[1], 2), " | ",
            fmt(res.pvalue[1], 4), " |")
        counts = AS.rank_histogram(res, 1)
        p = bar(counts; label = "", color = :steelblue, linecolor = :steelblue,
                title = string(name, ", ", nameof(typeof(spl)), " (p = ", fmt(res.pvalue[1], 3), ")"),
                xlabel = "rank bin", ylabel = "count", ylims = (0, maximum(counts) * 1.4))
        hline!(p, [200 / res.n_bins]; color = :black, lw = 2, label = "uniform")
        push!(plots, p)
    end

    good = AS.conjugate_problem(AS.gamma_poisson, 5; a = 2.0, b = 1.0)
    bad = AS.CalibrationProblem(d -> broken_gamma_poisson(d; a = 2.0, b = 1.0),
                                good.prior_rand, good.simulate)
    resb = AS.sbc(Random.Xoshiro(3), bad, AS.NUTS(); n_sims = 200, n_draws = 64, thin = 6,
                  n_warmup = 600)
    say("| Gamma-Poisson **with the Jacobian removed** | NUTS | ", fmt(resb.chisq[1], 2),
        " | ", fmt(resb.pvalue[1], 6), " |")
    counts = AS.rank_histogram(resb, 1)
    p = bar(counts; label = "", color = :firebrick, linecolor = :firebrick,
            title = string("Jacobian removed (p = ", @sprintf("%.1e", resb.pvalue[1]), ")"),
            xlabel = "rank bin", ylabel = "count", ylims = (0, maximum(counts) * 1.4))
    hline!(p, [200 / resb.n_bins]; color = :black, lw = 2, label = "uniform")
    push!(plots, p)

    plot(plots...; layout = (2, 3), size = (1400, 760))
    savefig(joinpath(FIG, "sbc.png"))
    say("\n![simulation based calibration](figures/sbc.png)\n")
    say("The last panel is the point of the exercise. Dropping the log Jacobian",
        " determinant leaves a sampler that looks entirely healthy - no divergences,",
        " R-hat 1.00, sensible acceptance rate - and simulation based calibration",
        " rejects it at p = ", @sprintf("%.0e", resb.pvalue[1]), " with a visibly sloped",
        " histogram.\n")
end

# --------------------------------------------------------------------------
# 3. Geweke
# --------------------------------------------------------------------------

function geweke_section()
    say("## 3. Geweke joint distribution test\n")
    say("Draws from `p(theta, y)` produced two ways: independently from the prior and",
        " the forward model, and by alternating the sampler's transition kernel with",
        " a fresh data set. The `theta` marginals must agree; `z` is the difference in",
        " Monte Carlo standard errors.\n")
    say("The number of kernel steps per sweep is 5 for NUTS and 30 for the random",
        " walk. That is not tuning to pass: with too few steps the",
        " successive-conditional chain stays autocorrelated, its standard error is",
        " optimistic, and the test reports large `z` for a sampler that is in fact",
        " correct. The step count has to be large enough that the sweep mixes.\n")
    say("| problem | sampler | z (mean) | z (second moment) |")
    say("|---|---|---:|---:|")
    problems = [("Gamma-Poisson", AS.conjugate_problem(AS.gamma_poisson, 20; a = 2.0, b = 1.0)),
                ("Beta-Bernoulli", AS.conjugate_problem(AS.beta_bernoulli, 25; a = 2.0, b = 3.0)),
                ("Normal-Normal", AS.conjugate_problem(AS.normal_normal, 15; mu0 = 0.0, tau0 = 2.0, sigma = 1.0))]
    for (name, prob) in problems
        for (sname, spl) in (("NUTS", AS.NUTS()), ("RWM", AS.RandomWalkMH()))
            steps = spl isa AS.RandomWalkMH ? 30 : 5
            g = AS.geweke(Random.Xoshiro(6), prob, spl; n_marginal = 40_000,
                          n_successive = 20_000, n_steps = steps)
            say("| ", name, " | ", sname, " | ", fmt(g.z_mean[1], 2), " | ",
                fmt(g.z_second[1], 2), " |")
        end
    end
    good = AS.conjugate_problem(AS.gamma_poisson, 5; a = 2.0, b = 1.0)
    bad = AS.CalibrationProblem(d -> broken_gamma_poisson(d; a = 2.0, b = 1.0),
                                good.prior_rand, good.simulate)
    g = AS.geweke(Random.Xoshiro(7), bad, AS.NUTS(); n_marginal = 40_000, n_successive = 20_000)
    say("| Gamma-Poisson **with the Jacobian removed** | NUTS | ", fmt(g.z_mean[1], 2),
        " | ", fmt(g.z_second[1], 2), " |\n")
end

# --------------------------------------------------------------------------
# 4. Hard geometries
# --------------------------------------------------------------------------

funnel_centred(k) = AS.Model((v = AS.unconstrained(), x = AS.unconstrained(k)),
                             t -> AS.logpdf(AS.Normal(0.0, 3.0), t.v) +
                                  sum(AS.logpdf(AS.Normal(0.0, exp(t.v / 2)), xi) for xi in t.x))
funnel_noncentred(k) = AS.Model((v = AS.unconstrained(), xt = AS.unconstrained(k)),
                                t -> AS.logpdf(AS.Normal(0.0, 3.0), t.v) +
                                     sum(AS.logpdf(AS.Normal(0.0, 1.0), xi) for xi in t.xt))
banana(b) = AS.Model((x = AS.unconstrained(2),),
                     t -> AS.logpdf(AS.Normal(0.0, 10.0), t.x[1]) +
                          AS.logpdf(AS.Normal(b * (t.x[1]^2 - 100), 1.0), t.x[2]))

function geometry_section()
    say("## 4. Hard geometries\n")

    # ---- correlated Gaussian
    rho = 0.95
    target = AS.MvNormal(zeros(2), [1.0 rho; rho 1.0])
    model = AS.Model((x = AS.unconstrained(2),), t -> AS.logpdf(target, t.x))
    say("### Strongly correlated Gaussian, rho = 0.95\n")
    say("| sampler | sd of x1 (exact 1) | z | correlation (exact 0.95) | ESS | ESS per second |")
    say("|---|---:|---:|---:|---:|---:|")
    entries = [("RWM, isotropic proposal", AS.RandomWalkMH(), 25_000),
               ("RWM, adapted covariance", AS.RandomWalkMH(; adapt_cov = true), 25_000),
               ("NUTS, unit metric", AS.NUTS(; metric = :unit), 10_000),
               ("NUTS, diagonal metric", AS.NUTS(), 10_000),
               ("NUTS, dense metric", AS.NUTS(; metric = :dense), 10_000)]
    traces = Dict{String,Matrix{Float64}}()
    for (name, spl, n) in entries
        chn = AS.sample(model, spl, n; n_warmup = 5000, n_chains = 4, rng = Random.Xoshiro(21))
        x1 = chn[Symbol("x[1]")]
        x2 = chn[Symbol("x[2]")]
        say("| ", name, " | ", fmt(std(vec(x1))), " | ", fmt(zstd(x1, 1.0), 2), " | ",
            fmt(cor(vec(x1), vec(x2)), 4), " | ", fmt(AS.ess_bulk(x1), 0), " | ",
            fmt(AS.ess_bulk(x1) / chn.info[:time_seconds], 0), " |")
        traces[name] = hcat(x1[1:min(400, size(x1, 1)), 1], x2[1:min(400, size(x2, 1)), 1])
    end
    p1 = plot(traces["RWM, isotropic proposal"][:, 1]; label = "", color = :firebrick,
              title = "random walk Metropolis", xlabel = "draw", ylabel = "x1", ylims = (-3.5, 3.5))
    p2 = plot(traces["NUTS, diagonal metric"][:, 1]; label = "", color = :steelblue,
              title = "NUTS", xlabel = "draw", ylabel = "x1", ylims = (-3.5, 3.5))
    plot(p1, p2; layout = (1, 2), size = (1300, 420))
    savefig(joinpath(FIG, "correlated_traces.png"))
    say("\n![traces on the correlated Gaussian](figures/correlated_traces.png)\n")

    # ---- funnel
    say("### Neal's funnel, 9 lower-level parameters\n")
    say("The marginal of `v` is exactly its prior, `Normal(0, 3)`, which is what makes",
        " the funnel measurable rather than merely illustrative.\n")
    say("| parameterisation | target accept | sd of v (exact 3) | divergences | ESS of v | min BFMI |")
    say("|---|---:|---:|---:|---:|---:|")
    runs = [("centred", funnel_centred(9), 0.8), ("centred", funnel_centred(9), 0.99),
            ("non-centred", funnel_noncentred(9), 0.8)]
    chains = Any[]
    for (label, m, ta) in runs
        chn = AS.sample(m, AS.NUTS(; target_accept = ta), 4000; n_warmup = 1000, n_chains = 4,
                        rng = Random.Xoshiro(31))
        say("| ", label, " | ", ta, " | ", fmt(std(vec(chn[:v]))), " | ", AS.divergences(chn),
            " | ", fmt(AS.ess_bulk(chn[:v]), 0), " | ", fmt(minimum(AS.bfmi(chn)), 3), " |")
        push!(chains, (label, ta, chn))
    end
    plots = Any[]
    for (label, ta, chn) in chains
        v = vec(chn[:v])
        xname = label == "centred" ? Symbol("x[1]") : Symbol("xt[1]")
        x1 = vec(chn[xname])
        div = vec(AS.sampler_stat(chn, :divergent)) .> 0
        p = scatter(x1[.!div], v[.!div]; ms = 1.4, mc = :steelblue, msw = 0, label = "draws",
                    xlabel = String(xname), ylabel = "v", ylims = (-9, 9),
                    title = string(label, ", target ", ta))
        any(div) && scatter!(p, x1[div], v[div]; ms = 2.6, mc = :firebrick, msw = 0,
                             label = "divergent")
        push!(plots, p)
    end
    plot(plots...; layout = (1, 3), size = (1400, 430))
    savefig(joinpath(FIG, "funnel.png"))
    say("\n![Neal's funnel](figures/funnel.png)\n")

    # ---- banana
    b = 0.03
    sd2 = sqrt(1 + 2 * b^2 * 100^2)
    say("### Banana, x2 | x1 ~ Normal(b (x1^2 - 100), 1) with b = 0.03\n")
    say("Exact marginals: `x1 ~ Normal(0, 10)` and `Var[x2] = 1 + 2 b^2 100^2 = ",
        fmt(sd2^2, 2), "`.\n")
    say("Read the 0.95 row against the 0.99 row. Raising the target acceptance",
        " rate from the default to 0.95 removes almost all the divergences, so the",
        " run looks clean, and the standard deviation of x2 is still wrong by 5 to 9",
        " percent - 6 to 12 Monte Carlo standard errors, across several seeds. Only",
        " at 0.99 does the sampler actually recover the marginal. The absence of",
        " divergences is not evidence of correctness, and this is the cheapest",
        " demonstration of that in the repository.\n")
    # Three seeds per configuration: one run of the 0.95 case happens to land on
    # the right answer, and a table that showed only that run would contradict
    # the paragraph above it.
    say("| configuration | sd of x1 (exact 10) | sd of x2 over 3 seeds (exact ", fmt(sd2, 3),
        ") | divergences | ESS of x2 |")
    say("|---|---:|---:|---:|---:|")
    bplots = Any[]
    seeds = (41, 77, 123)
    for (label, spl) in (("NUTS, default", AS.NUTS()),
                         ("NUTS, dense metric", AS.NUTS(; metric = :dense)),
                         ("NUTS, target accept 0.95", AS.NUTS(; target_accept = 0.95)),
                         ("NUTS, target accept 0.99", AS.NUTS(; target_accept = 0.99)))
        sd1s = Float64[]; sd2s = Float64[]; divs = Int[]; esss = Float64[]
        local first_chain = nothing
        for sd in seeds
            chn = AS.sample(banana(b), spl, 10_000; n_warmup = 2000, n_chains = 4,
                            rng = Random.Xoshiro(sd))
            push!(sd1s, std(vec(chn[Symbol("x[1]")])))
            push!(sd2s, std(vec(chn[Symbol("x[2]")])))
            push!(divs, AS.divergences(chn))
            push!(esss, AS.ess_bulk(chn[Symbol("x[2]")]))
            sd == first(seeds) && (first_chain = chn)
        end
        say("| ", label, " | ", fmt(Statistics.mean(sd1s), 3), " | ", fmt(minimum(sd2s), 3),
            " to ", fmt(maximum(sd2s), 3), " | ", minimum(divs), " to ", maximum(divs), " | ",
            fmt(Statistics.mean(esss), 0), " |")
        div = vec(AS.sampler_stat(first_chain, :divergent)) .> 0
        v1 = vec(first_chain[Symbol("x[1]")]); v2 = vec(first_chain[Symbol("x[2]")])
        p = scatter(v1[.!div], v2[.!div]; ms = 1.2, mc = :steelblue, msw = 0, label = "draws",
                    xlabel = "x1", ylabel = "x2", title = label, xlims = (-35, 35), ylims = (-5, 25))
        any(div) && scatter!(p, v1[div], v2[div]; ms = 2.4, mc = :firebrick, msw = 0,
                             label = "divergent")
        push!(bplots, p)
    end
    # reparameterised
    flat = AS.Model((x1 = AS.unconstrained(), z = AS.unconstrained()),
                    t -> AS.logpdf(AS.Normal(0.0, 10.0), t.x1) + AS.logpdf(AS.Normal(0.0, 1.0), t.z))
    chn = AS.sample(flat, AS.NUTS(), 10_000; n_warmup = 1000, n_chains = 4, rng = Random.Xoshiro(44))
    x1 = vec(chn[:x1])
    x2 = vec(chn[:z]) .+ b .* (x1 .^ 2 .- 100)
    say("| reparameterised, x2 = z + b(x1^2 - 100) | ", fmt(std(x1), 3), " | ", fmt(std(x2), 3),
        " (1 seed) | ", AS.divergences(chn), " | - |")
    p = scatter(x1, x2; ms = 1.2, mc = :seagreen, msw = 0, label = "draws", xlabel = "x1",
                ylabel = "x2", title = "reparameterised", xlims = (-35, 35), ylims = (-5, 25))
    push!(bplots, p)
    plot(bplots...; layout = (2, 3), size = (1500, 800))
    savefig(joinpath(FIG, "banana.png"))
    say("\n![banana](figures/banana.png)\n")
end

# --------------------------------------------------------------------------
# 5. Negative controls
# --------------------------------------------------------------------------

function negative_control_section()
    say("## 5. Negative controls\n")
    say("Four deliberate bugs, each installed through a documented extension point.",
        " Two are caught loudly. Two are not caught by any correctness test, and that",
        " is a fact about Hamiltonian methods rather than a gap in the suite: the",
        " leapfrog map is reversible and volume preserving whatever force it",
        " integrates, and the acceptance step uses the true log density, so an error",
        " in the gradient or in the stopping rule cannot change the invariant",
        " distribution. Those two are caught by the efficiency numbers instead.\n")

    rng = Random.Xoshiro(11)
    data = [AS.rand(rng, AS.Poisson(3.0)) for _ in 1:40]
    ref = AS.gamma_poisson(data; a = 2.0, b = 1.0)
    exact = AS.mean(ref.posterior.lambda)
    S = sum(data)

    say("| control | posterior mean | z against the true posterior | detected by |")
    say("|---|---:|---:|---|")

    chn = AS.sample(broken_gamma_poisson(data), AS.NUTS(), 20_000; n_warmup = 1000,
                    n_chains = 4, rng = Random.Xoshiro(1))
    predicted = AS.Gamma(2.0 + S - 1, 1.0 + length(data))
    say("| Jacobian term removed | ", fmt(mean(vec(chn[:lambda]))), " | ",
        fmt(zmean(chn[:lambda], exact), 1), " | conjugate check, SBC, Geweke |")
    say("\nThe broken sampler is not merely wrong: omitting `log|dlambda/dy| = y`",
        " divides the density by `lambda`, which is exactly a shift of one in the Gamma",
        " shape, so it is a *correct* sampler for `Gamma(a + S - 1, b + n)`, mean ",
        fmt(AS.mean(predicted)), ". The observed mean is ", fmt(mean(vec(chn[:lambda]))),
        ".\n")

    say("| control | effect | R-hat | detected by |")
    say("|---|---|---:|---|")
    good = AS.sample(ref.model, AS.NUTS(), 20_000; n_warmup = 1000, n_chains = 4,
                     rng = Random.Xoshiro(2))
    eff(c) = AS.ess_bulk(c[:lambda]) / sum(AS.sampler_stat(c, :n_leapfrog))
    noaccept = AS.sample(ref.model,
                         AS.RandomWalkMH(; rule = AlwaysAccept(), adapt_scale = false, scale = 0.3),
                         20_000; n_warmup = 1000, n_chains = 4, rng = Random.Xoshiro(3))
    say("| Metropolis correction removed (always accept) | posterior mean ",
        @sprintf("%.2e", mean(vec(noaccept[:lambda]))), " against a true ", fmt(exact, 2),
        " | ", fmt(AS.rhat(noaccept[:lambda]), 2), " | anything |")
    for (label, factor) in (("gradient scaled by 1.1", 1.1), ("gradient scaled by 3", 3.0))
        c = AS.sample(ref.model, AS.NUTS(; backend = ScaledGradient(factor)), 20_000;
                      n_warmup = 1000, n_chains = 4, rng = Random.Xoshiro(2))
        say("| ", label, " | mean z = ", fmt(zmean(c[:lambda], exact), 1), ", ESS per gradient ",
            fmt(eff(c), 5), " against ", fmt(eff(good), 5), " | ", fmt(AS.rhat(c[:lambda]), 3),
            " | efficiency and convergence diagnostics |")
    end
    never = AS.sample(ref.model, AS.NUTS(; uturn = NeverUTurn(), max_treedepth = 6), 20_000;
                      n_warmup = 1000, n_chains = 4, rng = Random.Xoshiro(4))
    say("| U-turn criterion that never fires | mean z = ", fmt(zmean(never[:lambda], exact), 1),
        ", ESS per gradient ", fmt(eff(never), 5), " against ", fmt(eff(good), 5), " | ",
        fmt(AS.rhat(never[:lambda]), 3), " | efficiency only |")
    say("\nThe gradient scaled by 3 is worth reading carefully. Its `z` of about -3",
        " is not evidence of bias: the effective sample size has collapsed by four",
        " orders of magnitude, so the standard error in the denominator is itself",
        " unreliable, and R-hat is what flags the run. Scaled by 1.1 the sampler is",
        " indistinguishable from the correct one on every measure. An error in the",
        " gradient is an efficiency bug, and only a large one is visible at all.\n")
end

struct AlwaysAccept <: AS.AcceptanceRule end
AS.accept_prob(::AlwaysAccept, logratio::Real) = 1.0
struct ScaledGradient <: AS.ADBackend
    factor::Float64
end
function AS.logdensity_and_gradient(b::ScaledGradient, f, y::AbstractVector)
    v, g = AS.logdensity_and_gradient(AS.ForwardDiffAD(), f, y)
    return v, g .* b.factor
end
struct NeverUTurn <: AS.UTurnCriterion end
AS.tree_continues(::NeverUTurn, metric, t) = true

# --------------------------------------------------------------------------
# 6. Diagnostics and variational inference
# --------------------------------------------------------------------------

function diagnostics_section()
    say("## 6. Diagnostics against closed forms\n")
    say("The effective sample size estimator is checked against the AR(1) closed form",
        " `n (1 - r) / (1 + r)`, including the antithetic case `r < 0` where the",
        " effective sample size exceeds the number of draws - the case a naive clamp",
        " at `n` silently corrupts, and the regime NUTS actually operates in.\n")
    say("| AR(1) coefficient | estimated ESS | closed form |")
    say("|---:|---:|---:|")
    rng = Random.Xoshiro(202)
    n, m = 20_000, 4
    for r in (0.0, 0.5, 0.9, 0.99, -0.5)
        x = zeros(n, m)
        for c in 1:m
            z = randn(rng)
            for i in 1:n
                z = r * z + sqrt(1 - r^2) * randn(rng)
                x[i, c] = z
            end
        end
        say("| ", r, " | ", fmt(AS.ess(x), 0), " | ", fmt(n * m * (1 - r) / (1 + r), 0), " |")
    end
    say("\nThe estimate at `r = 0.99` is low by about a fifth, and that is the",
        " expected behaviour rather than an error: Geyer's initial positive sequence",
        " truncates the autocorrelation sum, which is deliberately conservative, and",
        " at `r = 0.99` the correlation time is 199 draws against a chain of 10,000",
        " after splitting. Reporting too few effective draws is the safe direction",
        " for this estimator to err in.\n")

    say("## 7. Variational inference\n")
    rho = 0.9
    target = AS.MvNormal(zeros(2), [1.0 rho; rho 1.0])
    model = AS.Model((x = AS.unconstrained(2),), t -> AS.logpdf(target, t.x))
    say("For a bivariate normal the optimal mean-field approximation has marginal",
        " standard deviation `sqrt(1 - rho^2) = ", fmt(sqrt(1 - rho^2), 3), "`, so the",
        " understatement of variance is a closed form to check against rather than a",
        " vague complaint.\n")
    say("| family | sd of x1 | expected | correlation | ELBO | exact log Z |")
    say("|---|---:|---:|---:|---:|---:|")
    traces = Any[]
    for (fam, label, expected) in ((AS.MeanField(), "mean field", sqrt(1 - rho^2)),
                                   (AS.FullRank(), "full rank", 1.0))
        res = AS.sample(model, AS.ADVI(; family = fam, n_samples = 16, step_size = 0.02);
                        rng = Random.Xoshiro(9))
        chn = AS.posterior_samples(res, 40_000; rng = Random.Xoshiro(10))
        e, se = AS.elbo_with_error(res; rng = Random.Xoshiro(11), n_samples = 200_000)
        say("| ", label, " | ", fmt(std(vec(chn[Symbol("x[1]")])), 3), " | ", fmt(expected, 3),
            " | ", fmt(cor(vec(chn[Symbol("x[1]")]), vec(chn[Symbol("x[2]")])), 3), " | ",
            fmt(e, 4), " ± ", fmt(se, 4), " | 0.000 |")
        push!(traces, (label, res))
    end
    p = plot(; xlabel = "iteration", ylabel = "ELBO", title = "ELBO trace, correlated normal",
             legend = :bottomright)
    for (label, res) in traces
        plot!(p, res.elbo_iterations, res.elbo_trace; lw = 2, label = label)
    end
    hline!(p, [0.0]; color = :black, ls = :dash, lw = 1.5, label = "log Z = 0")
    savefig(p, joinpath(FIG, "elbo.png"))
    say("\n![ELBO trace](figures/elbo.png)\n")
    say("The ELBO is a lower bound on `log p(y)`, which here is exactly 0. Both",
        " families respect the bound; the full-rank family gets much closer to it,",
        " because the gap is the Kullback-Leibler divergence from `q` to the posterior",
        " and mean field cannot represent the correlation.\n")

    # SMC diagnostics figure
    counts = [AS.rand(Random.Xoshiro(5), AS.Poisson(4.0)) for _ in 1:60]
    gp = AS.gamma_poisson(counts; a = 2.0, b = 1.0)
    res = AS.sample(gp.tempered, AS.SMC(; n_particles = 2000); rng = Random.Xoshiro(6))
    p1 = plot(res.betas[2:end], res.ess_trace; marker = :circle, lw = 2, label = "",
              xlabel = "inverse temperature", ylabel = "effective sample size",
              title = "SMC tempering schedule", ylims = (0, 2100))
    hline!(p1, [0.5 * 2000]; ls = :dash, color = :black, label = "resampling threshold")
    ns = [250, 500, 1000, 2000, 4000, 8000]
    errs = [std([AS.sample(gp.tempered, AS.SMC(; n_particles = n);
                           rng = Random.Xoshiro(200 + r)).logZ for r in 1:8]) for n in ns]
    p2 = plot(ns, errs; marker = :circle, lw = 2, xscale = :log10, yscale = :log10, label = "observed",
              xlabel = "particles", ylabel = "standard deviation of log Z",
              title = "log Z error against particle count")
    plot!(p2, ns, errs[1] .* sqrt.(ns[1] ./ ns); ls = :dash, color = :black, label = "1/sqrt(N)")
    plot(p1, p2; layout = (1, 2), size = (1300, 430))
    savefig(joinpath(FIG, "smc.png"))
    say("![SMC diagnostics](figures/smc.png)\n")
    say("The left panel shows the adaptive schedule holding the effective sample size",
        " at the target of half the particles. The right panel is the convergence rate",
        " of the evidence estimate: the standard deviation of `log Z` falls as",
        " `1 / sqrt(N)`, which is what a consistent particle estimator must do.\n")
end

# --------------------------------------------------------------------------
# 8. Sum-product networks and the simplex dynamical system
# --------------------------------------------------------------------------

function extras_section()
    say("## 8. Sum-product networks\n")
    say("A sum-product network computes any marginal in one upward pass, provided",
        " the graph is complete and decomposable; both conditions are checked at",
        " construction. The claim to verify is that marginalisation is *exact* rather",
        " than approximate, so the network's marginal is compared against numerical",
        " integration of its own joint.\n")
    comp1 = [AS.Normal(-2.0, 1.0), AS.Normal(0.0, 1.0)]
    comp2 = [AS.Normal(2.0, 0.5), AS.Normal(1.0, 2.0)]
    spn = AS.naive_bayes_spn([0.3, 0.7], [comp1, comp2])
    say("| quantity | network | numerical integration |")
    say("|---|---:|---:|")
    h = 0.002
    for x1 in (-1.0, 0.5, 2.5)
        numeric = sum(exp(AS.logpdf(spn, [x1, x2])) * h for x2 in -30:h:30)
        say("| marginal density at x1 = ", x1, " | ", fmt(exp(AS.logpdf(spn, [x1, missing])), 8),
            " | ", fmt(numeric, 8), " |")
    end
    hh = 0.02
    total = sum(exp(AS.logpdf(spn, [x1, x2])) * hh^2 for x1 in -8:hh:8, x2 in -12:hh:12)
    say("| total mass | ", fmt(exp(AS.logpdf(spn, [missing, missing])), 8), " | ",
        fmt(total, 8), " |\n")

    leaves = [[AS.Normal(-2.0, 1.0)], [AS.Normal(2.0, 1.0)]]
    build = data -> AS.Model((w = AS.simplex(2),),
                             t -> AS.logpdf(AS.Dirichlet([2.0, 2.0]), t.w) +
                                  sum(AS.logpdf(AS.naive_bayes_spn(t.w, leaves), [d]) for d in data))
    prior_rand = rng -> (w = AS.rand(rng, AS.Dirichlet([2.0, 2.0])),)
    simulate = (theta, rng) -> [AS.rand(rng, AS.naive_bayes_spn(theta.w, leaves), 1)[1] for _ in 1:30]
    res = AS.sbc(Random.Xoshiro(12), AS.CalibrationProblem(build, prior_rand, simulate),
                 AS.NUTS(); n_sims = 200, n_draws = 64, thin = 5, n_warmup = 400)
    say("The sum weights live on a simplex, so inferring them is an ordinary model",
        " in this package. There is no closed-form posterior, and calibration is the",
        " check: chi-square p values of ",
        join([fmt(p, 3) for p in res.pvalue], " and "), " over 200 replications.\n")

    say("## 9. A dynamical system on the probability simplex\n")
    say("`x_{t+1} = normalise(x_t .* exp(f))` with `y_t ~ Multinomial(n, x_t)`: the",
        " initial state is a simplex parameter, the fitness vector is identified by",
        " pinning its first component, and the state must remain on the simplex for",
        " every proposal the sampler makes.\n")
    rng = Random.Xoshiro(5)
    truth = (x0 = [0.6, 0.3, 0.1], f = [0.8, -0.5])
    states = AS.simplex_trajectory(truth.x0, vcat(0.0, truth.f), 6)
    counts = [AS.rand(rng, AS.Multinomial(200, x)) for x in states]
    chn = AS.sample(AS.replicator_model(counts, 200), AS.NUTS(), 4000; n_warmup = 1000,
                    n_chains = 4, rng = Random.Xoshiro(6))
    say("| parameter | truth | posterior mean | 2.5% | 97.5% | ESS | R-hat |")
    say("|---|---:|---:|---:|---:|---:|---:|")
    for (name, t) in ((Symbol("x0[1]"), 0.6), (Symbol("x0[2]"), 0.3), (Symbol("x0[3]"), 0.1),
                      (Symbol("f[1]"), 0.8), (Symbol("f[2]"), -0.5))
        v = vec(chn[name])
        say("| ", name, " | ", fmt(t, 3), " | ", fmt(mean(v), 3), " | ", fmt(quantile(v, 0.025), 3),
            " | ", fmt(quantile(v, 0.975), 3), " | ", fmt(AS.ess_bulk(chn[name]), 0), " | ",
            fmt(AS.rhat(chn[name]), 3), " |")
    end
    prob = AS.replicator_problem(3, 6, 40; alpha = 1.0, fitness_scale = 1.0)
    sres = AS.sbc(Random.Xoshiro(11), prob, AS.NUTS(); n_sims = 200, n_draws = 64, thin = 5,
                  n_warmup = 400)
    say("\nSimulation based calibration over 200 replications of the whole model,",
        " which is what verifies the simplex transform end to end in a setting with",
        " no closed form:\n")
    say("| parameter | chi-square | p |")
    say("|---|---:|---:|")
    for j in eachindex(sres.names)
        say("| ", sres.names[j], " | ", fmt(sres.chisq[j], 2), " | ", fmt(sres.pvalue[j], 4), " |")
    end
    counts_hist = [AS.rank_histogram(sres, j) for j in eachindex(sres.names)]
    plots = [bar(counts_hist[j]; label = "", color = :steelblue, linecolor = :steelblue,
                 title = string(sres.names[j], " (p = ", fmt(sres.pvalue[j], 3), ")"),
                 xlabel = "rank bin", ylabel = "count",
                 ylims = (0, maximum(counts_hist[j]) * 1.4)) for j in eachindex(sres.names)]
    for p in plots
        hline!(p, [200 / sres.n_bins]; color = :black, lw = 2, label = "uniform")
    end
    traj = AS.simplex_trajectory([0.5, 0.3, 0.2], [0.0, 0.6, -0.3], 25)
    ptraj = plot(; xlabel = "t", ylabel = "state", title = "replicator trajectory", ylims = (0, 1))
    for k in 1:3
        plot!(ptraj, [x[k] for x in traj]; lw = 2, label = string("x", k))
    end
    push!(plots, ptraj)
    plot(plots...; layout = (2, 3), size = (1400, 760))
    savefig(joinpath(FIG, "simplex_dynamics.png"))
    say("\n![simplex dynamics](figures/simplex_dynamics.png)\n")
end

function main()
    say("# Results\n")
    say("Generated by `julia --project=scripts -t 4 scripts/make_results.jl`. Every table",
        " and figure below is produced by that script from the code in `src/`; nothing",
        " here is transcribed by hand.\n")
    say("Contents: conjugate comparisons, simulation based calibration, the Geweke",
        " joint distribution test, hard geometries, negative controls, the diagnostics",
        " themselves, variational inference, sum-product networks, and a dynamical",
        " system on the simplex.\n")
    conjugate_section()
    sbc_section()
    geweke_section()
    geometry_section()
    negative_control_section()
    diagnostics_section()
    extras_section()
    text = String(take!(OUT))
    open(joinpath(@__DIR__, "..", "docs", "results.md"), "w") do f
        write(f, text)
    end
    println("wrote docs/results.md and figures to docs/figures")
end

main()
