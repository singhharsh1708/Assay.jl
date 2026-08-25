# Precompilation workload.
#
# The first sample a new user takes is the slowest thing that will ever happen
# to them, because Julia compiles the whole sampler on the way. Running one
# tiny chain of each sampler at build time moves that cost into installation,
# where it is paid once and expected, rather than into the first line of their
# script, where it looks like the package is slow.
#
# The workload is deliberately small: a one parameter model, a handful of
# draws. It is compiling code paths, not producing results.

using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    data = [1, 0, 1, 1, 0]
    rng = Random.Xoshiro(1)

    @compile_workload begin
        model = Model((p = unit(),),
                      theta -> logpdf(Beta(2.0, 2.0), theta.p) +
                               loglikelihood(Bernoulli(theta.p), data))
        logdensity(model, [0.0])
        logdensity_and_gradient(model, [0.0])

        for spl in (RandomWalkMH(), HMC(; n_leapfrog = 2), NUTS(; max_treedepth = 2))
            chn = sample(model, spl, 2; n_warmup = 2, n_chains = 1, rng = rng)
            summarize(chn)
        end

        vi = sample(model, ADVI(; n_iterations = 2, check_every = 2, elbo_samples = 2); rng = rng)
        posterior_samples(vi, 2; rng = rng)

        # the transform layer, which every model touches
        for t in (positive(), unit(), simplex(3), ordered(2), corr_cholesky(2))
            y = zeros(udim(t))
            x, _ = to_constrained(t, y)
            to_unconstrained(t, x)
        end

        # and the diagnostics, which every result touches
        x = randn(rng, 40, 2)
        ess(x); rhat(x); mcse_mean(x); ess_bulk(x); ess_tail(x)
    end
end
