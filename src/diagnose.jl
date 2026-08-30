# Turning the diagnostics into advice.
#
# This package measures a great deal and, until now, said nothing about what any
# of it means. A user holding 427 divergences and a Bayesian fraction of missing
# information of 0.075 has to already know that this means lower the step size,
# and then reparameterise. That knowledge lives in `docs/results.md`, in prose,
# where it does not reach anyone holding a bad fit.
#
# The obvious risk is a horoscope: a function that always finds something to say
# and whose findings sound authoritative because they are printed. So every
# finding here names the number that produced it and the threshold it crossed,
# both stated in the finding itself, and a clean fit produces no findings at all
# rather than reassurance.

"""
    Finding

One thing worth saying about a fit: what was measured, what it was compared
against, and what to do about it.

`severity` is `:serious` (the draws should not be used as they are), `:warning`
(usable, with a caveat that changes what can be claimed) or `:note` (about
efficiency rather than correctness).
"""
struct Finding
    severity::Symbol
    check::Symbol
    observation::String
    threshold::String
    advice::String
end

const _SEVERITY_ORDER = Dict(:serious => 1, :warning => 2, :note => 3)

function Base.show(io::IO, f::Finding)
    label = f.severity === :serious ? "serious" : (f.severity === :warning ? "warning" : "note")
    print(io, uppercase(label), "  ", f.observation, "\n",
          "         threshold: ", f.threshold, "\n",
          "         ", f.advice)
end

"""
    Diagnosis

Everything [`diagnose`](@ref) found, most serious first. Empty means the checks
that could be run were all passed, which is not the same as the fit being right:
see [`healthy`](@ref).
"""
struct Diagnosis
    findings::Vector{Finding}
    checks_run::Vector{Symbol}
    checks_skipped::Vector{Symbol}
end

"""
    healthy(d::Diagnosis)

Whether nothing serious was found. A `true` here means no check fired, not that
the answer is right: no diagnostic can see a mode the sampler never visited, and
the tests in this package that catch a wrong answer with clean diagnostics are
there because that happens.
"""
healthy(d::Diagnosis) = !any(f -> f.severity === :serious, d.findings)

Base.isempty(d::Diagnosis) = isempty(d.findings)
Base.length(d::Diagnosis) = length(d.findings)

function Base.show(io::IO, d::Diagnosis)
    if isempty(d.findings)
        println(io, "No findings from ", length(d.checks_run), " checks.")
        print(io, "That means nothing fired, not that the answer is right: no diagnostic ",
              "sees a mode\nthe sampler never visited.")
    else
        n_serious = count(f -> f.severity === :serious, d.findings)
        println(io, length(d.findings), " finding", length(d.findings) == 1 ? "" : "s",
                " from ", length(d.checks_run), " checks, ", n_serious, " serious:\n")
        for f in d.findings
            show(io, f)
            println(io, "\n")
        end
    end
    if !isempty(d.checks_skipped)
        print(io, "\nNot checked (the sampler does not report it): ",
              join(sort(String.(d.checks_skipped)), ", "), ".")
    end
end

"""
    diagnose(chains; rhat_threshold = 1.01, ess_per_chain = 100, bfmi_threshold = 0.3,
             mcse_fraction = 0.1)

Read what a run already recorded and report what is worth acting on, most
serious first.

Every threshold has a source and is stated in the finding it produces:

  * R-hat above 1.01, from Vehtari, Gelman, Simpson, Carpenter and Bürkner (2021),
    which is where the rank-normalised split form and that threshold come from.
  * Effective sample size below 100 per chain, from the same paper: below that,
    R-hat itself is not reliable.
  * Any divergence at all. A small nonzero count is the dangerous case rather
    than the safe one, and `docs/results.md` has the measurement: on the banana
    at target acceptance 0.95 the run produced between 1 and 22 divergences and
    still got the standard deviation of `x2` wrong by 5 to 9 percent.
  * Bayesian fraction of missing information below 0.3, from Betancourt (2016).
  * Monte Carlo standard error above a tenth of the posterior standard
    deviation, which is when the reported mean is limited by the sampler rather
    than by the posterior.
  * Tree depth saturation, which costs efficiency rather than correctness.

Checks whose inputs the sampler did not record are skipped and listed, not
silently passed.
"""
function diagnose(c::Chains; rhat_threshold::Real = 1.01, ess_per_chain::Real = 100,
                  bfmi_threshold::Real = 0.3, mcse_fraction::Real = 0.1)
    findings = Finding[]
    run = Symbol[]
    skipped = Symbol[]
    n_chains = nchains(c)
    n_total = ndraws(c) * n_chains
    ess_floor = ess_per_chain * n_chains

    # ---- chains that did not finish -------------------------------------
    push!(run, :chain_failures)
    if failed(c)
        fs = failures(c)
        push!(findings, Finding(:serious, :chain_failures,
            @sprintf("%d of %d chains did not finish; the first died at %s iteration %d with a %s",
                     length(fs), get(c.info, :n_chains_requested, n_chains + length(fs)),
                     first(fs).phase, first(fs).iteration, typeof(first(fs).error)),
            "any chain failing",
            "The surviving chains are here, but they are not a random subset: the region that " *
            "killed the others is under-represented in what is left. `failures(chains)` has the " *
            "error and the position each one died at."))
    end

    # ---- R-hat ----------------------------------------------------------
    push!(run, :rhat)
    rhats = [rhat(c.value[:, j, :]) for j in 1:nparams(c)]
    worst_r = argmax(rhats)
    if n_chains < 2
        push!(skipped, :rhat_needs_chains)
    elseif rhats[worst_r] > rhat_threshold
        sev = rhats[worst_r] > 1.05 ? :serious : :warning
        push!(findings, Finding(sev, :rhat,
            @sprintf("R-hat is %.4f for %s, the worst of %d parameters",
                     rhats[worst_r], c.names[worst_r], nparams(c)),
            @sprintf("above %.2f", rhat_threshold),
            "The chains have not converged to the same distribution. Run longer warmup first. " *
            "If it persists, the chains are in different places and the posterior is probably " *
            "multimodal, which more draws will not fix."))
    end

    # ---- effective sample size ------------------------------------------
    push!(run, :ess)
    eb = [ess_bulk(c.value[:, j, :]) for j in 1:nparams(c)]
    et = [ess_tail(c.value[:, j, :]) for j in 1:nparams(c)]
    worst_b = argmin(eb)
    if eb[worst_b] < ess_floor
        push!(findings, Finding(:serious, :ess_bulk,
            @sprintf("bulk effective sample size is %.0f for %s, from %d draws",
                     eb[worst_b], c.names[worst_b], n_total),
            @sprintf("below %d, which is %d per chain", round(Int, ess_floor), ess_per_chain),
            "R-hat is not reliable at this effective sample size, so a clean R-hat above does " *
            "not mean much either. Draw more, or fix whatever is making the chain this " *
            "autocorrelated."))
    end
    worst_t = argmin(et)
    if et[worst_t] < ess_floor
        push!(findings, Finding(:warning, :ess_tail,
            @sprintf("tail effective sample size is %.0f for %s", et[worst_t], c.names[worst_t]),
            @sprintf("below %d", round(Int, ess_floor)),
            "The central estimates may be fine and the reported 2.5% and 97.5% quantiles are " *
            "not: those are what the tail effective sample size is about."))
    end
    # bulk explored, tails not. This is the shape of a fit that looks healthy
    # and has the wrong spread.
    ratio = et ./ max.(eb, 1.0)
    worst_ratio = argmin(ratio)
    if ratio[worst_ratio] < 0.5 && et[worst_ratio] >= ess_floor
        push!(findings, Finding(:note, :tail_vs_bulk,
            @sprintf("tail effective sample size is %.0f%% of bulk for %s (%.0f against %.0f)",
                     100 * ratio[worst_ratio], c.names[worst_ratio], et[worst_ratio],
                     eb[worst_ratio]),
            "below half",
            "The sampler is moving through the middle of the distribution more easily than " *
            "through its tails, so anything about the spread is less certain than the same run " *
            "makes the mean look."))
    end

    # ---- Monte Carlo error against the posterior spread ------------------
    push!(run, :mcse)
    sds = [Statistics.std(vec(c.value[:, j, :])) for j in 1:nparams(c)]
    rel = [sds[j] > 0 ? mcse_mean(c.value[:, j, :]) / sds[j] : 0.0 for j in 1:nparams(c)]
    worst_m = argmax(rel)
    if rel[worst_m] > mcse_fraction
        push!(findings, Finding(:warning, :mcse,
            @sprintf("Monte Carlo error on the mean of %s is %.0f%% of its posterior spread",
                     c.names[worst_m], 100 * rel[worst_m]),
            @sprintf("above %.0f%%", 100 * mcse_fraction),
            "The reported mean is limited by how long the sampler ran rather than by the " *
            "posterior. More draws move this directly."))
    end

    # ---- divergences ----------------------------------------------------
    if haskey(c.stats, :divergent)
        push!(run, :divergences)
        nd = divergences(c)
        if nd > 0
            rate = nd / n_total
            sev = rate >= 0.01 ? :serious : :warning
            where_str = _divergence_location(c)
            push!(findings, Finding(sev, :divergences,
                @sprintf("%d divergent transitions out of %d draws (%.2f%%)%s",
                         nd, n_total, 100 * rate, where_str),
                "any divergence at all",
                "The integrator left the level set it was following, so those draws are biased " *
                "and so is everything computed from them. Raise `target_accept` towards 0.99 " *
                "first. If they persist, the geometry is the problem rather than the step size, " *
                "and a non-centred parameterisation is the fix. A small count is not " *
                "reassurance: on the banana in `docs/results.md`, between 1 and 22 divergences " *
                "came with a standard deviation wrong by 5 to 9 percent."))
        end
    else
        push!(skipped, :divergences)
    end

    # ---- energy ---------------------------------------------------------
    if haskey(c.stats, :energy)
        push!(run, :bfmi)
        b = bfmi(c)
        if !isempty(b) && minimum(b) < bfmi_threshold
            push!(findings, Finding(:serious, :bfmi,
                @sprintf("Bayesian fraction of missing information is %.3f in the worst of %d chains",
                         minimum(b), length(b)),
                @sprintf("below %.1f", bfmi_threshold),
                "The momentum refreshment is not moving the chain between energy levels, which " *
                "is what a heavy-tailed or funnel-shaped posterior does to Hamiltonian methods. " *
                "A smaller step size will not fix this. Reparameterise."))
        end
    else
        push!(skipped, :bfmi)
    end

    # ---- tree depth -----------------------------------------------------
    sampler = get(c.info, :sampler, nothing)
    if haskey(c.stats, :treedepth) && sampler isa NUTS
        push!(run, :treedepth)
        maxed = count(>=(sampler.max_treedepth), c.stats[:treedepth])
        if maxed > 0
            push!(findings, Finding(:note, :treedepth,
                @sprintf("%d of %d transitions hit the maximum tree depth of %d",
                         maxed, length(c.stats[:treedepth]), sampler.max_treedepth),
                "any saturation",
                "The trajectories were cut short rather than stopped by the U-turn criterion, " *
                "so the sampler is paying for steps it does not get the benefit of. This costs " *
                "efficiency, not correctness. A better metric usually helps more than raising " *
                "`max_treedepth`."))
        end
    else
        push!(skipped, :treedepth)
    end

    # ---- acceptance against its own target ------------------------------
    if haskey(c.stats, :accept_prob) && sampler !== nothing && hasproperty(sampler, :target_accept)
        push!(run, :acceptance)
        # One-sided on purpose. The first version of this check fired on a
        # conjugate normal that was correct in every other respect: NUTS
        # averaged 0.91 against a target of 0.80, because the reported number is
        # the mean Metropolis probability over a whole trajectory and on easy
        # geometry that sits comfortably above what dual averaging aimed at.
        # Flagging it was the horoscope failure this function is supposed to
        # avoid. Acceptance far below target means adaptation did not get there;
        # above target means nothing.
        a = acceptance_rate(c)
        if a < sampler.target_accept - 0.15
            push!(findings, Finding(:note, :acceptance,
                @sprintf("mean acceptance probability is %.3f against a target of %.2f",
                         a, sampler.target_accept),
                "more than 0.15 below the target",
                "Step size adaptation did not reach its target, which usually means warmup was " *
                "too short for the geometry, or that the step size is still being pulled around " *
                "by a region the chain only found late."))
        end
    else
        push!(skipped, :acceptance)
    end

    sort!(findings; by = f -> _SEVERITY_ORDER[f.severity])
    return Diagnosis(findings, run, skipped)
end

# Where the divergences are, when there are enough of them to say. A divergent
# transition is informative about the region it happened in, and the region is
# usually one parameter being extreme, which is the sentence that turns "you
# have divergences" into "look at tau".
function _divergence_location(c::Chains)
    div = c.stats[:divergent]
    mask = vec(div) .> 0
    count(mask) >= 5 || return ""
    best_j, best_z = 0, 0.0
    for j in 1:nparams(c)
        v = vec(c.value[:, j, :])
        s = Statistics.std(v)
        s > 0 || continue
        z = (Statistics.mean(v[mask]) - Statistics.mean(v[.!mask])) / s
        abs(z) > abs(best_z) && ((best_j, best_z) = (j, z))
    end
    abs(best_z) >= 0.5 || return ""
    direction = best_z > 0 ? "larger" : "smaller"
    return @sprintf(", concentrated where %s is %s than usual (%.1f posterior sd away)",
                    c.names[best_j], direction, abs(best_z))
end
