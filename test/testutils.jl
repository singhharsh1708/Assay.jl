# Shared assertions for the verification suite.
#
# Every tolerance in this repository is stated in Monte Carlo standard errors
# rather than as a bare number, so a test that passes says something specific:
# the difference between the sampler and the closed-form answer is within what
# sampling noise can explain at the effective sample size actually achieved.

using ScratchBayes, Test, Statistics

const SB = ScratchBayes

"""
    check_mean(chain_matrix, exact_mean; nse = 4)

Posterior mean within `nse` Monte Carlo standard errors of the analytic value.
"""
function check_mean(x::AbstractMatrix, exact::Real; nse::Real = 4)
    m = mean(vec(x))
    se = SB.mcse_mean(x)
    ok = abs(m - exact) <= nse * se
    ok || @info "mean mismatch" sampled=m exact=exact mcse=se z=(m - exact) / se
    return ok
end

"""
    check_std(chain_matrix, exact_std; nse = 4)

Posterior standard deviation within `nse` standard errors. For a roughly normal
target the standard error of the sample standard deviation is
`sd / sqrt(2 * ess)`; the effective sample size of the squared deviations is
used so the check stays valid under autocorrelation.
"""
function check_std(x::AbstractMatrix, exact::Real; nse::Real = 4)
    s = std(vec(x))
    v = vec(x)
    e = SB.ess_bulk(reshape((v .- mean(v)) .^ 2, size(x)))
    se = s / sqrt(2 * max(e, 1.0))
    ok = abs(s - exact) <= nse * se
    ok || @info "sd mismatch" sampled=s exact=exact se=se z=(s - exact) / se
    return ok
end

"""
    check_quantile(chain_matrix, d, p; nse = 4)

Sample quantile against the analytic quantile of density `d`. The standard error
uses the usual `sqrt(p(1-p)/ess_tail) / pdf(q)` expression, which is what makes
the tail comparison meaningful rather than an arbitrary absolute tolerance.
"""
function check_quantile(x::AbstractMatrix, d, p::Real; nse::Real = 4)
    q_exact = SB.quantile(d, p)
    q_hat = quantile(vec(x), p)
    dens = exp(SB.logpdf(d, q_exact))
    e = max(SB.ess_tail(x), 1.0)
    se = sqrt(p * (1 - p) / e) / dens
    ok = abs(q_hat - q_exact) <= nse * se
    ok || @info "quantile mismatch" p=p sampled=q_hat exact=q_exact se=se z=(q_hat - q_exact) / se
    return ok
end

"""
    check_conjugate(chains, name, exact; nse = 4)

Mean, standard deviation and the 2.5%, 50%, 97.5% quantiles of one parameter
against its analytic posterior.
"""
function check_conjugate(chn, name::Symbol, exact; nse::Real = 4)
    x = chn[name]
    @test check_mean(x, SB.mean(exact); nse = nse)
    @test check_std(x, sqrt(SB.var(exact)); nse = nse)
    for p in (0.025, 0.5, 0.975)
        @test check_quantile(x, exact, p; nse = nse)
    end
    @test SB.rhat(x) < 1.01
end
