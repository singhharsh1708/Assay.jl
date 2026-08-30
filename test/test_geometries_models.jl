# Targets chosen because they are hard in different ways: strong correlation
# (wrong metric), varying scale (the funnel), and curvature (the banana). Every
# assertion here is against a closed form, so "NUTS handles it" is a measurement
# rather than an impression.

"""Neal's funnel: `v ~ N(0, 3)`, `x_i | v ~ N(0, exp(v/2))`. The marginal of `v`
is exactly the prior, which is what makes it a test."""
function funnel_model(k::Int)
    return AS.Model((v = AS.unconstrained(), x = AS.unconstrained(k)),
                    t -> AS.logpdf(AS.Normal(0.0, 3.0), t.v) +
                         sum(AS.logpdf(AS.Normal(0.0, exp(t.v / 2)), xi) for xi in t.x))
end

"""The same funnel written non-centred: `x = exp(v/2) * xtilde` with
`xtilde ~ N(0, 1)`. Identical posterior, completely different geometry."""
function funnel_noncentred(k::Int)
    return AS.Model((v = AS.unconstrained(), xt = AS.unconstrained(k)),
                    t -> AS.logpdf(AS.Normal(0.0, 3.0), t.v) +
                         sum(AS.logpdf(AS.Normal(0.0, 1.0), xi) for xi in t.xt))
end

"""Haario's banana: `x1 ~ N(0, 10^2)`, `x2 | x1 ~ N(b (x1^2 - 100), 1)`.
Marginally `E[x2] = 0` and `Var[x2] = 1 + 2 b^2 100^2`."""
function banana_model(b::Float64)
    return AS.Model((x = AS.unconstrained(2),),
                    t -> AS.logpdf(AS.Normal(0.0, 10.0), t.x[1]) +
                         AS.logpdf(AS.Normal(b * (t.x[1]^2 - 100), 1.0), t.x[2]))
end
