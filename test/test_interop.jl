using Tables: Tables
using DataFrames: DataFrame
using MCMCChains: MCMCChains

@testset "fitting into the wider ecosystem" begin
    data = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1]
    model = AS.Model((p = AS.unit(),),
                     t -> AS.logpdf(AS.Beta(2.0, 2.0), t.p) +
                          AS.loglikelihood(AS.Bernoulli(t.p), data))
    chn = AS.sample(model, AS.NUTS(), 200; n_warmup = 100, n_chains = 2,
                    rng = Random.Xoshiro(1))

    @testset "chains are a table" begin
        @test Tables.istable(typeof(chn))
        @test Tables.columnaccess(typeof(chn))
        cols = Tables.columns(chn)
        @test keys(cols) == (:chain, :draw, :p)
        @test length(cols.p) == AS.ndraws(chn) * AS.nchains(chn)
        # long format: the chain index survives rather than being folded away
        @test sort(unique(cols.chain)) == [1, 2]
        @test extrema(cols.draw) == (1, AS.ndraws(chn))
        @test cols.p[1] == chn.value[1, 1, 1]
        @test cols.p[AS.ndraws(chn) + 1] == chn.value[1, 1, 2]
        sch = Tables.schema(chn)
        @test sch.names == (:chain, :draw, :p)
    end

    @testset "and therefore a DataFrame" begin
        df = DataFrame(chn)
        @test size(df) == (AS.ndraws(chn) * AS.nchains(chn), 3)
        @test names(df) == ["chain", "draw", "p"]
        @test Statistics.mean(df.p) ≈ Statistics.mean(vec(chn[:p]))
    end

    @testset "conversion to MCMCChains carries the sampler statistics" begin
        mc = MCMCChains.Chains(chn)
        @test size(mc, 1) == AS.ndraws(chn)
        @test size(mc, 3) == AS.nchains(chn)
        @test :p in MCMCChains.names(mc)
        # the statistics land in :internals, where the ecosystem looks for them
        internals = mc.name_map[:internals]
        @test :divergent in internals
        @test :accept_prob in internals
        @test Symbol("p") in mc.name_map[:parameters]

        # their diagnostics and ours agree on the same draws, which is a check
        # on both implementations rather than only on the conversion
        their_ess = first(MCMCChains.ess(mc)[:, :ess])
        their_rhat = first(MCMCChains.rhat(mc)[:, :rhat])
        @test their_ess ≈ AS.ess_bulk(chn[:p]) rtol = 0.1
        @test their_rhat ≈ AS.rhat(chn[:p]) rtol = 0.02
    end
end
