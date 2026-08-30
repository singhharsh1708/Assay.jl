using Aqua: Aqua

# Package hygiene, checked rather than assumed. Aqua catches the failure modes
# that do not show up in any behavioural test: method ambiguities, stale or
# undeclared dependencies, exported names that do not exist, type piracy, and
# an incomplete compat section.
@testset "package quality" begin
    Aqua.test_all(AS;
                  ambiguities = false,          # checked separately, own package only
                  persistent_tasks = false)     # the test env loads Turing, which is slow here
    @testset "no ambiguities within the package" begin
        Aqua.test_ambiguities(AS)
    end

    @testset "every exported name has a docstring" begin
        # The same check CI runs. It lived only there, so a name exported
        # without documentation stayed invisible until after a push.
        missing_docs = Symbol[]
        for name in names(AS)
            name === :Assay && continue
            doc = string(Base.Docs.doc(Base.Docs.Binding(AS, name)))
            occursin("No documentation found", doc) && push!(missing_docs, name)
        end
        @test missing_docs == Symbol[]
    end
end
