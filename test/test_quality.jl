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
end
