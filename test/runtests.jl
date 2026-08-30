using Assay
using Test, Random, Statistics, LinearAlgebra

include("testutils.jl")

@testset "Assay" begin
    include("test_utils.jl")
    include("test_densities.jl")
    include("test_wrappers.jl")
    include("test_transforms.jl")
    include("test_correlation.jl")
    include("test_model.jl")
    include("test_quality.jl")
    include("test_mh_conjugate.jl")
    include("test_diagnostics.jl")
    include("test_psis.jl")
    include("test_usability.jl")
    include("test_progress.jl")
    include("test_hmc_nuts.jl")
    include("test_smc.jl")
    include("test_advi.jl")
    include("test_ecdf.jl")
    include("test_loo_pit.jl")
    include("test_calibration.jl")
    include("test_negative_controls.jl")
    include("test_geometries.jl")
    include("test_optimise.jl")
    include("test_diagnose.jl")
    include("test_spn.jl")
    include("test_dynamics.jl")
    # Order matters here, and by minutes rather than seconds. Loading a large
    # package invalidates compiled methods, so everything that runs afterwards
    # pays to recompile. Measured on this suite: with ReverseDiff loaded before
    # Turing, Turing's own load takes 131 seconds instead of 13 and the suite
    # takes 17 minutes; with it after, 7. Without these two files at all it is
    # under 4. Heavy dependencies go last, and ReverseDiff after Turing.
    include("test_interop.jl")
    include("test_turing.jl")
    include("test_external.jl")
    include("test_ad.jl")
end
