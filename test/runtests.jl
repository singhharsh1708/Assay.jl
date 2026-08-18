using ScratchBayes
using Test, Random, Statistics, LinearAlgebra

include("testutils.jl")

@testset "ScratchBayes" begin
    include("test_utils.jl")
    include("test_densities.jl")
    include("test_transforms.jl")
    include("test_model.jl")
    include("test_mh_conjugate.jl")
end
