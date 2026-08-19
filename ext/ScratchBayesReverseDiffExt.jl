module ScratchBayesReverseDiffExt

# Reverse-mode gradients. Loaded automatically when ReverseDiff is available.
#
# The whole extension is one method, which is the point of routing every
# gradient through `logdensity_and_gradient(backend, f, y)`: a new backend is a
# new method, and nothing in the samplers changes.

using ScratchBayes: ScratchBayes, ReverseDiffAD
using ReverseDiff: ReverseDiff
using DiffResults: DiffResults

function ScratchBayes.logdensity_and_gradient(b::ReverseDiffAD, f, y::AbstractVector)
    result = DiffResults.GradientResult(collect(float.(y)))
    if b.compile
        if b.tape === nothing || b.tape_length != length(y)
            tape = ReverseDiff.GradientTape(f, collect(float.(y)))
            b.tape = ReverseDiff.compile(tape)
            b.tape_length = length(y)
        end
        result = ReverseDiff.gradient!(result, b.tape, y)
    else
        result = ReverseDiff.gradient!(result, f, y)
    end
    return DiffResults.value(result), DiffResults.gradient(result)
end

end # module
