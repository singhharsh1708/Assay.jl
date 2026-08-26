module AssayZygoteExt

using Assay: Assay, ZygoteAD
using Zygote: Zygote

function Assay.logdensity_and_gradient(::ZygoteAD, f, y::AbstractVector)
    value, back = try
        Zygote.pullback(f, y)
    catch err
        _rethrow_mutation(err)
    end
    grad = try
        first(back(one(value)))
    catch err
        _rethrow_mutation(err)
    end
    grad === nothing && error("Zygote returned no gradient: the log density does not appear to " *
                              "depend on its argument.")
    return value, collect(grad)
end

"""
Turn Zygote's mutation error into one that says what to do about it. Three
transforms in this package build their output by writing into an array, which
Zygote cannot differentiate; forward mode and Enzyme both can.
"""
function _rethrow_mutation(err)
    msg = sprint(showerror, err)
    if occursin("Mutating arrays", msg) || occursin("setindex!", msg)
        error("Zygote cannot differentiate this model, because it mutates arrays. The " *
              "`simplex`, `ordered` and `corr_cholesky` transforms build their output that " *
              "way. Use ForwardDiffAD (the default), EnzymeAD, or ReverseDiffAD instead.")
    end
    rethrow(err)
end

end # module
