module AssayEnzymeExt

using Assay: Assay, EnzymeAD
using Enzyme: Enzyme

function Assay.logdensity_and_gradient(::EnzymeAD, f, y::AbstractVector)
    x = collect(float.(y))
    dx = zero(x)
    # Reverse mode with the seed carried in the shadow argument. Enzyme handles
    # the array mutation the simplex and correlation transforms rely on, which
    # is the reason to offer it alongside ReverseDiff.
    # Two annotations are needed, and neither is guessable from the error it
    # gives without them. `Const(f)`: the closure captures the model, and
    # Enzyme will not assume a captured value is read only unless told.
    # `set_runtime_activity`: the transforms write constant values, such as the
    # unit first entry of a correlation factor, into arrays that also carry
    # derivative data, which static activity analysis cannot prove safe.
    mode = Enzyme.set_runtime_activity(Enzyme.Reverse)
    Enzyme.autodiff(mode, Enzyme.Const(f), Enzyme.Active, Enzyme.Duplicated(x, dx))
    return f(y), dx
end

end # module
