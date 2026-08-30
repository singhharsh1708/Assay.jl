# The failures a caller can act on, as types rather than as message text.
#
# `error(...)` produces an `ErrorException` carrying a string, so anything
# wrapping the package that wants to respond to a bad initial point differently
# from a missing gradient backend has to match on that string, and the string is
# the part most likely to be reworded. Everything raised deliberately here has a
# type instead.
#
# Every one of them carries the position it failed at. That is the difference
# between a message saying a gradient was not finite and a message a user can
# paste back into their own log density to see it happen again.

"""
    AssayError

Base type for the errors this package raises deliberately. Catching this catches
a failure the package decided to report; it does not catch an error from inside
a user's log density, which propagates as whatever it already was.
"""
abstract type AssayError <: Exception end

# Positions are printed in full when short and truncated when not: a fifty
# dimensional vector in an error message helps nobody, and the first few
# coordinates plus the length usually identify which point it was.
function _show_position(io::IO, y::AbstractVector, k::Int = 8)
    n = length(y)
    if n <= k
        print(io, "[", join((@sprintf("%.6g", v) for v in y), ", "), "]")
    else
        print(io, "[", join((@sprintf("%.6g", y[i]) for i in 1:k), ", "),
              ", ... ] (", n, " coordinates)")
    end
end

"""
    NonDeterministicModelError

The log density returned two different values at the same point, so there is no
distribution for a sampler to target.

Carries the `position` and both values. The usual cause is randomness inside the
model closure.
"""
struct NonDeterministicModelError <: AssayError
    position::Vector{Float64}
    first::Float64
    second::Float64
end

function Base.showerror(io::IO, e::NonDeterministicModelError)
    print(io, "NonDeterministicModelError: the log density is not deterministic. Evaluating it ",
          "twice at the same point gave ", e.first, " and ", e.second,
          ". A sampler cannot target a moving distribution.\n",
          "The usual cause is randomness inside the model, such as data drawn with `rand` or ",
          "`randn` inside the closure; generate it once outside and close over it.\n",
          "Position: ")
    return _show_position(io, e.position)
end

"""
    NonFiniteDensityError

The log density, or one coordinate of its gradient, was not finite at
`position`.

`what` is `:logdensity` or `:gradient`, and `coordinate` names the offending
gradient entry, or is zero when the log density itself is at fault.
"""
struct NonFiniteDensityError <: AssayError
    position::Vector{Float64}
    value::Float64
    what::Symbol
    coordinate::Int
end

NonFiniteDensityError(y, v, what) = NonFiniteDensityError(collect(float.(y)), float(v), what, 0)

function Base.showerror(io::IO, e::NonFiniteDensityError)
    if e.what === :gradient
        print(io, "NonFiniteDensityError: coordinate ", e.coordinate,
              " of the gradient is ", e.value, ".\n",
              "A gradient that is not finite is usually a density that is flat or infinite in ",
              "that direction, or a term that has overflowed.\n")
    else
        print(io, "NonFiniteDensityError: the log density is ", e.value, ".\n",
              "If the model is correct, supply `init` explicitly; if a parameter is constrained, ",
              "check it is declared with the matching transform rather than being constrained ",
              "inside the log density.\n")
    end
    print(io, "Position: ")
    _show_position(io, e.position)
    return print(io, "\nEvaluate the log density there to see it directly.")
end

"""
    InitialisationError

No starting point with a finite log density was found by random search.

Carries how many points were tried, the box they were drawn from, and the last
point tried with its value, so the search can be repeated by hand.
"""
struct InitialisationError <: AssayError
    tries::Int
    scale::Float64
    dimension::Int
    last_position::Vector{Float64}
    last_value::Float64
end

function Base.showerror(io::IO, e::InitialisationError)
    print(io, "InitialisationError: no point with a finite log density found in ", e.tries,
          " draws from the box [-", e.scale, ", ", e.scale, "]^", e.dimension, ".\n",
          "Supply `init` explicitly, or check that the log density is finite anywhere: a ",
          "constraint enforced inside the density rather than by a transform makes almost every ",
          "point invalid.\nLast point tried had log density ", e.last_value, " at ")
    return _show_position(io, e.last_position)
end

"""
    BackendUnavailableError

A gradient backend was asked for a gradient before the package that defines it
was loaded, or has no method at all.
"""
struct BackendUnavailableError <: AssayError
    backend::DataType
    package::Union{Nothing,Symbol}
end

function Base.showerror(io::IO, e::BackendUnavailableError)
    if e.package === nothing
        print(io, "BackendUnavailableError: no gradient method for backend ", e.backend, ".")
    else
        print(io, "BackendUnavailableError: ", e.backend, " needs ", e.package,
              " to be loaded. `using ", e.package, "` brings in the package extension that ",
              "defines this method.")
    end
end
