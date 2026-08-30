# Progress reporting, without choosing a display for the user.
#
# A run that prints nothing for ten minutes is indistinguishable from a run that
# has hung, and the hard geometries in this repository take minutes. The obvious
# fix is a progress bar, and a library that draws one has decided that its
# caller is a terminal, which is wrong in a notebook, wrong in a log file and
# wrong inside another program.
#
# So progress goes out as log messages and the logger decides what to do with
# them. Two kinds, because they answer different questions:
#
#   * one message per update at `ProgressLevel`, carrying a fraction and a
#     stable id. This is the convention ProgressLogging.jl and TerminalLoggers.jl
#     consume, so a user with either installed gets a real progress bar and this
#     package never mentions a terminal.
#
#   * a throttled `@info` line, at most one every `interval` seconds. Below-Info
#     messages are filtered out by the default logger, so without this a plain
#     session with `progress = true` would see exactly nothing, which is the
#     complaint the keyword exists to answer.

"""
    ProgressLevel

The log level progress messages are emitted at, `LogLevel(-1)`. This is the
level ProgressLogging.jl defines and TerminalLoggers.jl renders; the default
logger filters it out.
"""
const ProgressLevel = Base.CoreLogging.LogLevel(-1)

"""
    ProgressReporter(total; on, interval)

Counts work across threads and emits progress messages for it.

`total` is the number of iterations the whole run will do, counting every chain.
Chains tick it from several threads at once, so the count is atomic and the
throttle is behind a lock: a progress report is not worth a race, and it is
certainly not worth one report per thread per update.
"""
struct ProgressReporter
    on::Bool
    total::Int
    interval::Float64
    id::Symbol
    what::String
    done::Threads.Atomic{Int}
    last_report::Base.RefValue{Float64}
    lock::ReentrantLock
    t0::Float64
end

function ProgressReporter(total::Int; on::Bool = false, interval::Real = 10.0,
                          what::AbstractString = "sampling")
    t0 = time()
    p = ProgressReporter(on, total, float(interval), gensym(:assay), String(what),
                         Threads.Atomic{Int}(0), Ref(t0), ReentrantLock(), t0)
    on && @logmsg ProgressLevel p.what progress = 0.0 _id = p.id
    return p
end

"""
    tick!(p, n = 1)

Record `n` more iterations, and report if the throttle allows it.
"""
function tick!(p::ProgressReporter, n::Int = 1)
    p.on || return nothing
    completed = Threads.atomic_add!(p.done, n) + n
    now = time()
    # Read the throttle before taking the lock: on the overwhelming majority of
    # iterations there is nothing to report and contending for a lock every step
    # would be its own slowdown.
    now - p.last_report[] < p.interval && completed < p.total && return nothing
    lock(p.lock) do
        now - p.last_report[] < p.interval && completed < p.total && return nothing
        p.last_report[] = now
        frac = clamp(completed / p.total, 0.0, 1.0)
        @logmsg ProgressLevel p.what progress = frac _id = p.id
        elapsed = now - p.t0
        remaining = frac > 0 ? elapsed * (1 - frac) / frac : NaN
        @info @sprintf("%s: %d%% (%d of %d iterations, %s elapsed, %s left)",
                       p.what, round(Int, 100 * frac), completed, p.total,
                       format_duration(elapsed), format_duration(remaining))
        return nothing
    end
    return nothing
end

"""
    report!(p, fraction; completed = nothing)

Report a fraction that is known directly rather than counted.

Sequential Monte Carlo is the case this exists for. It runs until the tempering
parameter reaches one, over a number of steps that is chosen adaptively and not
known in advance, so there is no denominator to count against. The temperature
itself is the progress.
"""
function report!(p::ProgressReporter, fraction::Real; completed::Union{Nothing,Int} = nothing)
    p.on || return nothing
    now = time()
    frac = clamp(float(fraction), 0.0, 1.0)
    now - p.last_report[] < p.interval && frac < 1 && return nothing
    lock(p.lock) do
        p.last_report[] = now
        completed === nothing || (p.done[] = completed)
        @logmsg ProgressLevel p.what progress = frac _id = p.id
        elapsed = now - p.t0
        remaining = frac > 0 ? elapsed * (1 - frac) / frac : NaN
        @info @sprintf("%s: %d%% (%s elapsed, %s left)", p.what, round(Int, 100 * frac),
                       format_duration(elapsed), format_duration(remaining))
        return nothing
    end
    return nothing
end

"""
    finish!(p)

Close the progress report. A bar that is never closed stays on the screen.
"""
function finish!(p::ProgressReporter)
    p.on || return nothing
    @logmsg ProgressLevel p.what progress = "done" _id = p.id
    @info @sprintf("%s: finished %d iterations in %s", p.what, p.done[],
                   format_duration(time() - p.t0))
    return nothing
end

"""
    format_duration(seconds)

Seconds as something a person reads at a glance.
"""
function format_duration(s::Real)
    isfinite(s) || return "unknown"
    s < 1 && return @sprintf("%.1fs", s)
    s < 60 && return @sprintf("%ds", round(Int, s))
    s < 3600 && return @sprintf("%dm %02ds", div(round(Int, s), 60), rem(round(Int, s), 60))
    return @sprintf("%dh %02dm", div(round(Int, s), 3600), div(rem(round(Int, s), 3600), 60))
end
