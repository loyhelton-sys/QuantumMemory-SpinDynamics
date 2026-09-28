module PlotPulse

using CairoMakie

export plot_pulse

function _unwrap_phase(phase)
    unwrapped = Float64.(phase)
    offset = 0.0
    for k in 2:length(unwrapped)
        step = phase[k] - phase[k - 1]
        if step > π
            offset -= 2π
        elseif step < -π
            offset += 2π
        end
        unwrapped[k] += offset
    end
    return unwrapped
end

function _pulse_traces(E, dt, amplitude_threshold)
    control = ComplexF64.(E)
    amplitude = abs.(control)
    phase = _unwrap_phase(angle.(control))

    cutoff = amplitude_threshold * maximum(amplitude; init=0.0)
    defined = amplitude .> cutoff
    phase[.!defined] .= NaN

    frequency_MHz = diff(_unwrap_phase(angle.(control))) ./ (2π * dt * 1e6)
    frequency_MHz[.!(defined[1:end-1] .& defined[2:end])] .= NaN
    return amplitude, phase, frequency_MHz
end

function _validate_pulse_inputs(E, dt, t_start, amplitude_threshold)
    E isa AbstractVector && !isempty(E) || error("E must be a non-empty vector.")
    all(isfinite, E) || error("Every pulse sample must be finite.")
    dt isa Real && isfinite(dt) && dt > 0 || error("dt must be finite and positive.")
    t_start isa Real && isfinite(t_start) || error("t_start must be finite.")
    amplitude_threshold isa Real && isfinite(amplitude_threshold) &&
        0 <= amplitude_threshold < 1 ||
        error("amplitude_threshold must lie in [0, 1).")
    return nothing
end

function _mask_frequency_windows!(frequency, frequency_times, windows)
    for window in windows
        length(window) == 2 ||
            error("Each frequency exclusion window must contain (start, stop).")
        window_start, window_end = window
        window_start isa Real && window_end isa Real &&
            isfinite(window_start) && isfinite(window_end) && window_end >= window_start ||
            error("Frequency exclusion windows must have finite start <= stop.")
        frequency[window_start .<= frequency_times .<= window_end] .= NaN
    end
    return frequency
end

"""
    plot_pulse(E, dt; reference=nothing, t_start=0.0, filename=nothing, ...)

Plot pulse amplitude, unwrapped phase, and instantaneous frequency. Time is
shown in microseconds and frequency in MHz. Pass `reference` to compare two
pulses sampled on the same time grid.
"""
function plot_pulse(E, dt;
    reference=nothing,
    t_start=0.0,
    filename=nothing,
    label="Pulse",
    reference_label="Reference",
    color=:darkorange,
    reference_color=:dodgerblue,
    amplitude_threshold=1e-6,
    frequency_exclusion_windows=(),
    title="Pulse",
    figsize=(900, 700),)

    _validate_pulse_inputs(E, dt, t_start, amplitude_threshold)
    if reference !== nothing
        _validate_pulse_inputs(reference, dt, t_start, amplitude_threshold)
        length(reference) == length(E) || error("reference and E must have the same length.")
    end

    time_us = (t_start .+ (0:length(E)-1) .* dt) .* 1e6
    frequency_time_us = (t_start .+ ((0:length(E)-2) .+ 0.5) .* dt) .* 1e6
    frequency_times = t_start .+ ((0:length(E)-2) .+ 0.5) .* dt
    amplitude, phase, frequency = _pulse_traces(E, dt, amplitude_threshold)
    _mask_frequency_windows!(frequency, frequency_times, frequency_exclusion_windows)

    fig = Figure(size=figsize)
    amplitude_axis = Axis(fig[1, 1]; ylabel="Dimensionless amplitude |E|", title=title)
    phase_axis = Axis(fig[2, 1]; ylabel="Phase (rad)")
    frequency_axis = Axis(fig[3, 1]; xlabel="Time (μs)", ylabel="Frequency (MHz)")

    if reference !== nothing
        ref_amplitude, ref_phase, ref_frequency =
            _pulse_traces(reference, dt, amplitude_threshold)
        _mask_frequency_windows!(
            ref_frequency, frequency_times, frequency_exclusion_windows)
        stairs!(amplitude_axis, time_us, ref_amplitude; label=reference_label, color=reference_color)
        stairs!(phase_axis, time_us, ref_phase; label=reference_label, color=reference_color)
        lines!(frequency_axis, frequency_time_us, ref_frequency; label=reference_label, color=reference_color)
    end

    # Draw the optimized/current pulse last so it stays on top when traces overlap.
    stairs!(amplitude_axis, time_us, amplitude; label=label, color=color)
    stairs!(phase_axis, time_us, phase; label=label, color=color)
    lines!(frequency_axis, frequency_time_us, frequency; label=label, color=color)

    reference !== nothing && axislegend(amplitude_axis; position=:rt)

    linkxaxes!(amplitude_axis, phase_axis, frequency_axis)
    hidexdecorations!(amplitude_axis; grid=false)
    hidexdecorations!(phase_axis; grid=false)

    if filename !== nothing
        filename isa AbstractString && !isempty(filename) ||
            error("filename must be a non-empty string or nothing.")
        output_dir = dirname(filename)
        output_dir == "." || mkpath(output_dir)
        save(filename, fig)
    end

    return fig
end

"""Sample a continuous pulse function and plot its three control traces."""
function plot_pulse(E_of_t::Function;
    t_start, t_end, dt, kwargs...)

    t_start isa Real && isfinite(t_start) || error("t_start must be finite.")
    t_end isa Real && isfinite(t_end) && t_end > t_start ||
        error("t_end must be finite and larger than t_start.")
    dt isa Real && isfinite(dt) && dt > 0 || error("dt must be finite and positive.")

    times = collect(t_start:dt:(t_end - dt))
    isempty(times) && error("The requested time interval contains no pulse samples.")
    E = ComplexF64[E_of_t(t) for t in times]
    return plot_pulse(E, dt; t_start=t_start, kwargs...)
end

end
