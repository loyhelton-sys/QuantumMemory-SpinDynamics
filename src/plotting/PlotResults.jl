module PlotResults

using CairoMakie

export plot_sz_final_map, compute_phase_coherence, plot_coherence_timeline

const _DETUNING_SCALE = 2π * 1e6
const _COUPLING_SCALE = 2π * 1e3

function _reshape_g_fast(values, M_delta, M_g)
    length(values) == M_delta * M_g ||
        error("The number of values must equal M_delta * M_g.")
    return permutedims(reshape(values, M_g, M_delta))
end

function _expand_singleton_axes(x, y, z; delta_halfwidth=0.05, coupling_halfwidth=0.015)
    x_plot, y_plot, z_plot = x, y, z
    if length(x_plot) == 1
        x0 = only(x_plot)
        x_plot = [x0 - delta_halfwidth, x0 + delta_halfwidth]
        z_plot = repeat(z_plot, 2, 1)
    end
    if length(y_plot) == 1
        y0 = only(y_plot)
        y_plot = [y0 - coupling_halfwidth, y0 + coupling_halfwidth]
        z_plot = repeat(z_plot, 1, 2)
    end
    return x_plot, y_plot, z_plot
end

function _plot_sz_final_map(delta_b_1d, g_b_1d, sz_per_spin, M_delta, M_g;
    filename=nothing, title="Final normalized longitudinal spin",
    delta_halfwidth=0.05, coupling_halfwidth=0.015,
    delta_window_MHz=nothing,)

    length(delta_b_1d) == M_delta || error("length(delta_b_1d) must equal M_delta.")
    length(g_b_1d) == M_g || error("length(g_b_1d) must equal M_g.")
    all(isfinite, delta_b_1d) || error("All detuning values must be finite.")
    all(isfinite, g_b_1d) || error("All coupling values must be finite.")
    all(isfinite, sz_per_spin) || error("All final Sz values must be finite.")
    delta_halfwidth > 0 || error("delta_halfwidth must be positive.")
    coupling_halfwidth > 0 || error("coupling_halfwidth must be positive.")
    if delta_window_MHz !== nothing
        delta_window_MHz isa Real && isfinite(delta_window_MHz) &&
            delta_window_MHz > 0 ||
            error("delta_window_MHz must be positive and finite, or nothing.")
    end

    delta_MHz = Float64.(delta_b_1d) ./ _DETUNING_SCALE
    coupling_kHz = Float64.(g_b_1d) ./ _COUPLING_SCALE
    sz_map = _reshape_g_fast(real.(sz_per_spin), M_delta, M_g)
    x_plot, y_plot, z_plot = _expand_singleton_axes(
        delta_MHz, coupling_kHz, sz_map;
        delta_halfwidth=delta_halfwidth,
        coupling_halfwidth=coupling_halfwidth,
    )

    fig = Figure(size=(800, 650))
    ax = Axis(fig[1, 1];
        xlabel="Detuning Δ / 2π (MHz)",
        ylabel="Coupling g / 2π (kHz)",
        title=title,
        aspect=AxisAspect(1),
    )
    hm = heatmap!(ax, x_plot, y_plot, z_plot; colorrange=(-0.5, 0.5))
    if delta_window_MHz !== nothing
        vlines!(ax, [-delta_window_MHz, delta_window_MHz];
            color=:red, linestyle=:dash, linewidth=2,
            label="Target detuning window")
        axislegend(ax; position=:rt)
    end
    Colorbar(fig[1, 2], hm; label="S_z / N_j")
    xlims!(ax, extrema(x_plot))
    ylims!(ax, extrema(y_plot))

    if filename !== nothing
        filename isa AbstractString && !isempty(filename) ||
            error("filename must be a non-empty string or nothing.")
        output_dir = dirname(filename)
        output_dir == "." || mkpath(output_dir)
        save(filename, fig)
    end
    return fig
end

"""Plot `Sz/Nj` at one saved time from solver output."""
function plot_sz_final_map(data;
    time_index=lastindex(data.t_saved), filename=nothing, kwargs...)

    required = (:M_delta, :M_g, :t_saved, :delta_b_1d, :g_b_1d, :Nj, :Sz_keep)
    for field in required
        hasproperty(data, field) || error("Simulation data must contain $(field).")
    end
    data.M_delta isa Integer && data.M_delta > 0 || error("data.M_delta must be a positive integer.")
    data.M_g isa Integer && data.M_g > 0 || error("data.M_g must be a positive integer.")
    Nt = length(data.t_saved)
    time_index isa Integer && 1 <= time_index <= Nt ||
        error("time_index must select an element of data.t_saved.")
    M = data.M_delta * data.M_g
    size(data.Sz_keep) == (M, Nt) || error("data.Sz_keep has an inconsistent size.")
    length(data.Nj) == M || error("data.Nj has an inconsistent length.")
    all(n -> n isa Real && isfinite(n) && n > 0, data.Nj) ||
        error("Every population in data.Nj must be finite and positive.")

    sz_per_spin = real.(data.Sz_keep[:, time_index]) ./ data.Nj
    return _plot_sz_final_map(
        data.delta_b_1d, data.g_b_1d, sz_per_spin, data.M_delta, data.M_g;
        filename=filename, kwargs...,
    )
end

"""Plot an already normalized final-Sz vector, as returned by the GRAPE code."""
function plot_sz_final_map(delta_b, g_b, Sz_final;
    M_delta, M_g, filename=nothing, kwargs...)

    M_delta isa Integer && M_delta > 0 || error("M_delta must be a positive integer.")
    M_g isa Integer && M_g > 0 || error("M_g must be a positive integer.")
    delta_flat = Array(delta_b)
    coupling_flat = Array(g_b)
    length(delta_flat) == M_delta * M_g || error("delta_b has an inconsistent length.")
    length(coupling_flat) == M_delta * M_g || error("g_b has an inconsistent length.")
    delta_b_1d = delta_flat[1:M_g:end]
    g_b_1d = coupling_flat[1:M_g]

    return _plot_sz_final_map(
        delta_b_1d, g_b_1d, Array(Sz_final), M_delta, M_g;
        filename=filename, kwargs...,
    )
end

"""
    compute_phase_coherence(data; detuning_max_MHz=nothing, amplitude_floor=1e-14)

Compute amplitude-weighted phase coherence. Each bin is converted to its
single-spin transverse coherence and then weighted by its population.
"""
function compute_phase_coherence(data;
    detuning_max_MHz=nothing, amplitude_floor=1e-14,)

    required = (:t_saved, :delta_b, :Nj, :Sp_keep)
    for field in required
        hasproperty(data, field) || error("Simulation data must contain $(field).")
    end
    M, Nt = size(data.Sp_keep)
    length(data.t_saved) == Nt || error("data.t_saved has an inconsistent length.")
    length(data.delta_b) == M || error("data.delta_b has an inconsistent length.")
    length(data.Nj) == M || error("data.Nj has an inconsistent length.")
    all(n -> n isa Real && isfinite(n) && n > 0, data.Nj) ||
        error("Every population in data.Nj must be finite and positive.")
    amplitude_floor isa Real && isfinite(amplitude_floor) && amplitude_floor >= 0 ||
        error("amplitude_floor must be finite and non-negative.")

    mask = trues(M)
    if detuning_max_MHz !== nothing
        detuning_max_MHz isa Real && isfinite(detuning_max_MHz) && detuning_max_MHz > 0 ||
            error("detuning_max_MHz must be finite and positive, or nothing.")
        mask .= abs.(data.delta_b) .< detuning_max_MHz * _DETUNING_SCALE
    end
    any(mask) || error("The selected detuning window contains no ensemble bins.")

    populations = Float64.(data.Nj[mask])
    weights = populations ./ sum(populations)
    per_spin = data.Sp_keep[mask, :] ./ reshape(populations, :, 1)
    coherence = zeros(Float64, Nt)
    for k in 1:Nt
        denominator = sum(weights .* abs.(per_spin[:, k]))
        coherence[k] = denominator > amplitude_floor ?
            abs(sum(weights .* per_spin[:, k])) / denominator : 0.0
    end
    return coherence
end

"""Plot a precomputed coherence timeline."""
function plot_coherence_timeline(t, coherence;
    pulse_start=nothing, pulse_end=nothing, echo_time=nothing,
    xtick_step_us=nothing, xlimits_us=nothing,
    filename=nothing, title="Phase coherence",)

    length(t) == length(coherence) || error("t and coherence must have the same length.")
    !isempty(t) || error("t and coherence cannot be empty.")
    all(isfinite, t) || error("All time values must be finite.")
    all(isfinite, coherence) || error("All coherence values must be finite.")
    xtick_step_us === nothing ||
        (xtick_step_us isa Real && isfinite(xtick_step_us) && xtick_step_us > 0) ||
        error("xtick_step_us must be positive and finite, or nothing.")
    xlimits_us === nothing ||
        (length(xlimits_us) == 2 && all(isfinite, xlimits_us) && xlimits_us[2] > xlimits_us[1]) ||
        error("xlimits_us must contain two finite increasing values, or be nothing.")

    fig = Figure(size=(850, 500))
    ax = Axis(fig[1, 1];
        xlabel="Time (μs)", ylabel="Phase coherence C(t)", title=title,
    )
    lines!(ax, Float64.(t) .* 1e6, coherence)
    pulse_start !== nothing && vlines!(ax, [pulse_start * 1e6]; linestyle=:dash)
    pulse_end !== nothing && vlines!(ax, [pulse_end * 1e6]; linestyle=:dash)
    echo_time !== nothing && vlines!(ax, [echo_time * 1e6]; linestyle=:dot)
    ylims!(ax, 0.0, 1.05)
    if xlimits_us !== nothing
        xlims!(ax, xlimits_us...)
    end
    if xtick_step_us !== nothing
        tick_start, tick_stop = xlimits_us === nothing ? extrema(Float64.(t) .* 1e6) : xlimits_us
        ax.xticks = collect(tick_start:xtick_step_us:tick_stop)
    end

    if filename !== nothing
        filename isa AbstractString && !isempty(filename) ||
            error("filename must be a non-empty string or nothing.")
        output_dir = dirname(filename)
        output_dir == "." || mkpath(output_dir)
        save(filename, fig)
    end
    return fig
end

"""Compute and plot coherence directly from solver output."""
function plot_coherence_timeline(data; detuning_max_MHz=nothing, kwargs...)
    coherence = compute_phase_coherence(data; detuning_max_MHz=detuning_max_MHz)
    return plot_coherence_timeline(data.t_saved, coherence; kwargs...)
end

end
