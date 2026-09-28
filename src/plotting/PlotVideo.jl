module PlotVideo

using CairoMakie
using Printf

export plot_sz_video, plot_phase_video

const _DETUNING_SCALE = 2pi * 1e6  # rad/s -> MHz
const _COUPLING_SCALE = 2pi  # rad/s -> Hz

function _validate_video_data(data, field::Symbol)
    required = (:M_delta, :M_g, :t_saved, :delta_b_1d, :g_b_1d, :Nj, field)
    for name in required
        hasproperty(data, name) || error("Simulation data must contain $(name).")
    end

    M_delta = data.M_delta
    M_g = data.M_g
    M_delta isa Integer && M_delta > 0 || error("data.M_delta must be a positive integer.")
    M_g isa Integer && M_g > 0 || error("data.M_g must be a positive integer.")

    Nt = length(data.t_saved)
    Nt > 0 || error("data.t_saved cannot be empty.")
    length(data.delta_b_1d) == M_delta ||
        error("length(data.delta_b_1d) must equal data.M_delta.")
    length(data.g_b_1d) == M_g ||
        error("length(data.g_b_1d) must equal data.M_g.")
    length(data.Nj) == M_delta * M_g ||
        error("length(data.Nj) must equal data.M_delta * data.M_g.")
    all(n -> n isa Real && isfinite(n) && n > 0, data.Nj) ||
        error("Every population in data.Nj must be finite and positive.")

    values = getproperty(data, field)
    size(values) == (M_delta * M_g, Nt) ||
        error("data.$(field) must have size (M_delta * M_g, length(t_saved)).")

    return M_delta, M_g, Nt, values
end

function _map_coordinates(data)
    delta_MHz = Float64.(data.delta_b_1d) ./ _DETUNING_SCALE
    coupling_Hz = Float64.(data.g_b_1d) ./ _COUPLING_SCALE
    return delta_MHz, coupling_Hz
end

function _reshape_g_fast(values, M_delta, M_g)
    length(values) == M_delta * M_g || error("Map data has an inconsistent length.")
    return permutedims(reshape(values, M_g, M_delta))
end

function _expand_singleton_axes(x, y, z; delta_halfwidth=0.05, coupling_halfwidth=15.0)
    x_plot = x
    y_plot = y
    z_plot = z

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

function _record_map_video(frame_data, data;
    filename, fps, frame_step, colorbar_label,
    colormap=:viridis, colorrange=nothing, delta_halfwidth=0.05,
    coupling_halfwidth=15.0,)

    filename isa AbstractString && !isempty(filename) ||
        error("A non-empty filename is required when recording a video.")
    fps isa Real && isfinite(fps) && fps > 0 || error("fps must be finite and positive.")
    frame_step isa Integer && frame_step > 0 || error("frame_step must be a positive integer.")
    delta_halfwidth > 0 || error("delta_halfwidth must be positive.")
    coupling_halfwidth > 0 || error("coupling_halfwidth must be positive.")

    delta_MHz, coupling_Hz = _map_coordinates(data)
    z0 = frame_data(1)
    x_plot, y_plot, z0_plot = _expand_singleton_axes(
        delta_MHz, coupling_Hz, z0;
        delta_halfwidth=delta_halfwidth,
        coupling_halfwidth=coupling_halfwidth,)

    fig = Figure(size=(1000, 800))
    ax = Axis(fig[1, 1];
        xlabel="Detuning / 2pi (MHz)",
        ylabel="Coupling g / 2pi (Hz)",
        title=@sprintf("t = %.1f μs", data.t_saved[1] * 1e6),
        aspect=AxisAspect(1),)

    map_observable = Observable(z0_plot)
    heatmap_options = isnothing(colorrange) ?
        (; colormap=colormap) : (; colormap=colormap, colorrange=colorrange)
    hm = heatmap!(ax, x_plot, y_plot, map_observable; heatmap_options...)
    Colorbar(fig[1, 2], hm; label=colorbar_label)
    xlims!(ax, extrema(x_plot))
    ylims!(ax, extrema(y_plot))

    output_dir = dirname(filename)
    output_dir == "." || mkpath(output_dir)
    frames = 1:frame_step:length(data.t_saved)
    record(fig, filename, frames; framerate=fps) do k
        _, _, z_plot = _expand_singleton_axes(
            delta_MHz, coupling_Hz, frame_data(k);
            delta_halfwidth=delta_halfwidth,
            coupling_halfwidth=coupling_halfwidth,)

        map_observable[] = z_plot
        time_us = data.t_saved[k] * 1e6
        ax.title = @sprintf("t = %.1f μs", time_us)
    end

    return fig
end

"""Record the normalized longitudinal spin Sz/Nj over the ensemble."""
function plot_sz_video(data;
    filename="sz_video.mp4", fps=30, frame_step=1,
    delta_halfwidth=0.05, coupling_halfwidth=15.0,)

    M_delta, M_g, _, Sz = _validate_video_data(data, :Sz_keep)
    Nj = Float64.(data.Nj)
    frame_data = k -> _reshape_g_fast(real.(Sz[:, k]) ./ Nj, M_delta, M_g)

    return _record_map_video(frame_data, data;
        filename=filename, fps=fps,
        frame_step=frame_step,
        colorbar_label="Sz / Nj",
        colorrange=(-0.5, 0.5),
        delta_halfwidth=delta_halfwidth,
        coupling_halfwidth=coupling_halfwidth,)
end

"""Record the transverse-spin phase over the ensemble."""
function plot_phase_video(data;
    filename="phase_video.mp4", fps=30, frame_step=1,
    phase_threshold=1e-10, delta_halfwidth=0.05,
    coupling_halfwidth=15.0,)

    M_delta, M_g, _, Sp = _validate_video_data(data, :Sp_keep)
    phase_threshold isa Real && isfinite(phase_threshold) && phase_threshold >= 0 ||
        error("phase_threshold must be finite and non-negative.")
    Nj = Float64.(data.Nj)

    function frame_data(k)
        transverse = Sp[:, k]
        phase = angle.(transverse)
        phase[abs.(transverse) ./ Nj .< phase_threshold] .= NaN
        return _reshape_g_fast(phase, M_delta, M_g)
    end

    return _record_map_video(frame_data, data;
        filename=filename,
        fps=fps,
        frame_step=frame_step,
        colorbar_label="Phase (rad)",
        colormap=:hsv,
        colorrange=(-pi, pi),
        delta_halfwidth=delta_halfwidth,
        coupling_halfwidth=coupling_halfwidth,
    )
end

end
