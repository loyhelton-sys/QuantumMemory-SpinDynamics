using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CUDA
using JLD2
using Printf
using SpinDynamics

include(joinpath(@__DIR__, "..", "src", "plotting", "PlotPulse.jl"))
using .PlotPulse
include(joinpath(@__DIR__, "..", "src", "plotting", "PlotResults.jl"))
using .PlotResults

CUDA.functional() || error("A functional CUDA GPU is required for RASE optimization.")

# The defaults are a full optimization run. For a shorter trial, for example:
# SPINDYNAMICS_MAX_ITER=10 julia --project=. examples/three_pulse_rase_optimization.jl
env_int(name, default) = parse(Int, get(ENV, name, string(default)))
const M_DELTA = env_int("SPINDYNAMICS_M_DELTA", 101)
const M_G = env_int("SPINDYNAMICS_M_G", 101)
const MAX_ITER = env_int("SPINDYNAMICS_MAX_ITER", 1)
const N_SUB = env_int("SPINDYNAMICS_N_SUB", 16)
const RECOVER_FROM_CHECKPOINT = false

# =============================================================
# Inhomogeneous ensemble
# =============================================================
FWHM_delta = 2π * 0.4e6
σ_delta = FWHM_delta / (2sqrt(2log(2)))
freq_cfg = (kind=:gaussian,
            FWHM=FWHM_delta,
            span_sigma=(2π * 0.5e6) / σ_delta,
            renormalize=true,)

g_mean = 2π * 100e3
FWHM_g = 2π * 10e3
σ_g = FWHM_g / (2sqrt(2log(2)))
g_cfg = (kind=:gaussian,
         mean=g_mean,
         FWHM=FWHM_g,
         span_sigma=(2π * 10e3) / σ_g,
         renormalize=true,)

ens = build_equal_weight_ensemble(freq_cfg, g_cfg, M_DELTA, M_G)
delta_b = CuArray(ens.delta_b)
g_b = CuArray(ens.g_b)
spin_weight = CuArray(ens.Nj ./ ens.N_total)


# Three-pulse control grid. The control itself is loaded from saved data below.
durations = (10e-6, 20e-6, 10e-6)
T_pulse = sum(durations)
dt = 0.05e-6
phase_amp_low_frac = 0.001
phase_amp_high_frac = 0.005
# Phase and instantaneous frequency are not meaningful when the control is
# nearly zero. A 3% plotting mask suppresses endpoint/dropout
# artifacts (notably near 0 and 40 us) without changing the optimized control.
phase_plot_frac = 0.03

# ======================================================================
# Objective: population inversion, echo coherence, and smoothness
# ======================================================================
delta_window = 2π * 0.30e6
t_dephase = 5e-6
t_echo_free = 5e-6
Sz_init = -0.499
Sz_target = 0.499
w_pop_inside = 3.2
w_pop_outside = 2.6
w_coh = 2.6
w_smooth = 0.008
η = 0.02

output_dir = joinpath(@__DIR__, "..", "Results", "three_pulse_rase_optimization")
mkpath(output_dir)
checkpoint_file = joinpath(output_dir, "checkpoint.jld2")
result_file = joinpath(output_dir, "optimized_three_pulse.jld2")

checkpoint_metadata = (
    version=1, M_delta=M_DELTA, M_g=M_G, n_sub=N_SUB,
    delta_window_MHz=delta_window / (2π * 1e6),
    durations=durations, T_pulse=T_pulse, dt=dt,
    freq_cfg=freq_cfg, g_cfg=g_cfg, delta_window=delta_window,
    t_dephase=t_dephase, t_echo_free=t_echo_free,
    Sz_init=Sz_init, Sz_target=Sz_target,
    w_pop_inside=w_pop_inside, w_pop_outside=w_pop_outside,
    w_coh=w_coh, w_smooth=w_smooth,
    phase_amp_low_frac=phase_amp_low_frac,
    phase_amp_high_frac=phase_amp_high_frac)

function checkpoint_is_compatible(saved, current)
    saved isa NamedTuple || return false
    # Only the time grid determines whether a saved control vector can be reused.
    # Ensemble, objective weights, and solver accuracy may intentionally change
    # between continuation runs.
    fields = (:durations, :T_pulse, :dt)
    return all(field -> hasproperty(saved, field) &&
        isequal(getproperty(saved, field), getproperty(current, field)), fields)
end

function report_metadata_changes(saved, current)
    saved isa NamedTuple || return
    changed = String[]
    for field in propertynames(current)
        hasproperty(saved, field) || continue
        isequal(getproperty(saved, field), getproperty(current, field)) && continue
        push!(changed, string(field, ": ", getproperty(saved, field),
            " -> ", getproperty(current, field)))
    end
    isempty(changed) || println("Continuing with changed configuration:\n  ",
        join(changed, "\n  "))
end

function load_resume_state(checkpoint_file, result_file, current_metadata)
    source_file = if RECOVER_FROM_CHECKPOINT
        isfile(checkpoint_file) ||
            error("Checkpoint recovery was requested, but no checkpoint exists: $checkpoint_file")
        checkpoint_file
    elseif isfile(result_file)
        result_file
    elseif isfile(checkpoint_file)
        println("Final result is missing; falling back to checkpoint recovery.")
        checkpoint_file
    else
        error("No existing optimization data found. Expected $checkpoint_file or $result_file. " *
              "This program only continues an existing optimization.")
    end

    saved = load(source_file)
    haskey(saved, "E") || error("Saved optimization data has no E field: $source_file")
    saved_metadata = get(saved, "checkpoint_metadata", nothing)
    checkpoint_is_compatible(saved_metadata, current_metadata) ||
        error("Saved control time grid is incompatible with the current configuration: " *
              source_file * ". Refusing to overwrite it.")
    report_metadata_changes(saved_metadata, current_metadata)

    E = ComplexF64.(saved["E"])
    expected_length = round(Int, current_metadata.T_pulse / current_metadata.dt)
    length(E) == expected_length ||
        error("Saved control has $(length(E)) samples; expected $expected_length.")

    # Every time sample is open to amplitude optimization, so the former pulse
    # junctions may fill in and the three sections may merge naturally.
    control_mask = trues(length(E))

    iteration_offset = Int(get(saved, "iter", get(saved, "completed_iterations", 0)))
    backup_file = joinpath(dirname(source_file),
        "checkpoint_before_run_iter$(iteration_offset).jld2")
    if !isfile(backup_file)
        cp(source_file, backup_file)
        println("Backed up resume data to ", backup_file)
    end
    η = Float64(0.02)
    println("Resuming from ", source_file, " at iteration ", iteration_offset,
        " with configured eta=", η,
        " (saved eta=", get(saved, "η", "not available"), ")")
    return E, control_mask, iteration_offset, η
end

E, control_mask, iteration_offset, η =
    Base.invokelatest(load_resume_state, checkpoint_file, result_file, checkpoint_metadata)

relative_amplitude = abs.(E) ./ maximum(abs, E)
phase_gate = clamp.((relative_amplitude .- phase_amp_low_frac) ./
    (phase_amp_high_frac - phase_amp_low_frac), 0.0, 1.0)
phase_weights = phase_gate.^2 .* (3 .- 2 .* phase_gate)

# =============================================================
# Current objective values before optimization
# =============================================================
E_before_optimization = copy(E)

J0, Jpop_inside0, Jpop_outside0, Jcoh0, Jsmooth0, _, _ = rase_objective(
    E_before_optimization, delta_b, g_b, spin_weight, dt;
    delta_window=delta_window,
    t_dephase=t_dephase,
    t_echo_free=t_echo_free,
    Sz_init=Sz_init,
    Sz_target=Sz_target,
    w_pop_inside=w_pop_inside,
    w_pop_outside=w_pop_outside,
    w_coh=w_coh,
    w_smooth=w_smooth,
    n_sub=N_SUB,
    control_mask=control_mask,
    phase_weights=phase_weights,)

# =============================================================
# Run optimization
# =============================================================
E, J, Jpop_inside, Jpop_outside, Jcoh, Jsmooth = optimize_rase!(
    E, delta_b, g_b, spin_weight, dt;
    delta_window=delta_window,
    t_dephase=t_dephase,
    t_echo_free=t_echo_free,
    Sz_init=Sz_init,
    Sz_target=Sz_target,
    w_pop_inside=w_pop_inside,
    w_pop_outside=w_pop_outside,
    w_coh=w_coh,
    w_smooth=w_smooth,
    η=η,
    max_iter=MAX_ITER,
    n_sub=N_SUB,
    n_edge=2,
    control_mask=control_mask,
    phase_weights=phase_weights,
    coh_tol=1e-4,
    pop_inside_tol=1e-4,
    pop_outside_tol=1e-4,
    smooth_tol=1e-6,
    checkpoint_every=5,
    checkpoint_file=checkpoint_file,
    iteration_offset=iteration_offset,
    checkpoint_metadata=checkpoint_metadata,)

J, Jpop_inside, Jpop_outside, Jcoh, Jsmooth, Sp_echo, Sz_final = rase_objective(
    E, delta_b, g_b, spin_weight, dt;
    delta_window=delta_window,
    t_dephase=t_dephase,
    t_echo_free=t_echo_free,
    Sz_init=Sz_init,
    Sz_target=Sz_target,
    w_pop_inside=w_pop_inside,
    w_pop_outside=w_pop_outside,
    w_coh=w_coh,
    w_smooth=w_smooth,
    n_sub=N_SUB,
    control_mask=control_mask,
    phase_weights=phase_weights,)

completed_iterations = Int(load(checkpoint_file)["iter"])
metadata = merge(checkpoint_metadata, (completed_iterations=completed_iterations,))

final_checkpoint = load(checkpoint_file)
η = Float64(final_checkpoint["η"])
@save result_file E E_before_optimization control_mask η metadata checkpoint_metadata completed_iterations J Jpop_inside Jpop_outside Jcoh Jsmooth J0 Jpop_inside0 Jpop_outside0 Jcoh0 Jsmooth0

# =============================================================
# Plot results
# =============================================================
PlotPulse.plot_pulse(E, dt;
    reference=E_before_optimization,
    label="After this run",
    reference_label="Before this run",
    amplitude_threshold=phase_plot_frac,
    filename=joinpath(output_dir, "pulse_comparison.png"),
    title="Optimized three-pulse RASE control",)

PlotResults.plot_sz_final_map(delta_b, g_b, Sz_final;
    M_delta=M_DELTA, M_g=M_G,
    delta_window_MHz=delta_window / (2π * 1e6),
    filename=joinpath(output_dir, "final_sz.png"),
    title="Final Sz after optimized three-pulse control",)

timeline_end = 60e-6
timeline_free_after = timeline_end - (t_dephase + T_pulse)
t_full, C_full = rase_coherence_timeline(
    E, delta_b, g_b, spin_weight, dt;
    delta_window=delta_window,
    t_dephase=t_dephase,
    t_free_after=timeline_free_after,
    Sz_init=Sz_init,
    n_sub=N_SUB,)

PlotResults.plot_coherence_timeline(t_full, C_full;
    pulse_start=t_dephase,
    pulse_end=t_dephase + T_pulse,
    echo_time=t_dephase + T_pulse + t_echo_free,
    xtick_step_us=5.0,
    xlimits_us=(0.0, timeline_end * 1e6),
    filename=joinpath(output_dir, "coherence_timeline.png"),
    title="Coherence during optimized three-pulse sequence",)

@printf("Final: J=%.8f  Jpop_inside=%.8f  Jpop_outside=%.8f  Jcoh=%.8f  Jsmooth=%.8f\n",
    J, Jpop_inside, Jpop_outside, Jcoh, Jsmooth)
@printf("ΔJ=%.8e  ΔJpop_inside=%.8e  ΔJpop_outside=%.8e  ΔJcoh=%.8e  ΔJsmooth=%.8e\n",
    J-J0, Jpop_inside-Jpop_inside0, Jpop_outside-Jpop_outside0,
    Jcoh-Jcoh0, Jsmooth-Jsmooth0)
println("Saved results to: $output_dir")
