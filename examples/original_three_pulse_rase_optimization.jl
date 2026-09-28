using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CUDA
using JLD2
using Printf
using SpinDynamics

# Load the accelerated backend independently of the installed SpinDynamics module.
# The same script can therefore be run in an existing Julia session.
if !isdefined(@__MODULE__, :RaseGPUAccelerated)
    include(joinpath(@__DIR__, "..", "src", "grape", "RaseGPUAccelerated.jl"))
end
println("RASE backend: accelerated GPU (fused propagation and GPU gradient reduction)")

include(joinpath(@__DIR__, "..", "src", "plotting", "PlotPulse.jl"))
using .PlotPulse
include(joinpath(@__DIR__, "..", "src", "plotting", "PlotResults.jl"))
using .PlotResults

CUDA.functional() || error("A functional CUDA GPU is required for RASE optimization.")

# The defaults are a full optimization run. For a shorter trial, for example:
# SPINDYNAMICS_MAX_ITER=10 julia --project=. examples/original_three_pulse_rase_optimization.jl
env_int(name, default) = parse(Int, get(ENV, name, string(default)))
const M_DELTA = env_int("SPINDYNAMICS_M_DELTA", 101)
const M_G = env_int("SPINDYNAMICS_M_G", 101)
const MAX_ITER = env_int("SPINDYNAMICS_MAX_ITER",100)
const N_SUB = env_int("SPINDYNAMICS_N_SUB", 4)
# Resume the newest manual or completed automatic result by iteration count.
# SPINDYNAMICS_RESUME=false restarts from the frozen single-ARP handoff pulse.
const RESUME = lowercase(get(ENV, "SPINDYNAMICS_RESUME", "true")) in ("1", "true", "yes")

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


# Single-ARP handoff: preserve all 1600 samples over 40 μs.
durations = (40e-6,)
T_pulse = sum(durations)
dt = 0.025e-6  # 1600 control intervals over 40 μs; no resampling
phase_amp_low_frac = 0.001
phase_amp_high_frac = 0.005
# Phase and instantaneous frequency are not meaningful when the control is
# nearly zero. A 1% plotting mask suppresses endpoint/dropout
# artifacts (notably near 0 and 40 us) without changing the optimized control.
phase_plot_frac = 0.01

# ======================================================================
# Objective: population inversion, echo coherence, and smoothness
# ======================================================================
delta_window = 2π * 0.30e6
t_dephase = 5e-6
t_echo_free = 5e-6
Sz_init = -0.499
Sz_target = 0.499
w_pop_inside = 5.0
w_pop_outside = 1.0
w_coh = 2.6
w_smooth =8e-5
# Backtrack within each iteration; grow accepted steps by 20% with no default cap.
η = 0.4

output_dir = joinpath(@__DIR__, "..", "Results", "40us_single_ARP", "manual_from_iq_4336")
println("Manual optimization directory: ", abspath(output_dir))
mkpath(output_dir)
checkpoint_file = joinpath(output_dir, "checkpoint.jld2")
result_file = joinpath(output_dir, "optimized_single_arp.jld2")
auto_result_file = joinpath(@__DIR__, "..", "Results", "40us_single_ARP",
                            "auto_20000", "final_result.jld2")
input_file = joinpath(output_dir, "input_pulse.jld2")
input_data = load(input_file)

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

# Preserve the original ARP parameters and the frozen optimization phase mask.
original_pulse_parameters = input_data["checkpoint_metadata"].original_pulse_parameters
checkpoint_metadata = merge(checkpoint_metadata, (
    initialization=:saved_single_arp,
    original_pulse_parameters=original_pulse_parameters,
    input_file=input_file,
))

function load_previous_control(checkpoint_file, result_file, current_metadata, extra_files=String[])
    candidates = filter(isfile, unique(vcat([result_file, checkpoint_file], extra_files)))
    isempty(candidates) && return nothing
    # Prefer the greatest completed iteration, then the most recently saved file.
    saved_runs = [(file=file, data=load(file)) for file in candidates]
    saved_iteration(data) = Int(get(data, "iter", get(data, "completed_iterations", 0)))
    sort!(saved_runs; by=run -> (saved_iteration(run.data), mtime(run.file)))
    source_file, saved = last(saved_runs).file, last(saved_runs).data
    saved_metadata = get(saved, "checkpoint_metadata", nothing)
    saved_metadata isa NamedTuple || error("Missing control metadata: $source_file")
    for field in (:durations, :T_pulse)
        hasproperty(saved_metadata, field) ||
            error("Saved control metadata is missing $field: $source_file")
        saved_value = getproperty(saved_metadata, field)
        current_value = getproperty(current_metadata, field)
        isequal(saved_value, current_value) ||
            error("Saved control time grid differs at $field: saved=$(repr(saved_value)), configured=$(repr(current_value)). Reload the full script with matching grid settings. File: $source_file")
    end
    E = ComplexF64.(saved["E"])
    saved_dt = saved_metadata.dt
    length(E) == round(Int, saved_metadata.T_pulse / saved_dt) ||
        error("Saved control length does not match its time grid: $source_file")
    if saved_dt != current_metadata.dt
        refinement = round(Int, saved_dt / current_metadata.dt)
        refinement >= 1 && isapprox(saved_dt, refinement * current_metadata.dt; rtol=1e-12, atol=0.0) ||
            error("Resume supports only integer subdivision of saved control intervals.")
        # Preserve the piecewise-constant waveform exactly while refining its grid.
        old_length = length(E)
        E = repeat(E; inner=refinement)
        println("Refined saved control from ", old_length, " to ", length(E), " intervals.")
    end
    length(E) == round(Int, current_metadata.T_pulse / current_metadata.dt) ||
        error("Refined control length does not match the requested time grid.")
    all(isfinite, E) || error("Saved control contains non-finite values: $source_file")
    iteration_offset = saved_iteration(saved)
    println("Resuming from ", source_file, " at iteration ", iteration_offset)
    return (; E, iteration_offset, source_file, η=Float64(get(saved,"η",get(saved,"eta",0.2))))
end

# Reuse fixed output filenames for plots, results, and checkpoint.
resume_state = RESUME ?
    Base.invokelatest(load_previous_control, checkpoint_file, result_file,
                     checkpoint_metadata, [auto_result_file,
                         joinpath(@__DIR__, "..", "Results", "40us_single_ARP",
                                  "population_priority_10000", "final_result.jld2"),
                         joinpath(@__DIR__, "..", "Results", "40us_single_ARP",
                                  "coherence_population_resume", "final_result.jld2"),
                         joinpath(@__DIR__, "..", "Results", "40us_single_ARP",
                                  "coherence_until_plateau", "final_result.jld2")]) : nothing
if isnothing(resume_state)
    resume_state = Base.invokelatest(load_previous_control, input_file, input_file, checkpoint_metadata)
end
E = resume_state.E
iteration_offset = resume_state.iteration_offset
# Resume controls and iteration count, but use the configured starting learning rate.
println("Starting this run with eta=", η)
control_mask = trues(length(E))

# Keep the same smoothness metric as the automated run (do not re-mask
# against the now-optimized pulse amplitude on each manual continuation).
phase_weights = Float64.(input_data["phase_weights"])
length(phase_weights) == length(E) || error("Handoff phase mask and control grid differ.")
checkpoint_metadata = merge(checkpoint_metadata, (; phase_weights))

# =============================================================
# Current objective values before optimization
# =============================================================
E_before_optimization = copy(E)

J0, Jpop_inside0, Jpop_outside0, Jcoh0, Jsmooth0, _, _ = RaseGPUAccelerated.fast_objective(
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
E, J, Jpop_inside, Jpop_outside, Jcoh, Jsmooth = RaseGPUAccelerated.fast_optimize_rase!(
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
    pop_inside_tol=1e-3,
    pop_outside_tol=1e-3,
    smooth_tol=1e-3,
    checkpoint_every=5,
    checkpoint_file=checkpoint_file,
    iteration_offset=iteration_offset,
    checkpoint_metadata=checkpoint_metadata,)

J, Jpop_inside, Jpop_outside, Jcoh, Jsmooth, Sp_echo, Sz_final = RaseGPUAccelerated.fast_objective(
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
@save result_file E E_before_optimization control_mask phase_weights η metadata checkpoint_metadata completed_iterations J Jpop_inside Jpop_outside Jcoh Jsmooth J0 Jpop_inside0 Jpop_outside0 Jcoh0 Jsmooth0

# =============================================================
# Plot results
# =============================================================
PlotPulse.plot_pulse(E, dt;
    reference=E_before_optimization,
    label="After this run",
    reference_label="Before this run",
    amplitude_threshold=phase_plot_frac,
    filename=joinpath(output_dir, "pulse_comparison.png"),
    title="Manually optimized 40 us single-ARP control",)

PlotResults.plot_sz_final_map(delta_b, g_b, Sz_final;
    M_delta=M_DELTA, M_g=M_G,
    delta_window_MHz=delta_window / (2π * 1e6),
    filename=joinpath(output_dir, "final_sz.png"),
    title="Final Sz after optimized 40 us control",)

timeline_end = t_dephase + T_pulse + t_echo_free + 10e-6
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
    title="Coherence during optimized 40 us control",)

@printf("Final: J=%.8f  Jpop_inside=%.8f  Jpop_outside=%.8f  Jcoh=%.8f  Jsmooth=%.8f\n",
    J, Jpop_inside, Jpop_outside, Jcoh, Jsmooth)
@printf("ΔJ=%.8e  ΔJpop_inside=%.8e  ΔJpop_outside=%.8e  ΔJcoh=%.8e  ΔJsmooth=%.8e\n",
    J-J0, Jpop_inside-Jpop_inside0, Jpop_outside-Jpop_outside0,
    Jcoh-Jcoh0, Jsmooth-Jsmooth0)
println("Saved results to: $output_dir")
