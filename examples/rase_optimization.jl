using CUDA
using JLD2
using Printf
using Statistics
using SpinDynamics

include(joinpath(@__DIR__, "..", "src", "plotting", "PlotResults.jl"))
using .PlotResults
include(joinpath(@__DIR__, "..", "src", "plotting", "PlotPulse.jl"))
using .PlotPulse

# ============================================================
# Ensemble
# ============================================================
M_delta, M_g = 101, 101

FWHM_delta = 2π * 0.4e6
σ_delta = FWHM_delta / (2 * sqrt(2 * log(2)))
freq_cfg = (
    kind = :gaussian,
    FWHM = FWHM_delta,
    span_sigma = (2π * 0.5e6) / σ_delta,
)

g_mean = 2π * 100e3
FWHM_g = 2π * 10e3
σ_g = FWHM_g / (2 * sqrt(2 * log(2)))
g_cfg = (
    kind = :gaussian,
    mean = g_mean,
    FWHM = FWHM_g,
    span_sigma = (2π * 10e3) / σ_g,
)

ens = SpinDynamics.build_ensemble(1.0, freq_cfg, g_cfg, M_delta, M_g)
delta_b = CuArray(ens.delta_b)
g_b = CuArray(ens.g_b)
spin_weight = CuArray(ens.Nj ./ ens.N_total)

# ============================================================
# Initial pulse
# ============================================================
pulsefile = joinpath(@__DIR__, "../Results/E_rase_20μs.jld2")
d = load(pulsefile)
E = ComplexF64.(d["E"])
T_pulse = 20e-6
N = length(E)
dt = T_pulse / N

# ============================================================
# Optional pulse editing
# ============================================================
# taper_edges!(E; n_edge=8, amp_floor_frac=0.05)
# smooth_phase_region!(E, 784, 796)
# limit_phase_slew!(E, dt; f_max=2e6)

# Refine time grid if needed
# E, dt = resample_pulse(E, dt; factor=0.5)
# N = length(E)

E_initial = copy(E)
# ============================================================
# RASE optimization
# ============================================================
delta_window = 2π * 0.30e6
t_dephase = 5e-6
t_echo_free = 5e-6

Sz_init=-0.499
Sz_target=0.499

w_coh=1.0
w_pop_inside=0.5
w_pop_outside=0.5
w_smooth=6e-4

n_sub=16 
η=0.02
max_iter=100

E, J, Jpop_inside, Jpop_outside, Jcoh, Jsmooth = optimize_rase!(
    E, delta_b, g_b, spin_weight, dt;
    delta_window=delta_window,
    t_dephase=t_dephase,
    t_echo_free=t_echo_free,
    Sz_init=Sz_init,
    Sz_target=Sz_target,
    w_coh=w_coh, w_pop_inside=w_pop_inside, w_pop_outside=w_pop_outside,
    w_smooth=w_smooth, η=η,
    max_iter=max_iter, n_sub=n_sub, 

    coh_tol=5e-4,
    pop_inside_tol=5e-4,
    pop_outside_tol=5e-4,
    smooth_tol=0.0,)

# ============================================================
# Save
# ============================================================
savefile = joinpath(@__DIR__, "../Results/E_rase_20μs.jld2")
@save savefile E E_initial T_pulse dt J Jpop_inside Jpop_outside Jcoh Jsmooth

println("\nSaved to:")
println(savefile) 

# ============================================================
# Final evaluation
# ============================================================
J, Jpop_inside, Jpop_outside, Jcoh, Jsmooth, Sp_echo, Sz_final = rase_objective(E, delta_b, g_b, spin_weight, dt;
    delta_window=delta_window,
    t_dephase=t_dephase, t_echo_free=t_echo_free,
    Sz_init=Sz_init, Sz_target=Sz_target,
    w_pop_inside=w_pop_inside, w_pop_outside=w_pop_outside, w_coh=w_coh,
    n_sub=n_sub, w_smooth=w_smooth)

@printf("\n--- Final result ---\n")
@printf("J_total = %.8f\n", J)
@printf("J_pop_inside  = %.8f\n", Jpop_inside)
@printf("J_pop_outside = %.8f\n", Jpop_outside)
@printf("J_coh   = %.8f\n", Jcoh)
@printf("J_smooth = %.8f\n", Jsmooth)

# ============================================================
# Plot 1: pulse comparison
# ============================================================
PlotPulse.plot_pulse(E, dt;
    reference=E_initial, label="Optimized", reference_label="Initial",
    filename=joinpath(@__DIR__, "../Results/20μs_pulse.png"),)

# ============================================================
# Plot 2: final Sz map
# ============================================================
PlotResults.plot_sz_final_map(Array(delta_b), Array(g_b), Sz_final;
    M_delta=ens.M_delta, M_g=ens.M_g,
    filename=joinpath(@__DIR__, "../Results/20μs_final_Sz.png"),)

# ============================================================
# Plot 3: coherence timeline
# ============================================================
t_full, C_full = rase_coherence_timeline(E, delta_b, g_b, spin_weight, dt;
    delta_window=delta_window,
    t_dephase=t_dephase,
    t_free_after=t_echo_free,
    Sz_init=Sz_init, n_sub=n_sub,)

PlotResults.plot_coherence_timeline(t_full, C_full; pulse_start=t_dephase,
    pulse_end=t_dephase + T_pulse,
    echo_time=t_dephase + T_pulse + t_echo_free,
    filename=joinpath(@__DIR__, "../Results/20μs_coherence.png"),)