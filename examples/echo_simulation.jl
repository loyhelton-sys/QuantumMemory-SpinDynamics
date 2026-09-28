using CUDA
using CairoMakie
using Statistics
using Printf
using SpinDynamics

include(joinpath(@__DIR__, "..", "src", "plotting", "PlotPulse.jl"))
include(joinpath(@__DIR__, "..", "src", "plotting", "PlotResults.jl"))
using .PlotPulse
using .PlotResults

# ============================================================
# Ensemble settings
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
# Settings
# ============================================================
t_dephase = 5e-6
T_pulse = 10e-6
t_gap = 10e-6
t_free_after = 15e-6
n_sub = 16

# ============================================================
# Spin weights
# ============================================================
delta_window = 2π * 0.30e6
delta_cpu = Array(delta_b)
weight_cpu = Array(spin_weight)

inside = abs.(delta_cpu) .< delta_window

function coherence(Splus)
    S = Splus[inside]
    w = weight_cpu[inside]

    return abs(sum(w .* S)) / sum(w .* abs.(S))
end

# ============================================================
# Load optimized pulse
# ============================================================
# pulsefile = joinpath(@__DIR__, "../Results/grape_inversion/20μs.jld2")
# pulsefile = joinpath(@__DIR__, "../Results/20μs''.jld2")
# @load pulsefile E_opt

N_pulse = 400
dt = T_pulse / N_pulse

pulse = wurst_drive(t_center=0.0, duration=T_pulse, amp=1.0,
    bandwidth=2π * 1e6,
    n=20.0, omega0=0.0,
    chirp_sign=+1.0, phase0=0.0,
    edge_frac=0.01,)

t_control = collect(range(-T_pulse/2, T_pulse/2; length=N_pulse))

E_control = ComplexF64.(pulse.(t_control))
# E_control = ComplexF64.(E_opt)
# N_pulse = length(E_control)
# dt = T_pulse / N_pulse

# amp_control = abs.(E_control)

# t_control = (0:N_pulse-1) .* dt

# ============================================================
# Initial coherence
# ============================================================
M = length(delta_cpu)
Splus0 = fill(0.5 + 0im, M)

# ============================================================
# 1. Free dephasing
# ============================================================
Splus_d = Splus0 .* exp.(1im .* delta_cpu .* t_dephase)

Sx_d = real.(Splus_d)
Sy_d = imag.(Splus_d)
Sz_d = zeros(M)

# ============================================================
# 2. Pulse 1
# ==========================================================
Sx1, Sy1, Sz1 = grape_forward_manual(E_control, delta_b, g_b, dt;
    n_sub=n_sub,
    Sx0=CuArray(Sx_d),
    Sy0=CuArray(Sy_d),
    Sz0=CuArray(Sz_d),)

Sx1_end = Array(Sx1[:, end])
Sy1_end = Array(Sy1[:, end])
Sz1_end = Array(Sz1[:, end])

# ============================================================
# 3. Free gap
# ============================================================
Splus1 = Sx1_end .+ 1im .* Sy1_end
Splus_gap = Splus1 .* exp.(1im .* delta_cpu .* t_gap)

Sx_gap = real.(Splus_gap)
Sy_gap = imag.(Splus_gap)

# ============================================================
# 4. Pulse 2
# ============================================================
Sx2, Sy2, Sz2 = grape_forward_manual(E_control, delta_b, g_b, dt;
    n_sub=n_sub,
    Sx0=CuArray(Sx_gap),
    Sy0=CuArray(Sy_gap),
    Sz0=CuArray(Sz1_end),)

Sx2_end = Array(Sx2[:, end])
Sy2_end = Array(Sy2[:, end])

Splus2 = Sx2_end .+ 1im .* Sy2_end

# ============================================================
# Full timeline
# ============================================================
T_total = t_dephase + 2T_pulse + t_gap + t_free_after
Nt_plot = 1501
t_full = collect(range(0.0, T_total; length=Nt_plot))

t1s = t_dephase
t1e = t1s + T_pulse
t2s = t1e + t_gap
t2e = t2s + T_pulse

C_full = zeros(Float64, Nt_plot)
E_full = zeros(ComplexF64, Nt_plot)

Sx1_cpu = Array(Sx1)
Sy1_cpu = Array(Sy1)

Sx2_cpu = Array(Sx2)
Sy2_cpu = Array(Sy2)

for k in eachindex(t_full)
    t = t_full[k]

    if t < t1s
        S = Splus0 .* exp.(1im .* delta_cpu .* t)
        C_full[k] = coherence(S)

    elseif t < t1e
        τ = t - t1s
        idx = clamp(floor(Int, τ / dt) + 1, 1, N_pulse + 1)

        S = Sx1_cpu[:, idx] .+ 1im .* Sy1_cpu[:, idx]
        C_full[k] = coherence(S)

        idxE = clamp(idx, 1, N_pulse)
        E_full[k] = E_control[idxE]

    elseif t < t2s
        τ = t - t1e
        S = Splus1 .* exp.(1im .* delta_cpu .* τ)
        C_full[k] = coherence(S)

    elseif t < t2e
        τ = t - t2s
        idx = clamp(floor(Int, τ / dt) + 1, 1, N_pulse + 1)

        S = Sx2_cpu[:, idx] .+ 1im .* Sy2_cpu[:, idx]
        C_full[k] = coherence(S)

        idxE = clamp(idx, 1, N_pulse)
        E_full[k] = E_control[idxE]

    else
        τ = t - t2e
        S = Splus2 .* exp.(1im .* delta_cpu .* τ)
        C_full[k] = coherence(S)
    end
end

# ============================================================
# Echo peak after pulse 2
# ============================================================
after2 = t_full .>= t2e
idx_after = findall(after2)
idx_peak = idx_after[argmax(C_full[after2])]

@printf("\n--- Two-pulse echo test ---\n")
@printf("Maximum coherence C = %.5f\n", C_full[idx_peak])
@printf("Echo time           = %.3f us\n", t_full[idx_peak] * 1e6)

# ============================================================
# Plot
# ============================================================
dt_plot = t_full[2] - t_full[1]
PlotPulse.plot_pulse(E_full, dt_plot;
    t_start=first(t_full),
    filename=joinpath(@__DIR__, "../Results/two_pulse_control.png"),
    title="Two-pulse control",
)
PlotResults.plot_coherence_timeline(t_full, C_full;
    pulse_start=t1s,
    pulse_end=t2e,
    echo_time=t_full[idx_peak],
    filename=joinpath(@__DIR__, "../Results/two_pulse_coherence.png"),
)

println("Saved two-pulse control and coherence plots.")