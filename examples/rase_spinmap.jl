using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CairoMakie
using SpinDynamics
using JLD2
using Dates

include(joinpath(@__DIR__, "..", "src", "plotting", "PlotVideo.jl"))

using CUDA
using Printf
using Statistics

include(joinpath(@__DIR__, "..", "src", "plotting", "PlotPulse.jl"))
using .PlotPulse
include(joinpath(@__DIR__, "..", "src", "plotting", "PlotResults.jl"))
using .PlotResults
include(joinpath(@__DIR__, "..", "src", "pulses", "pulses.jl"))

timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
file_name = joinpath(@__DIR__, "..", "Data", "sim_$(timestamp).jld2")

# ============================================================
# Optimized GRAPE pulse
# ============================================================
"""
pulsefile = joinpath(@__DIR__, "../Results/E_rase_20μs.jld2")
d = load(pulsefile)

E_opt = ComplexF64.(d["E_opt"])

T_pulse = 20e-6
N_control = length(E_opt)
dt = T_pulse / N_control

t_dephase = 5e-6
t_pulse_start = t_dephase

grape_pulse = piecewise_drive(E_opt, t_pulse_start, dt)
"""
# ============================================================
# Simulation
# ============================================================
SIM_SETTING = (
    simulation_order = :order1,
    N = 2000,
    M_delta = 101,
    M_g = 101,

    Ttotal = 40e-6,
    Nt_save = 801,

    initial_condition = (
        kind = :rase,
        delta_window = 2π * 0.30e6,
        Sz_init = -0.499,
        phase = 0.0,
    ),

    reltol = 1e-7,
    abstol = 1e-7,

    saved_file_name = nothing,
)

SYSTEM_CONFIG = (
    freq_inhomogeneity = (
        kind = :gaussian,
        FWHM = 2π * 0.4e6,
        span_sigma = 2.9435,
        renormalize = true,
    ),

    g_inhomogeneity = (
        kind = :gaussian,
        mean = 2π * 100e3,
        FWHM = 2π * 10e3,
        span_sigma = 2.35482,
    ),
)


PULSE_CONFIG = [
    (
        #kind = :custom,
        #f = grape_pulse,
        name       = "RASE ARP π pulse 1",
        kind       = :wurst,

        t_center   = 7.5e-6,
        duration   = 5e-6,

        amp        = 1e4,
        bandwidth  = 2π * 10e6,

        n          = 20.0,
        omega0     = 0.0,
        chirp_sign = +1.0,
        phase0     = 0.0,
        edge_frac  = 0.01,
    ),

    (
        name       = "RASE ARP π pulse 2",
        kind       = :wurst,

        t_center   = 15e-6,
        duration   = 10e-6,

        amp        = 1e4,
        bandwidth  = 2π * 10e6,

        n          = 20.0,
        omega0     = 0.0,
        chirp_sign = +1.0,
        phase0     = 0.0,
        edge_frac  = 0.01,
    ),

    (
        name       = "RASE ARP π pulse 3",
        kind       = :wurst,

        t_center   = 22.5e-6,
        duration   = 5e-6,

        amp        = 1e4,
        bandwidth  = 2π * 10e6,

        n          = 20.0,
        omega0     = 0.0,
        chirp_sign = +1.0,
        phase0     = 0.0,
        edge_frac  = 0.01,
    ),
]

p1 = wurst_drive(
    t_center   = 7.5e-6,
    duration   = 5e-6,
    amp        = 1e4,
    bandwidth  = 2π * 10e6,
    n          = 20.0,
    omega0     = 0.0,
    chirp_sign = +1.0,
    phase0     = 0.0,
    edge_frac  = 0.01,
)

p2 = wurst_drive(
    t_center   = 15e-6,
    duration   = 10e-6,
    amp        = 1e4,
    bandwidth  = 2π * 10e6,
    n          = 20.0,
    omega0     = 0.0,
    chirp_sign = +1.0,
    phase0     = 0.0,
    edge_frac  = 0.01,
)

p3 = wurst_drive(
    t_center   = 22.5e-6,
    duration   = 5e-6,
    amp        = 1e4,
    bandwidth  = 2π * 10e6,
    n          = 20.0,
    omega0     = 0.0,
    chirp_sign = +1.0,
    phase0     = 0.0,
    edge_frac  = 0.01,
)

E_three(t) = p1(t) + p2(t) + p3(t)
fig = PlotPulse.plot_pulse(E_three;
    t_start = 5e-6, t_end = 25e-6, dt = 0.01e-6,
    filename = joinpath(@__DIR__, "../Results/20μs_threeARP_pulse.png"),)

display(fig)

# ============================================================
# Run
# ============================================================
data = run_simulation(SIM_SETTING, SYSTEM_CONFIG,PULSE_CONFIG,)
@save "Results/RASE_threeARP_20us.jld2" data

println("Simulation finished.")

PlotVideo.plot_sz_video(data;
   filename=joinpath(@__DIR__, "..", "Results", "Sz_sim_$(timestamp).mp4"),
   fps=20,)

PlotVideo.plot_phase_video(data;
    filename = joinpath(@__DIR__, "..", "Results", "Phase_sim_$(timestamp).mp4"),
    fps = 20,)     # 每秒播放的帧数

C = PlotResults.compute_phase_coherence(data)

fig = Figure(size=(800, 400))
ax = Axis(fig[1, 1],
    xlabel = "Time (μs)",
    ylabel = "Phase coherence",
    limits = (nothing, (0, 1)),
    xticks = 0:5:40,)
lines!(ax, data.t_saved .* 1e6, C,)

fig