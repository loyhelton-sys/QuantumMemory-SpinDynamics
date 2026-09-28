using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CairoMakie
using JLD2
using Dates
using SpinDynamics

timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
file_name = joinpath(@__DIR__, "..", "Data", "sim_$(timestamp).jld2")

include(joinpath(@__DIR__, "..", "..", "src", "plotting", "PlotVideo.jl"))
using .PlotVideo

SIM_SETTING = (
    simulation_order = :order1,   # first-order dynamics
    reltol = 1e-6, abstol = 1e-6,
    initial_condition = :ground,        # :ground, :inverted, :mixed, :rase, or :custom

    N = 1000, M_delta = 401, M_g  = 401,
    
    Ttotal = 100e-6, Nt_save = 2001,
    saved_file_name = nothing,
)

SYSTEM_CONFIG = (
    # --- frequency inhomogeneity ---
    freq_inhomogeneity = (
        kind = :lorentzian,
        FWHM = 2π * 1e6,
        span_gamma = 2.5,
        renormalize = false,
    ),

    # --- coupling inhomogeneity ---
    g_inhomogeneity = (kind = :gaussian,
        mean = 2π * 100,
        FWHM = 2π * 5 * 2.35482,
        span_sigma = 3.0,
        renormalize = true,

    #g_inhomogeneity = (
    #    kind = :constant,
    #    g_value = 2π * 100,
    )
)

PULSE_CONFIG = (
    (   name  = "Gaussian input signal",
        kind  = :gaussian,

        t0    = 10e-6,
        sigma = 1e-6,
        amp   = 2e1,
        omega = 0.0,
        phase = 0.0,
    ),

    (   name       = "ROSE ARP π 1",
        kind       = :wurst,

        t_center   = 30e-6,
        duration   = 10e-6,

        amp        = 1e4,
        bandwidth  = 2π * 10e6,

        n          = 20.0,
        omega0     = 0.0,
        chirp_sign = +1.0,
        phase0     = 0.0,
        edge_frac  = 0.01,
    ),

    (   name       = "ROSE ARP π 2",
        kind       = :wurst,

        t_center   = 70e-6,
        duration   = 10e-6,

        amp        = 1e4,
        bandwidth  = 2π * 10e6,

        n          = 20.0,
        omega0     = 0.0,
        chirp_sign = +1.0,
        phase0     = 0.0,
        edge_frac  = 0.01,
    ),
)

data = run_simulation(SIM_SETTING, SYSTEM_CONFIG, PULSE_CONFIG,)

PlotVideo.plot_phase_video(data;
    filename = joinpath(@__DIR__, "..", "Results", "sim_$(timestamp).mp4"),
    fps = 20,     # 每秒播放的帧数
)