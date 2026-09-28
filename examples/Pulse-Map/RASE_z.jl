using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CairoMakie
using JLD2
using Dates
using .SpinDynamics

timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
file_name = joinpath(@__DIR__, "..", "Data", "sim_$(timestamp).jld2")

include(joinpath(@__DIR__, "..", "..", "src", "plotting", "PlotVideo.jl"))
using .PlotVideo

# Default to 401 × 401; retain the optimization grid environment overrides.
pulse_map_m_delta = parse(Int, get(ENV, "SPINDYNAMICS_M_DELTA", "401"))
pulse_map_m_g = parse(Int, get(ENV, "SPINDYNAMICS_M_G", "401"))

SIM_SETTING = (
    simulation_order = :order1,   # first-order dynamics
    reltol = 1e-6, abstol = 1e-6,
    initial_condition = (
        kind = :rase,
        delta_window = 2π * 0.30e6,
        Sz_init = -0.499,
        phase = 0.0,
    ),        # :ground, :inverted, :mixed, :rase, or :custom

    N = pulse_map_m_delta * pulse_map_m_g, M_delta = pulse_map_m_delta, M_g = pulse_map_m_g,
    
    Ttotal = 70e-6, Nt_save = 801,
    saved_file_name = nothing,
)

# Optimization sampling and detuning distribution; coupling scaled to 100 Hz.
ENSEMBLE_CONFIG = (
    freq_inhomogeneity = (
        kind = :gaussian,
        FWHM = 2π * 0.4e6,
        span_sigma = (2π * 0.5e6) / ((2π * 0.4e6) / (2 * sqrt(2 * log(2)))),
        renormalize = true,
    ),

    g_inhomogeneity = (
        kind = :gaussian,
        mean = 2π * 100,
        FWHM = 2π * 10,
        span_sigma = (2π * 10) / ((2π * 10) / (2 * sqrt(2 * log(2)))),
        renormalize = true,
    )
)

# Reuse the optimization's quantile coordinates. Custom arrays preserve equal
# weights in run_simulation, giving the same Nj and flattened ensemble grid.
ens = SpinDynamics.build_equal_weight_ensemble(
    ENSEMBLE_CONFIG.freq_inhomogeneity, ENSEMBLE_CONFIG.g_inhomogeneity,
    pulse_map_m_delta, pulse_map_m_g,
)
SYSTEM_CONFIG = (
    freq_inhomogeneity = (kind=:custom, delta_values=ens.delta_b_1d),
    g_inhomogeneity = (kind=:custom, g_values=ens.g_b_1d),
)

PULSE_CONFIG = (
    (   name       = "RASE ARP π pulse 1",
        kind       = :wurst,

        t_center   = 10e-6,
        duration   = 10e-6,

        amp        = 10000,
        bandwidth  = 2π * 10e6,

        n          = 20.0,
        omega0     = 0.0,
        chirp_sign = +1.0,
        phase0     = 0.0,
        edge_frac  = 0.01,
    ),

    (   name       = "RASE ARP π pulse 2",
        kind       = :wurst,

        t_center   = 30e-6,
        duration   = 20e-6,

        amp        = 10000,
        bandwidth  = 2π * 10e6,

        n          = 20.0,
        omega0     = 0.0,
        chirp_sign = +1.0,
        phase0     = 0.0,
        edge_frac  = 0.01,
    ),
    
    (   name       = "RASE ARP π pulse 3",
        kind       = :wurst,

        t_center   = 50e-6,
        duration   = 10e-6,

        amp        = 10000,
        bandwidth  = 2π * 10e6,

        n          = 20.0,
        omega0     = 0.0,
        chirp_sign = +1.0,
        phase0     = 0.0,
        edge_frac  = 0.01,
    ),
)

data = run_simulation(SIM_SETTING, SYSTEM_CONFIG, PULSE_CONFIG,)

PlotVideo.plot_sz_video(data;
   filename=joinpath(@__DIR__, "..", "..", "Results", "sim_$(timestamp).mp4"),
   fps=20,)