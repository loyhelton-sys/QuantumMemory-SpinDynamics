# The run_demo.jl file serves as an demo to check whether the package is well installed.
using JLD2
using Dates
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using SpinDynamics

SIM_SETTING = (
    N = 9,
    # --- simulation order ---
    simulation_order = :order1,   # first-order dynamics

    # --- discretization ---
    M_delta = 3,
    M_g     = 3,

    # --- initial condition ---
    initial_condition = :ground,        # :ground, :inverted, :mixed, :rase, or :custom

    # --- simulation time ---
    Ttotal = 10e-5,

    # --- solver ---
    Nt_save = 1001,
    reltol  = 1e-9,
    abstol  = 1e-9,

    # --- saving ---
    saved_file_name = nothing,
)

SYSTEM_CONFIG = (
    freq_inhomogeneity = (
    kind = :custom,
    delta_values = 2π .* [0, 1e5, 2e5],), 

    g_inhomogeneity = (
        kind = :custom,
        g_values =  2π .* [5e4, 1e5, 1.5e5],)
)

PULSE_CONFIG = (
    (
        name       = "ARP WURST",
        kind       = :wurst,
        t_center   = 40e-6,      # center of the pulse
        duration   = 40e-6,      # duration of the pulse 
        amp        = 1.0,
        bandwidth  = 2π * 2e6,   # bandwidth of the pulse
        n          = 20.0,       # the order of the WURST pulse
        omega0     = 0.0,        # the center frequency of the pulse
        chirp_sign = +1.0,       # 扫频方向
        phase0     = 0.0,        # 初始相位
        edge_frac  = 1e-2,       # 边缘平滑程度
    ),
)

data = SpinDynamics.run_simulation(SIM_SETTING, SYSTEM_CONFIG, PULSE_CONFIG)
timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
filename = joinpath(@__DIR__, "..", "Data", "sim_$(timestamp).jld2")
@save filename data

include(joinpath(@__DIR__, "..", "..", "src", "plotting", "PlotBloch.jl"))

println("Plotting Bloch sphere animation...")

plot_bloch(data;
    M_delta = SIM_SETTING.M_delta,
    M_g = SIM_SETTING.M_g,
    frame_step = 2)

println("Plot finished.")

nothing