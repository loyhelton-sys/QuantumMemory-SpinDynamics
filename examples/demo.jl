# The run_demo.jl file serves as an demo to check whether the package is well installed.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using SpinDynamics

SIM_SETTING = (
    N = 1,
    # --- simulation order ---
    simulation_order = :order1,   # :order1 (first-order), :order2 (second-order)

    # --- discretization ---
    M_delta = 1,
    M_g     = 1,

    # --- initial condition ---
    initial_condition = :ground,        # :ground, :inverted, or :custom

    # --- simulation time ---
    Ttotal = 20e-6,

    # --- solver ---
    Nt_save = 201,
    reltol  = 1e-8,
    abstol  = 1e-8,

    # --- saving ---
    saved_file_name = nothing,
)

SYSTEM_CONFIG = (
    # --- detuning distribution ---
    freq_inhomogeneity = (kind = :constant, delta_value = 2π * 2e5, FWHM = 0.0,),

    # --- coupling distribution ---
    g_inhomogeneity = ( kind = :constant, g_value = 2*pi*100,),
)

PULSE_CONFIG = (
    (name = "Constant pulse", 
     kind = :constant, 
     amp = 2π * 5e4,),)

data = SpinDynamics.run_simulation(SIM_SETTING, SYSTEM_CONFIG, PULSE_CONFIG)

println("Run finished.")

include("../plots/PlotBloch.jl")
plot_bloch(data, filename=joinpath(@__DIR__, "..", "Results", "Rabi_amp5e4_delta2e5.mp4"))
println("Plot finished.")

nothing