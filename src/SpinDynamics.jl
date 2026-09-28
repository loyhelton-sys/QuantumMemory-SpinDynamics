module SpinDynamics

using DifferentialEquations
using CUDA
using JLD2
using Distributions
using QuadGK: quadgk
using DiffEqCallbacks

include(joinpath(@__DIR__, "config.jl"))

include(joinpath(@__DIR__, "ensemble", "frequency_inhomogeneity.jl"))
include(joinpath(@__DIR__, "ensemble", "coupling_inhomogeneity.jl"))

include(joinpath(@__DIR__, "ensemble", "ensemble.jl"))

include(joinpath(@__DIR__, "pulses", "pulses.jl"))
include(joinpath(@__DIR__, "pulses", "pulse_editing.jl"))

include(joinpath(@__DIR__, "grape", "grape_manual.jl"))
include(joinpath(@__DIR__, "grape", "grape_rase.jl"))
include(joinpath(@__DIR__, "dynamics", "state_layout_1st_order.jl"))
include(joinpath(@__DIR__, "dynamics", "initial_conditions_1st_order.jl"))

include(joinpath(@__DIR__, "dynamics", "rhs_1st_order.jl"))
include(joinpath(@__DIR__, "dynamics", "solver_1st_order.jl"))

include(joinpath(@__DIR__, "simulation_api.jl"))

# ---------------- User API ----------------
export run_simulation
export build_ensemble
export build_equal_weight_ensemble

# GRAPE optimization
export grape_forward_manual
export grape_objective_manual
export grape_gradient_manual
export grape_step_manual
export grape_objective_coherence
export grape_gradient_coherence
export rase_objective
export rase_population_objectives
export rase_gradient
export rase_coherence_timeline
export optimize_rase!

# Pulse utilities
export gaussian_pulse_amplitude
export gaussian_drive
export wurst_drive
export three_wurst_drive
export constant_drive
export custom_drive
export piecewise_drive

# Pulse editing utilities
export unwrap_phase
export taper_edges!
export smooth_phase_region!
export smooth_amplitude!
export limit_phase_slew!
export resample_pulse
export inspect_pulse
export frequency_smooth_penalty
export frequency_smooth_gradient

end
