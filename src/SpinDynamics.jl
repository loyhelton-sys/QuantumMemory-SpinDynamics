module SpinDynamics

using LinearAlgebra
using DifferentialEquations
using CUDA
using JLD2
using Distributions
import QuadGK
using QuadGK: quadgk
using DiffEqCallbacks

include("config.jl")

include("frequency_inhomogeneity.jl")
include("coupling_inhomogeneity.jl")

include("ensemble.jl")

include("pulses.jl")
include("initial_conditions_1st_order.jl")
include("state_layout_1st_order.jl")

include("rhs_1st_order.jl")
include("solver_1st_order.jl")

include("simulation_api.jl")

end