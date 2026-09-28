using JLD2
using CUDA
using Printf
using Statistics
using SpinDynamics

include(joinpath(@__DIR__, "..", "src", "plotting", "PlotResults.jl"))
using .PlotResults
include(joinpath(@__DIR__, "..", "src", "plotting", "PlotPulse.jl"))
using .PlotPulse

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
# Optimization settings
# ============================================================
T = 20e-6
N = 400
dt = T / N

n_sub = 8
η = 0.5
N_iter = 0

# target
delta_window = 2π * 0.30e6
target_Sz = ifelse.(abs.(delta_b) .< delta_window, 0.5, -0.5,)

# ============================================================
# Initial pulse
# ============================================================
savefile = joinpath(@__DIR__, "../Results/20μs''.jld2")

if isfile(savefile)
    @load savefile E_opt
    E_opt = ComplexF64.(E_opt)

    if length(E_opt) != N
        error("Saved control has N=$(length(E_opt)), but current N=$N.")
    end

    println("Loaded previous manual-GRAPE control.")

else
    println("No saved control found. Starting from initial guess.")
    
    t = collect(range(0.0, T; length=N))
    bw = 2π * 1.0e6
    t0 = T / 2
    A0 = 0.8
    phase = 0.5 .* (bw / T) .* (t .- t0).^2

    # ARP_like 
    σ = T / 5
    amp = A0 .* exp.(-0.5 .* ((t .- t0) ./ σ).^2)

    """
    # t_rise = 9e-6
    # t_flat = 12e-6
    # t_fall = 9e-6
    # amp = similar(t)

    for k in eachindex(t)
        tk = t[k]
        if tk < t_rise
            amp[k] = A0 * tk / t_rise
        elseif tk < t_rise + t_flat
            amp[k] = A0
        else
            amp[k] = A0 * (T - tk) / t_fall
        end
    end
    """

    E_opt = ComplexF64.(amp .* exp.(1im .* phase))   
end

E_initial = copy(E_opt)
# ============================================================
# Manual GRAPE optimization
# ============================================================
for iter in 1:N_iter
    J = grape_objective_manual(E_opt, delta_b, g_b, dt; 
    target_Sz=target_Sz, spin_weight=spin_weight, n_sub=n_sub,)
    grad = grape_gradient_manual(E_opt, delta_b, g_b, dt; 
    target_Sz=target_Sz, spin_weight=spin_weight, n_sub=n_sub,)

    E_opt .+= η .* grad
    J_new = grape_objective_manual(E_opt, delta_b, g_b, dt; 
    target_Sz=target_Sz, spin_weight=spin_weight, n_sub=n_sub,)

    @printf("iter = %d   J = %.9f → %.9f\n", iter, J, J_new)
end

@save savefile E_opt

# ============================================================
# Final inversion
# ============================================================
_, _, Sz_hist = grape_forward_manual(E_opt, delta_b, g_b, dt; n_sub=n_sub,)
Sz_final = Array(Sz_hist[:, end])

Sz_cpu = Array(real.(Sz_final))
delta_cpu = Array(delta_b)
weight_cpu = Array(spin_weight)

inside  = abs.(delta_cpu) .< delta_window 
outside = abs.(delta_cpu) .>= delta_window

Sz_in = Sz_cpu[inside]
Sz_out = Sz_cpu[outside]

w_in = weight_cpu[inside]
w_out = weight_cpu[outside]

mean_in = sum(w_in .* Sz_in) / sum(w_in)
mean_out = sum(w_out .* Sz_out) / sum(w_out)

@printf("Inside window:\n")
@printf("  weighted mean = %.5f\n", mean_in)
@printf("  median        = %.5f\n", median(Sz_in))
@printf("  range         = %.5f ~ %.5f\n", minimum(Sz_in), maximum(Sz_in))

@printf("Outside window:\n")
@printf("  weighted mean = %.5f\n", mean_out)
@printf("  median        = %.5f\n", median(Sz_out))
@printf("  range         = %.5f ~ %.5f\n", minimum(Sz_out), maximum(Sz_out))

bad_in = inside .& (Sz_cpu .< 0.4)
bad_out = outside .& (Sz_cpu .> -0.4)

w_cpu = Array(spin_weight)
weighted_bad_in = sum(w_cpu[bad_in]) / sum(w_cpu[inside])
weighted_bad_out = sum(w_cpu[bad_out]) / sum(w_cpu[outside])

@printf("Inside weighted bad fraction  = %.2f%%\n", 100 * weighted_bad_in)
@printf("Outside weighted bad fraction = %.2f%%\n", 100 * weighted_bad_out)

# ============================================================
# Plots
# ============================================================
PlotPulse.plot_pulse(E_opt, dt;
    reference=E_initial, label="Optimized", reference_label="Initial",
    filename=joinpath(@__DIR__, "../Results/20μs_pulse''.png"),)

# delta_plot = Array(delta_b)
# g_plot = Array(g_b)

# PlotResults.plot_sz_final_map(delta_plot, g_plot, Sz_final;
#    M_delta=M_delta, M_g=M_g,
#    filename=joinpath(@__DIR__, "../Results/20μs_Sz''.png"),)

