using CUDA
using JLD2

function validate_rase_inputs(E, delta_b, g_b, spin_weight, dt, delta_window, n_sub)
    isempty(E) && error("E cannot be empty.")
    length(delta_b) == length(g_b) == length(spin_weight) ||
        error("delta_b, g_b, and spin_weight must have equal lengths.")
    delta_b isa CuArray && g_b isa CuArray && spin_weight isa CuArray ||
        error("RASE propagation requires delta_b, g_b, and spin_weight on the GPU.")
    dt isa Real && isfinite(dt) && dt > 0 ||
        error("dt must be positive and finite.")
    delta_window isa Real && isfinite(delta_window) && delta_window > 0 ||
        error("delta_window must be positive and finite.")
    n_sub isa Integer && n_sub > 0 || error("n_sub must be a positive integer.")
    any(spin_weight .< 0) && error("spin_weight cannot contain negative values.")
    sum(spin_weight) > 0 || error("spin_weight must have a positive sum.")
    any(abs.(delta_b) .< delta_window) ||
        error("delta_window contains no ensemble bins.")
    return nothing
end

function validate_rase_weights(w_pop_inside, w_pop_outside, w_coh, w_smooth)
    all(w -> w isa Real && isfinite(w) && w >= 0,
        (w_pop_inside, w_pop_outside, w_coh, w_smooth)) ||
        error("Objective weights must be non-negative and finite.")
    return nothing
end

function validate_rase_physics(t_dephase, t_echo, Sz_init, Sz_target)
    t_dephase isa Real && isfinite(t_dephase) && t_dephase >= 0 ||
        error("t_dephase must be non-negative and finite.")
    t_echo isa Real && isfinite(t_echo) && t_echo >= 0 ||
        error("echo/free-evolution time must be non-negative and finite.")
    Sz_init isa Real && isfinite(Sz_init) && -0.5 <= Sz_init <= 0.5 ||
        error("Sz_init must be finite and lie in [-0.5, 0.5].")
    Sz_target isa Real && isfinite(Sz_target) && -0.5 <= Sz_target <= 0.5 ||
        error("Sz_target must be finite and lie in [-0.5, 0.5].")
    return nothing
end

function validate_rase_optimizer(η, max_iter, coh_tol, pop_inside_tol,
    pop_outside_tol, smooth_tol,
    checkpoint_every, checkpoint_file)
    η isa Real && isfinite(η) && η > 0 || error("η must be positive and finite.")
    max_iter isa Integer && max_iter >= 0 || error("max_iter must be a non-negative integer.")
    all(tol -> tol isa Real && isfinite(tol) && tol >= 0,
        (coh_tol, pop_inside_tol, pop_outside_tol, smooth_tol)) ||
        error("Objective tolerances must be non-negative and finite.")
    checkpoint_every isa Integer && checkpoint_every > 0 ||
        error("checkpoint_every must be a positive integer.")
    checkpoint_file isa AbstractString && !isempty(checkpoint_file) ||
        error("checkpoint_file must be a non-empty string.")
    return nothing
end

function rase_population_objectives(Sz_final, delta_b, spin_weight;
    delta_window, Sz_target=0.499)
    length(Sz_final) == length(delta_b) == length(spin_weight) ||
        error("Sz_final, delta_b, and spin_weight must have equal lengths.")
    delta_window isa Real && isfinite(delta_window) && delta_window > 0 ||
        error("delta_window must be positive and finite.")
    Sz_target isa Real && isfinite(Sz_target) && -0.5 <= Sz_target <= 0.5 ||
        error("Sz_target must be finite and lie in [-0.5, 0.5].")
    any(spin_weight .< 0) && error("spin_weight cannot contain negative values.")
    sum(spin_weight) > 0 || error("spin_weight must have a positive sum.")

    inside = abs.(delta_b) .< delta_window
    outside = .!inside
    any(inside) || error("delta_window contains no ensemble bins.")
    w_inside = ifelse.(inside, spin_weight, 0.0)
    w_inside ./= sum(w_inside)
    Jpop_inside = -sum(w_inside .* (Sz_final .- Sz_target).^2)

    Jpop_outside = if any(outside)
        w_outside = ifelse.(outside, spin_weight, 0.0)
        w_outside ./= sum(w_outside)
        -sum(w_outside .* (Sz_final .+ 0.5).^2)
    else
        zero(Jpop_inside)
    end
    return Jpop_inside, Jpop_outside
end

# ============================================================
# RASE objective
# ============================================================
function rase_objective(E, delta_b, g_b, spin_weight, dt;
    delta_window, t_dephase, t_echo_free,
    Sz_init=-0.499, Sz_target=0.499,
    w_pop_inside=1.0, w_pop_outside=1.0,
    w_coh=1.0, w_smooth=1e-4, n_sub=8, control_mask=nothing,
    phase_weights=nothing)

    validate_rase_inputs(E, delta_b, g_b, spin_weight, dt, delta_window, n_sub)
    validate_rase_weights(w_pop_inside, w_pop_outside, w_coh, w_smooth)
    validate_rase_physics(t_dephase, t_echo_free, Sz_init, Sz_target)

    inside = abs.(delta_b) .< delta_window
    weights = spin_weight ./ sum(spin_weight)
    Sp_amp = sqrt(0.25 - Sz_init^2)

    Sx0 = ifelse.(inside, Sp_amp, 0.0)
    Sy0 = similar(delta_b); fill!(Sy0, 0)
    Sz0 = ifelse.(inside, Sz_init, -0.5)

    θ = delta_b .* t_dephase
    Sx_d = Sx0 .* cos.(θ) .- Sy0 .* sin.(θ)
    Sy_d = Sx0 .* sin.(θ) .+ Sy0 .* cos.(θ)

    Sx_hist, Sy_hist, Sz_hist = grape_forward_manual(
        E, delta_b, g_b, dt;
        n_sub=n_sub, Sx0=Sx_d, Sy0=Sy_d, Sz0=Sz0)

    θe = delta_b .* t_echo_free
    X = Sx_hist[:,end] .* cos.(θe) .- Sy_hist[:,end] .* sin.(θe)
    Y = Sx_hist[:,end] .* sin.(θe) .+ Sy_hist[:,end] .* cos.(θe)
    Sz_final = Sz_hist[:,end]

    Jpop_inside, Jpop_outside = rase_population_objectives(
        Sz_final, delta_b, spin_weight;
        delta_window=delta_window, Sz_target=Sz_target)

    w = ifelse.(inside, spin_weight, 0.0)
    w ./= sum(w)
    R = sqrt.(X.^2 .+ Y.^2)
    Qx, Qy = sum(w .* X), sum(w .* Y)
    A = sum(w .* R)
    Jcoh = (Qx^2 + Qy^2) / max(A^2, 1e-28)

    control_mask === nothing || length(control_mask) == length(E) ||
        error("control_mask must have the same length as E.")
    phase_weights === nothing || length(phase_weights) == length(E) ||
        error("phase_weights must have the same length as E.")
    Jsmooth = frequency_smooth_penalty(E; λ_freq=1.0, control_mask=phase_weights)
    J = w_pop_inside * Jpop_inside + w_pop_outside * Jpop_outside +
        w_coh * Jcoh + w_smooth * Jsmooth

    return J, Jpop_inside, Jpop_outside, Jcoh, Jsmooth,
        X .+ 1im .* Y, Sz_final
end

# ============================================================
# RASE gradient
# ============================================================
function rase_gradient(E, delta_b, g_b, spin_weight, dt;
    delta_window, t_dephase, t_echo_free,
    Sz_init=-0.499, Sz_target=0.499,
    w_pop_inside=1.0, w_pop_outside=1.0,
    w_coh=1.0, w_smooth=1e-4, n_sub=8, n_edge=0, control_mask=nothing,
    phase_weights=nothing)

    validate_rase_inputs(E, delta_b, g_b, spin_weight, dt, delta_window, n_sub)
    validate_rase_weights(w_pop_inside, w_pop_outside, w_coh, w_smooth)
    validate_rase_physics(t_dephase, t_echo_free, Sz_init, Sz_target)
    n_edge isa Integer && 0 <= n_edge <= length(E) ||
        error("n_edge must lie between 0 and length(E).")
    control_mask === nothing || length(control_mask) == length(E) ||
        error("control_mask must have the same length as E.")
    phase_weights === nothing || length(phase_weights) == length(E) ||
        error("phase_weights must have the same length as E.")

    M, N = length(delta_b), length(E)
    h = dt / n_sub
    inside = abs.(delta_b) .< delta_window
    Sp_amp = sqrt(0.25 - Sz_init^2)
    Sx0 = ifelse.(inside, Sp_amp, 0.0)
    Sy0 = similar(delta_b); fill!(Sy0, 0)
    Sz0 = ifelse.(inside, Sz_init, -0.5)

    θ = delta_b .* t_dephase
    Sx_d = Sx0 .* cos.(θ) .- Sy0 .* sin.(θ)
    Sy_d = Sx0 .* sin.(θ) .+ Sy0 .* cos.(θ)

    Sx_hist, Sy_hist, Sz_hist = grape_forward_manual(E, delta_b, g_b, dt;
        n_sub=n_sub, Sx0=Sx_d, Sy0=Sy_d, Sz0=Sz0)

    Sp_echo = (Sx_hist[:, end] .+ 1im .* Sy_hist[:, end]) .*
              exp.(1im .* delta_b .* t_echo_free)
    Sz_final = Sz_hist[:, end]

    outside = .!inside
    w_inside = ifelse.(inside, spin_weight, 0.0)
    w_inside ./= sum(w_inside)
    w_outside = ifelse.(outside, spin_weight, 0.0)
    any(outside) && (w_outside ./= sum(w_outside))
    λz = -2w_pop_inside .* w_inside .* (Sz_final .- Sz_target) .-
         2w_pop_outside .* w_outside .* (Sz_final .+ 0.5)

    w = ifelse.(inside, spin_weight, 0.0)
    w ./= sum(w)

    X, Y = real.(Sp_echo), imag.(Sp_echo)
    R = sqrt.(X.^2 .+ Y.^2)
    Qx, Qy = sum(w .* X), sum(w .* Y)
    A = max(sum(w .* R), 1e-14)
    Q2 = Qx^2 + Qy^2
    R = max.(R, 1e-14)

    λx = 2w_coh .* w .* (Qx/A^2 .- Q2 .* X ./ (A^3 .* R))
    λy = 2w_coh .* w .* (Qy/A^2 .- Q2 .* Y ./ (A^3 .* R))

    c, s = cos.(delta_b .* t_echo_free), sin.(delta_b .* t_echo_free)
    λx, λy = λx .* c .+ λy .* s, -λx .* s .+ λy .* c


    grad_Ex, grad_Ey = zeros(N), zeros(N)
    gx, gy = similar(delta_b, M), similar(delta_b, M)

    for k in N:-1:1
        Ex, Ey = real(E[k]), imag(E[k])
        Sx, Sy, Sz = copy(Sx_hist[:,k+1]), copy(Sy_hist[:,k+1]), copy(Sz_hist[:,k+1])

        fill!(gx, 0)
        fill!(gy, 0)

        for _ in 1:n_sub
            gradient_accumulate_step_gpu!(
                gx, gy, Sx, Sy, Sz, λx, λy, λz, g_b, h)
            rk4_step_gpu!(Sx, Sy, Sz, Ex, Ey, -h, delta_b, g_b)
            rk4_step_gpu!(λx, λy, λz, Ex, Ey, -h, delta_b, g_b)
        end

        grad_Ex[k] = sum(gx)
        grad_Ey[k] = sum(gy)
    end

    grad = grad_Ex .+ 1im .* grad_Ey
    w_smooth > 0 && (grad .+= w_smooth .* frequency_smooth_gradient(
        E; λ_freq=1.0, control_mask=phase_weights))

    control_mask !== nothing && (grad[.!control_mask] .= 0)

    if n_edge > 0
        grad[1:n_edge] .= 0
        grad[end-n_edge+1:end] .= 0
    end

    return grad
end

# ============================================================
# RASE coherence timeline
# ============================================================
function rase_coherence_timeline(E_control, delta_b, g_b, spin_weight, dt;
    delta_window, t_dephase, t_free_after,
    Sz_init=-0.499, n_sub=8, Nt_free=201)

    validate_rase_inputs(E_control, delta_b, g_b, spin_weight, dt, delta_window, n_sub)
    validate_rase_physics(t_dephase, t_free_after, Sz_init, Sz_init)
    Nt_free isa Integer && Nt_free >= 2 || error("Nt_free must be an integer of at least 2.")

    δ = Array(delta_b)
    weights = Array(spin_weight)
    inside = abs.(δ) .< delta_window

    M, N = length(δ), length(E_control)
    T_pulse = N * dt
    Splus_init = sqrt(0.25 - Sz_init^2)

    w = weights[inside]
    w ./= sum(w)

    function coherence(Splus)
        S = Splus[inside]
        denom = sum(w .* abs.(S))
        return denom > 1e-14 ? abs(sum(w .* S)) / denom : 0.0
    end

    Splus0 = ifelse.(inside, ComplexF64(Splus_init), 0.0 + 0.0im)
    Sz0 = ifelse.(inside, Sz_init, -0.5)

    t_before = collect(range(0.0, t_dephase; length=Nt_free))
    C_before = [coherence(Splus0 .* exp.(1im .* δ .* t)) for t in t_before]

    Splus_d = Splus0 .* exp.(1im .* δ .* t_dephase)

    Sx_hist, Sy_hist, _ = grape_forward_manual(
        E_control, delta_b, g_b, dt;
        n_sub=n_sub,
        Sx0=CuArray(real.(Splus_d)),
        Sy0=CuArray(imag.(Splus_d)),
        Sz0=CuArray(Sz0))

    Sx, Sy = Array(Sx_hist), Array(Sy_hist)
    t_pulse = t_dephase .+ (0:N) .* dt
    C_pulse = [coherence(Sx[:,k] .+ 1im .* Sy[:,k]) for k in 1:N+1]

    Splus_after = Sx[:,end] .+ 1im .* Sy[:,end]

    τ_after = collect(range(0.0, t_free_after; length=Nt_free))
    t_after = t_dephase + T_pulse .+ τ_after
    C_after = [coherence(Splus_after .* exp.(1im .* δ .* τ)) for τ in τ_after]

    t_full = vcat(t_before, t_pulse[2:end], t_after[2:end])
    C_full = vcat(C_before, C_pulse[2:end], C_after[2:end])

    return t_full, C_full
end

# ============================================================
# Optimizer
# ============================================================
function optimize_rase!(E, delta_b, g_b, spin_weight, dt;
    delta_window, t_dephase, t_echo_free,
    Sz_init=-0.499, Sz_target=0.499,
    w_pop_inside=1.0, w_pop_outside=1.0,
    w_coh=1.0, w_smooth=1e-4,
    η=0.01, max_iter=100, n_sub=8, n_edge=0, n_tail=0, control_mask=nothing,
    phase_weights=nothing, η_max=Inf, η_growth=1.2,
    coh_tol=1e-5, pop_inside_tol=5e-4, pop_outside_tol=5e-4,
    smooth_tol=1e-5, checkpoint_every=10,
    checkpoint_file="Results/E_rase_checkpoint.jld2",
    iteration_offset=0, checkpoint_metadata=nothing, log_every=10)

    # A fixed tail retains its supplied values, including nonzero taper samples.
    n_tail isa Integer && 0 <= n_tail <= length(E) ||
        error("n_tail must lie between 0 and length(E).")
    validate_rase_optimizer(η, max_iter, coh_tol, pop_inside_tol,
        pop_outside_tol, smooth_tol, checkpoint_every, checkpoint_file)
    η_max isa Real && η_max >= η ||
        error("η_max must be at least the starting learning rate (Inf means no cap).")
    η_growth isa Real && isfinite(η_growth) && η_growth > 1 ||
        error("η_growth must be finite and greater than one.")
    iteration_offset isa Integer && iteration_offset >= 0 ||
        error("iteration_offset must be a non-negative integer.")
    log_every isa Integer && log_every > 0 ||
        error("log_every must be a positive integer.")
    control_mask === nothing || length(control_mask) == length(E) ||
        error("control_mask must have the same length as E.")
    control_mask !== nothing && (E[.!control_mask] .= 0)

    J, Jpop_inside, Jpop_outside, Jcoh, Jsmooth, _, _ = rase_objective(
        E, delta_b, g_b, spin_weight, dt;
        delta_window=delta_window, t_dephase=t_dephase, t_echo_free=t_echo_free,
        Sz_init=Sz_init, Sz_target=Sz_target,
        w_pop_inside=w_pop_inside, w_pop_outside=w_pop_outside,
        w_coh=w_coh, w_smooth=w_smooth, n_sub=n_sub,
        control_mask=control_mask, phase_weights=phase_weights)

    mkpath(dirname(checkpoint_file))
    println("initial  Jcoh=", round(Jcoh, digits=7),
        "  Jpop_inside=", round(Jpop_inside, digits=7),
        "  Jpop_outside=", round(Jpop_outside, digits=7),
        "  Jsmooth=", round(Jsmooth, sigdigits=7),
        "  eta_start=", round(η, sigdigits=6))

    last_iter = iteration_offset
    for local_iter in 1:max_iter
        iter = iteration_offset + local_iter
        last_iter = iter
        grad = rase_gradient(E, delta_b, g_b, spin_weight, dt;
            delta_window=delta_window, t_dephase=t_dephase,
            t_echo_free=t_echo_free, Sz_init=Sz_init, Sz_target=Sz_target,
            w_pop_inside=w_pop_inside, w_pop_outside=w_pop_outside,
            w_coh=w_coh, w_smooth=w_smooth, n_sub=n_sub, n_edge=n_edge,
            control_mask=control_mask, phase_weights=phase_weights)

        n_tail > 0 && (grad[end-n_tail+1:end] .= 0)

        # Reuse this gradient while backtracking; rejected trials do not
        # consume another outer iteration or change the saved control.
        accepted = false
        while η >= 1e-8
            E_trial = E .+ η .* grad
            control_mask !== nothing && (E_trial[.!control_mask] .= 0)
            Jn, Jpop_inside_n, Jpop_outside_n, Jcoh_n, Jsmooth_n, _, _ =
                rase_objective(E_trial, delta_b, g_b, spin_weight, dt;
                    delta_window=delta_window, t_dephase=t_dephase,
                    t_echo_free=t_echo_free, Sz_init=Sz_init, Sz_target=Sz_target,
                    w_pop_inside=w_pop_inside, w_pop_outside=w_pop_outside,
                    w_coh=w_coh, w_smooth=w_smooth, n_sub=n_sub,
                    control_mask=control_mask, phase_weights=phase_weights)
    
            delta_coh = Jcoh_n - Jcoh
            delta_pop_inside = Jpop_inside_n - Jpop_inside
            delta_pop_outside = Jpop_outside_n - Jpop_outside
            delta_smooth = Jsmooth_n - Jsmooth
            accept = delta_coh >= -coh_tol &&
                     delta_pop_inside >= -pop_inside_tol &&
                     # A disabled outside-population objective must not veto updates.
                     (iszero(w_pop_outside) || delta_pop_outside >= -pop_outside_tol) &&
                     delta_smooth >= -smooth_tol && Jn > J

            if accept
                E .= E_trial
                J, Jpop_inside, Jpop_outside, Jcoh, Jsmooth =
                    Jn, Jpop_inside_n, Jpop_outside_n, Jcoh_n, Jsmooth_n
                accepted = true
                η_used = η
                # Store the next trial step in checkpoints, as on rejection.
                η = min(η_max, η * η_growth)
                if iter % log_every == 0
                    println("iter=", iter, " ACCEPT  Jcoh=", round(Jcoh, digits=7),
                        "  Jpop_inside=", round(Jpop_inside, digits=7),
                        "  Jpop_outside=", round(Jpop_outside, digits=7),
                        "  Jsmooth=", round(Jsmooth, sigdigits=7),
                        "  eta_used=", round(η_used, sigdigits=6),
                        "  eta_next=", round(η, sigdigits=6))
                end
                break
            end

            η *= 0.5
        end
        if !accepted
            println("iter=", iter, " STOP: no acceptable step at eta >= 1e-8.")
        end

        iter % checkpoint_every == 0 &&
            @save checkpoint_file E control_mask J Jpop_inside Jpop_outside Jcoh Jsmooth iter η checkpoint_metadata
        !accepted && break
    end

    iter = last_iter
    @save checkpoint_file E control_mask J Jpop_inside Jpop_outside Jcoh Jsmooth iter η checkpoint_metadata

    return E, J, Jpop_inside, Jpop_outside, Jcoh, Jsmooth
end
