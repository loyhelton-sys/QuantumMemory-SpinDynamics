module RaseGPUAccelerated
using CUDA, JLD2
using SpinDynamics: frequency_smooth_penalty, frequency_smooth_gradient,
    validate_rase_inputs, validate_rase_weights, validate_rase_physics,
    rase_population_objectives, validate_rase_optimizer
@inline function scalar_rk4(sx,sy,sz,Ex,Ey,h,δ,g)
        # k1
        k1x = -δ * sy + 2g * Ey * sz
        k1y =  δ * sx - 2g * Ex * sz
        k1z = 2g * (Ex * sy - Ey * sx)

        # k2
        sx2 = sx + h/2 * k1x
        sy2 = sy + h/2 * k1y
        sz2 = sz + h/2 * k1z

        k2x = -δ * sy2 + 2g * Ey * sz2
        k2y =  δ * sx2 - 2g * Ex * sz2
        k2z = 2g * (Ex * sy2 - Ey * sx2)

        # k3
        sx3 = sx + h/2 * k2x
        sy3 = sy + h/2 * k2y
        sz3 = sz + h/2 * k2z

        k3x = -δ * sy3 + 2g * Ey * sz3
        k3y =  δ * sx3 - 2g * Ex * sz3
        k3z = 2g * (Ex * sy3 - Ey * sx3)

        # k4
        sx4 = sx + h * k3x
        sy4 = sy + h * k3y
        sz4 = sz + h * k3z

        k4x = -δ * sy4 + 2g * Ey * sz4
        k4y =  δ * sx4 - 2g * Ex * sz4
        k4z = 2g * (Ex * sy4 - Ey * sx4)

        ox = sx + h/6 * (k1x + 2k2x + 2k3x + k4x)
        oy = sy + h/6 * (k1y + 2k2y + 2k3y + k4y)
        oz = sz + h/6 * (k1z + 2k2z + 2k3z + k4z)
    return ox,oy,oz
end
function forward_kernel!(hx,hy,hz,E,delta,g,sx0,sy0,sz0,h,nsub,history)
    j=(blockIdx().x-1)*blockDim().x+threadIdx().x
    if j<=length(delta)
        sx,sy,sz=sx0[j],sy0[j],sz0[j]
        if history
            hx[j,1]=sx;hy[j,1]=sy;hz[j,1]=sz
        end
        for k in 1:length(E)
            ex,ey=real(E[k]),imag(E[k])
            for sub in 1:nsub
                sx,sy,sz=scalar_rk4(sx,sy,sz,ex,ey,h,delta[j],g[j])
            end
            if history
                hx[j,k+1]=sx;hy[j,k+1]=sy;hz[j,k+1]=sz
            end
        end
        if !history
            hx[j,1]=sx;hy[j,1]=sy;hz[j,1]=sz
        end
    end
    return
end
function fast_forward(E,delta,g,dt;n_sub=8,Sx0,Sy0,Sz0,history=true)
    M=length(delta);cols=history ? length(E)+1 : 1
    hx=CUDA.zeros(Float64,M,cols);hy=similar(hx);hz=similar(hx)
    de=CuArray(E)
    @cuda threads=256 blocks=cld(M,256) forward_kernel!(hx,hy,hz,de,delta,g,Sx0,Sy0,Sz0,dt/n_sub,n_sub,history)
    return hx,hy,hz
end
function backward_chunk!(gx,gy,hx,hy,hz,lx,ly,lz,E,delta,g,h,nsub,lo,hi)
    j=(blockIdx().x-1)*blockDim().x+threadIdx().x
    if j<=length(delta)
        ax,ay,az=lx[j],ly[j],lz[j]
        δ,gj=delta[j],g[j]
        for k in hi:-1:lo
            sx,sy,sz=hx[j,k+1],hy[j,k+1],hz[j,k+1]
            ex,ey=real(E[k]),imag(E[k]);dx=0.0;dy=0.0
            for sub in 1:nsub
                dx+=h*2gj*(-ay*sz+az*sy)
                dy+=h*2gj*(ax*sz-az*sx)
                sx,sy,sz=scalar_rk4(sx,sy,sz,ex,ey,-h,δ,gj)
                ax,ay,az=scalar_rk4(ax,ay,az,ex,ey,-h,δ,gj)
            end
            gx[j,k-lo+1]=dx;gy[j,k-lo+1]=dy
        end
        lx[j]=ax;ly[j]=ay;lz[j]=az
    end
    return
end

function fast_objective(E, delta_b, g_b, spin_weight, dt;
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

    Sx_hist, Sy_hist, Sz_hist = fast_forward(
        E, delta_b, g_b, dt;
        n_sub=n_sub, Sx0=Sx_d, Sy0=Sy_d, Sz0=Sz0, history=false)

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
function fast_gradient(E, delta_b, g_b, spin_weight, dt;
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

    Sx_hist, Sy_hist, Sz_hist = fast_forward(E, delta_b, g_b, dt;
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


    # Batch 32 time points; reduce on-device and transfer the gradient once.
    chunk=32
    gx=CUDA.zeros(Float64,M,chunk);gy=similar(gx)
    dx=CUDA.zeros(Float64,N);dy=similar(dx);de=CuArray(E)
    for hi in N:-chunk:1
        lo=max(1,hi-chunk+1);count=hi-lo+1
        @cuda threads=256 blocks=cld(M,256) backward_chunk!(
            gx,gy,Sx_hist,Sy_hist,Sz_hist,λx,λy,λz,de,delta_b,g_b,h,n_sub,lo,hi)
        view(dx,lo:hi).=vec(sum(view(gx,:,1:count);dims=1))
        view(dy,lo:hi).=vec(sum(view(gy,:,1:count);dims=1))
    end
    grad=Array(complex.(dx,dy))
    w_smooth > 0 && (grad .+= w_smooth .* frequency_smooth_gradient(
        E; λ_freq=1.0, control_mask=phase_weights))

    control_mask !== nothing && (grad[.!control_mask] .= 0)

    if n_edge > 0
        grad[1:n_edge] .= 0
        grad[end-n_edge+1:end] .= 0
    end

    return grad
end


# Same acceptance rules and checkpoint format as optimize_rase!.
function fast_optimize_rase!(E, delta_b, g_b, spin_weight, dt;
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

    J, Jpop_inside, Jpop_outside, Jcoh, Jsmooth, _, _ = fast_objective(
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
        grad = fast_gradient(E, delta_b, g_b, spin_weight, dt;
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
                fast_objective(E_trial, delta_b, g_b, spin_weight, dt;
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

end
