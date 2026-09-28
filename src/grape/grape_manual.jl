using CUDA

function validate_grape_inputs(E_control, delta_b, g_b, dt, n_sub)
    E_control isa AbstractVector && !isempty(E_control) ||
        error("E_control must be a non-empty vector.")
    E_control isa CuArray &&
        error("E_control must remain on the CPU; spin propagation arrays may be on the GPU.")
    all(isfinite, E_control) || error("All control samples must be finite.")
    delta_b isa AbstractVector && g_b isa AbstractVector ||
        error("delta_b and g_b must be vectors.")
    !isempty(delta_b) || error("delta_b and g_b cannot be empty.")
    length(delta_b) == length(g_b) ||
        error("delta_b and g_b must have equal lengths.")
    (delta_b isa CuArray) == (g_b isa CuArray) ||
        error("delta_b and g_b must use the same CPU or GPU backend.")
    all(isfinite, delta_b) && all(isfinite, g_b) ||
        error("All detuning and coupling values must be finite.")
    dt isa Real && isfinite(dt) && dt > 0 ||
        error("dt must be positive and finite.")
    n_sub isa Integer && n_sub > 0 ||
        error("n_sub must be a positive integer.")
    return nothing
end

function validate_grape_state_component(value, name, M)
    if value isa Number
        isfinite(value) || error("$(name) must be finite.")
    elseif value isa AbstractVector
        length(value) == M || error("$(name) must have length M.")
        all(isfinite, value) || error("All $(name) values must be finite.")
    else
        error("$(name) must be a finite scalar or a vector of length M.")
    end
    return nothing
end

function validate_grape_weights(spin_weight, delta_b)
    M = length(delta_b)
    spin_weight isa AbstractVector && length(spin_weight) == M ||
        error("spin_weight must be a vector of length M.")
    (spin_weight isa CuArray) == (delta_b isa CuArray) ||
        error("spin_weight and spin arrays must use the same backend.")
    all(isfinite, spin_weight) || error("All spin weights must be finite.")
    any(spin_weight .< 0) && error("spin_weight cannot contain negative values.")
    sum(spin_weight) > 0 || error("spin_weight must have a positive sum.")
    return nothing
end

# ============================================================
# Bloch RHS
# ============================================================
function bloch_rhs!(dSx, dSy, dSz, Sx, Sy, Sz, Ex, Ey, delta_b, g_b)
    dSx .= -delta_b .* Sy .+ 2 .* g_b .* Ey .* Sz
    dSy .=  delta_b .* Sx .- 2 .* g_b .* Ex .* Sz
    dSz .= 2 .* g_b .* (Ex .* Sy .- Ey .* Sx)
    return nothing
end

# ============================================================
# RK4
# ============================================================
struct RK4Workspace{T}
    k1x::T; k1y::T; k1z::T
    k2x::T; k2y::T; k2z::T
    k3x::T; k3y::T; k3z::T
    k4x::T; k4y::T; k4z::T
    tx::T; ty::T; tz::T
end

function RK4Workspace(x)
    a() = similar(x)
    return RK4Workspace(
        a(), a(), a(),
        a(), a(), a(),
        a(), a(), a(),
        a(), a(), a(),
        a(), a(), a(),
    )
end

function rk4_step!(Sx, Sy, Sz, Ex, Ey, h, delta_b, g_b, w::RK4Workspace)
    k1x, k1y, k1z = w.k1x, w.k1y, w.k1z
    k2x, k2y, k2z = w.k2x, w.k2y, w.k2z
    k3x, k3y, k3z = w.k3x, w.k3y, w.k3z
    k4x, k4y, k4z = w.k4x, w.k4y, w.k4z
    tx, ty, tz = w.tx, w.ty, w.tz

    bloch_rhs!(k1x, k1y, k1z, Sx, Sy, Sz, Ex, Ey, delta_b, g_b)

    tx .= Sx .+ h/2 .* k1x
    ty .= Sy .+ h/2 .* k1y
    tz .= Sz .+ h/2 .* k1z
    bloch_rhs!(k2x, k2y, k2z, tx, ty, tz, Ex, Ey, delta_b, g_b)

    tx .= Sx .+ h/2 .* k2x
    ty .= Sy .+ h/2 .* k2y
    tz .= Sz .+ h/2 .* k2z
    bloch_rhs!(k3x, k3y, k3z, tx, ty, tz, Ex, Ey, delta_b, g_b)

    tx .= Sx .+ h .* k3x
    ty .= Sy .+ h .* k3y
    tz .= Sz .+ h .* k3z
    bloch_rhs!(k4x, k4y, k4z, tx, ty, tz, Ex, Ey, delta_b, g_b)

    Sx .+= h/6 .* (k1x .+ 2 .* k2x .+ 2 .* k3x .+ k4x)
    Sy .+= h/6 .* (k1y .+ 2 .* k2y .+ 2 .* k3y .+ k4y)
    Sz .+= h/6 .* (k1z .+ 2 .* k2z .+ 2 .* k3z .+ k4z)

    return nothing
end

# ============================================================
# CUDA RK4 kernel
# ============================================================
function rk4_kernel!(Sx, Sy, Sz, Ex, Ey, h, delta_b, g_b, M)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if j <= M
        sx = Sx[j]
        sy = Sy[j]
        sz = Sz[j]

        δ = delta_b[j]
        g = g_b[j]

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

        Sx[j] = sx + h/6 * (k1x + 2k2x + 2k3x + k4x)
        Sy[j] = sy + h/6 * (k1y + 2k2y + 2k3y + k4y)
        Sz[j] = sz + h/6 * (k1z + 2k2z + 2k3z + k4z)
    end

    return
end

function rk4_step_gpu!(Sx, Sy, Sz, Ex, Ey, h, delta_b, g_b)
    M = length(Sx)
    threads = 256
    blocks = cld(M, threads)

    @cuda threads=threads blocks=blocks rk4_kernel!(
        Sx, Sy, Sz, Ex, Ey, h, delta_b, g_b, M
    )

    return nothing
end

# ============================================================
# Gradient contribution kernel
# ============================================================
function gradient_kernel!(gx, gy, Sx, Sy, Sz, λx, λy, λz, g_b, M)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if j <= M
        g = g_b[j]

        gx[j] = 2g * (-λy[j] * Sz[j] + λz[j] * Sy[j])
        gy[j] = 2g * ( λx[j] * Sz[j] - λz[j] * Sx[j])
    end

    return
end

function gradient_step_gpu!(gx, gy, Sx, Sy, Sz, λx, λy, λz, g_b)
    M = length(Sx)
    threads = 256
    blocks = cld(M, threads)

    @cuda threads=threads blocks=blocks gradient_kernel!(
        gx, gy, Sx, Sy, Sz, λx, λy, λz, g_b, M
    )

    return nothing
end

function gradient_accumulate_kernel!(gx, gy, Sx, Sy, Sz,
    λx, λy, λz, g_b, h, M)

    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if j <= M
        g = g_b[j]
        gx[j] += h * 2g * (-λy[j] * Sz[j] + λz[j] * Sy[j])
        gy[j] += h * 2g * ( λx[j] * Sz[j] - λz[j] * Sx[j])
    end
    return
end

function gradient_accumulate_step_gpu!(gx, gy, Sx, Sy, Sz,
    λx, λy, λz, g_b, h)

    M = length(Sx)
    threads = 256
    blocks = cld(M, threads)
    @cuda threads=threads blocks=blocks gradient_accumulate_kernel!(
        gx, gy, Sx, Sy, Sz, λx, λy, λz, g_b, h, M)
    return nothing
end

# ============================================================
# Forward propagation
# ============================================================
function grape_forward_manual(E_control, delta_b, g_b, dt;
    n_sub=4, Sx0=0.0, Sy0=0.0, Sz0=-0.5)

    validate_grape_inputs(E_control, delta_b, g_b, dt, n_sub)
    M = length(delta_b)
    validate_grape_state_component(Sx0, "Sx0", M)
    validate_grape_state_component(Sy0, "Sy0", M)
    validate_grape_state_component(Sz0, "Sz0", M)

    N = length(E_control)
    h = dt / n_sub

    Sx_hist = similar(delta_b, M, N + 1)
    Sy_hist = similar(delta_b, M, N + 1)
    Sz_hist = similar(delta_b, M, N + 1)

    fill!(Sx_hist, 0.0)
    fill!(Sy_hist, 0.0)
    fill!(Sz_hist, 0.0)

    Sx_hist[:, 1] .= Sx0
    Sy_hist[:, 1] .= Sy0
    Sz_hist[:, 1] .= Sz0

    w = RK4Workspace(delta_b)

    for k in 1:N
        Sx = copy(Sx_hist[:, k])
        Sy = copy(Sy_hist[:, k])
        Sz = copy(Sz_hist[:, k])

        Ex, Ey = real(E_control[k]), imag(E_control[k])

        for _ in 1:n_sub
            if Sx isa CuArray
                rk4_step_gpu!(Sx, Sy, Sz, Ex, Ey, h, delta_b, g_b)
            else
                rk4_step!(Sx, Sy, Sz, Ex, Ey, h, delta_b, g_b, w)
            end
        end

        Sx_hist[:, k+1] .= Sx
        Sy_hist[:, k+1] .= Sy
        Sz_hist[:, k+1] .= Sz
    end

    return Sx_hist, Sy_hist, Sz_hist
end

# ============================================================
# Objective
# J = -mean[(Sz(T)-target)^2]
# ============================================================
function grape_objective_manual(E_control, delta_b, g_b, dt;
    target_Sz, spin_weight, n_sub=4)

    validate_grape_weights(spin_weight, delta_b)
    validate_grape_state_component(target_Sz, "target_Sz", length(delta_b))

    _, _, Sz_hist = grape_forward_manual(
        E_control, delta_b, g_b, dt; n_sub=n_sub)

    Sz_final = @view Sz_hist[:, end]

    return -sum(spin_weight .* (Sz_final .- target_Sz).^2) /
            sum(spin_weight)
end

# ============================================================
# Manual adjoint gradient
# ============================================================
function grape_gradient_manual(E_control, delta_b, g_b, dt; 
        target_Sz, spin_weight, n_sub=4)

    validate_grape_weights(spin_weight, delta_b)
    validate_grape_state_component(target_Sz, "target_Sz", length(delta_b))
    M, N = length(delta_b), length(E_control)
    h = dt / n_sub

    Sx_hist, Sy_hist, Sz_hist = grape_forward_manual(E_control, delta_b, g_b, dt; n_sub=n_sub)

    λx = similar(delta_b, M)
    λy = similar(delta_b, M)
    λz = similar(delta_b, M)

    fill!(λx, 0.0)
    fill!(λy, 0.0)

    λz .= -2 .* spin_weight .* (Sz_hist[:, end] .- target_Sz) / sum(spin_weight)

    grad_Ex = zeros(Float64, N)
    grad_Ey = zeros(Float64, N)

    w_state = RK4Workspace(delta_b)
    w_costate = RK4Workspace(delta_b)

    gx = similar(delta_b, M)
    gy = similar(delta_b, M)

    for k in N:-1:1
        Ex, Ey = real(E_control[k]), imag(E_control[k])

        Sx = copy(Sx_hist[:, k+1])
        Sy = copy(Sy_hist[:, k+1])
        Sz = copy(Sz_hist[:, k+1])

        for _ in 1:n_sub
            if Sx isa CuArray
                gradient_step_gpu!(gx, gy, Sx, Sy, Sz, λx, λy, λz, g_b)

                grad_Ex[k] += h * sum(gx)
                grad_Ey[k] += h * sum(gy)
            else
                grad_Ex[k] += h * sum(2 .* g_b .* (-λy .* Sz .+ λz .* Sy))
                grad_Ey[k] += h * sum(2 .* g_b .* ( λx .* Sz .- λz .* Sx))
            end
            
            if Sx isa CuArray
                rk4_step_gpu!(Sx, Sy, Sz, Ex, Ey, -h, delta_b, g_b)
                rk4_step_gpu!(λx, λy, λz, Ex, Ey, -h, delta_b, g_b)
            else
                rk4_step!(Sx, Sy, Sz, Ex, Ey, -h, delta_b, g_b, w_state)
                rk4_step!(λx, λy, λz, Ex, Ey, -h, delta_b, g_b, w_costate)
            end
        end
    end

    return grad_Ex .+ 1im .* grad_Ey
end

# ============================================================
# One optimization step
# ============================================================
function grape_step_manual(E_control, delta_b, g_b, dt, η;
     target_Sz, spin_weight, n_sub=4)
    grad = grape_gradient_manual(E_control, delta_b, g_b, dt;
     target_Sz=target_Sz, spin_weight=spin_weight, n_sub=n_sub)
    return E_control .+ η .* grad
end

# ============================================================
# Coherence objective
# ============================================================
function grape_objective_coherence(E_control, delta_b, g_b, dt;
    spin_weight, n_sub=4,
    Sx0=0.5, Sy0=0.0, Sz0=0.0,
    phi_target=0.0, A_target=0.5)

    validate_grape_weights(spin_weight, delta_b)
    phi_target isa Real && isfinite(phi_target) || error("phi_target must be finite.")
    A_target isa Real && isfinite(A_target) && A_target > 0 ||
        error("A_target must be positive and finite.")

    Sx_hist, Sy_hist, _ = grape_forward_manual(
        E_control, delta_b, g_b, dt;
        n_sub=n_sub,
        Sx0=Sx0, Sy0=Sy0, Sz0=Sz0,)

    Sx_final = @view Sx_hist[:, end]
    Sy_final = @view Sy_hist[:, end]

    proj = cos(phi_target) .* Sx_final .+ sin(phi_target) .* Sy_final

    return sum(spin_weight .* proj) / (A_target * sum(spin_weight))
end

# ============================================================
# Coherence adjoint gradient
# ============================================================
function grape_gradient_coherence(E_control, delta_b, g_b, dt;
    spin_weight, n_sub=4,
    Sx0=0.5, Sy0=0.0, Sz0=0.0,
    phi_target=0.0, A_target=0.5)

    validate_grape_weights(spin_weight, delta_b)
    phi_target isa Real && isfinite(phi_target) || error("phi_target must be finite.")
    A_target isa Real && isfinite(A_target) && A_target > 0 ||
        error("A_target must be positive and finite.")

    M, N = length(delta_b), length(E_control)
    h = dt / n_sub

    Sx_hist, Sy_hist, Sz_hist = grape_forward_manual(
        E_control, delta_b, g_b, dt;
        n_sub=n_sub,
        Sx0=Sx0, Sy0=Sy0, Sz0=Sz0,)

    norm_w = A_target * sum(spin_weight)

    λx = similar(delta_b, M)
    λy = similar(delta_b, M)
    λz = similar(delta_b, M)

    λx .= spin_weight .* cos(phi_target) ./ norm_w
    λy .= spin_weight .* sin(phi_target) ./ norm_w
    fill!(λz, 0.0)

    grad_Ex = zeros(Float64, N)
    grad_Ey = zeros(Float64, N)

    w_state = RK4Workspace(delta_b)
    w_costate = RK4Workspace(delta_b)

    gx = similar(delta_b, M)
    gy = similar(delta_b, M)

    for k in N:-1:1
        Ex, Ey = real(E_control[k]), imag(E_control[k])

        Sx = copy(Sx_hist[:, k+1])
        Sy = copy(Sy_hist[:, k+1])
        Sz = copy(Sz_hist[:, k+1])

        for _ in 1:n_sub
            if Sx isa CuArray
                gradient_step_gpu!(gx, gy,
                    Sx, Sy, Sz,
                    λx, λy, λz,
                    g_b,)

                grad_Ex[k] += h * sum(gx)
                grad_Ey[k] += h * sum(gy)
            else
                grad_Ex[k] += h * sum(2 .* g_b .* (-λy .* Sz .+ λz .* Sy))
                grad_Ey[k] += h * sum(2 .* g_b .* (λx .* Sz .- λz .* Sx))
            end

            if Sx isa CuArray
                rk4_step_gpu!(Sx, Sy, Sz,
                    Ex, Ey, -h,
                    delta_b, g_b,)

                rk4_step_gpu!(λx, λy, λz,
                    Ex, Ey, -h,
                    delta_b, g_b,)
            else
                rk4_step!(Sx, Sy, Sz,
                    Ex, Ey, -h,
                    delta_b, g_b,
                    w_state,)

                rk4_step!(λx, λy, λz,
                    Ex, Ey, -h,
                    delta_b, g_b,
                    w_costate,)
            end
        end
    end

    return grad_Ex .+ 1im .* grad_Ey
end