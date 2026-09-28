# ============================================================
# FIRST-ORDER SPIN EQUATIONS
# ============================================================

function rhs_1st_order!(du, u, p, t)
    delta_b = p.delta_b
    g_b = p.g_b
    M = p.M
    E_of_t = p.E_of_t

    Sp, Sz = unpack_state_1st_order(u, M)
    dSp, dSz = unpack_state_1st_order(du, M)
    E_t = E_of_t(t)

    # dSp/dt = i delta Sp - 2i g E Sz
    # dSz/dt = -i g conj(E) Sp + i g E conj(Sp)
    # Broadcasting is fused so no temporary Rabi-frequency array is allocated.
    dSp .= 1im .* delta_b .* Sp .- 2im .* g_b .* E_t .* Sz
    dSz .= -1im .* g_b .* conj(E_t) .* Sp .+
            1im .* g_b .* E_t .* conj.(Sp)

    return nothing
end
