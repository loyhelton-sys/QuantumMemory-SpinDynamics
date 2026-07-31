# ============================================================
# 1st-order RHS
# ============================================================

function rhs_1st_order!(du, u, p, t)   # 计算1st-order微分方程的右边
    delta_b_gpu, M, E_of_t = p         # 解包参数（失谐、总自旋数、驱动脉冲）

    Sp, Sz = unpack_state_1st_order_u(u, M)
    dSp, dSz = unpack_state_1st_order_du(du, M)
    E_t = E_of_t(t)

    # --------------------------------------------------------
    # Spin equations
    # dSp/dt = iΔSp - 2iESz
    # dSz/dt = -iESp + iE*Sm
    # --------------------------------------------------------
    dSp .= 1im .* delta_b_gpu .* Sp .- 2im .* E_t .* Sz
    dSz .= -1im .* E_t .* Sp .+ 1im .* conj(E_t) .* conj.(Sp)

    return nothing
end