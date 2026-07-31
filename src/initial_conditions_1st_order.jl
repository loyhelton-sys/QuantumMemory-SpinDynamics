# ============================================================
# INITIAL CONDITIONS
# ============================================================

function build_u0_gpu_1st_order(M, Nj, initial_condition)  # 根据指定的初始条件构建初始状态向量u0
    if initial_condition == :ground                        # 如果初始条件为基态，则调用build_u0_gpu_1st_order_ground函数
        return build_u0_gpu_1st_order_ground(M, Nj)

    elseif initial_condition == :inverted                  # 如果初始条件为反转态，则调用build_u0_gpu_1st_order_inverted函数
        return build_u0_gpu_1st_order_inverted(M, Nj)

    elseif initial_condition == :custom                    # 如果初始条件为自定义，则调用build_u0_gpu_1st_order_custom函数
        return build_u0_gpu_1st_order_custom(M, Nj)

    else
        error("Unknown initial_condition = $(initial_condition). " *
            "Use :ground, :inverted, or :custom.")
    end
end

function build_u0_gpu_1st_order_ground(M, Nj)
    u0 = zeros(ComplexF64, state_length_1st_order(M))
    idx = IDX1_Sp_start

    # S+
    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    # Sz
    u0[idx:idx+M-1] .= - Nj ./ 2
    idx += M

    return CuArray(u0)
end

function build_u0_gpu_1st_order_inverted(M, Nj)
    u0 = zeros(ComplexF64, state_length_1st_order(M))
    idx = IDX1_Sp_start

    # S+
    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    # Sz
    u0[idx:idx+M-1] .= Nj ./ 2
    idx += M

    return CuArray(u0)
end

function build_u0_gpu_1st_order_custom(M, Nj)
    u0 = zeros(ComplexF64, state_length_1st_order(M))
    idx = IDX1_Sp_start

    # S+
    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    # Sz
    u0[idx:idx+M-1] .= 0.0 + 0.0im
    idx += M

    return CuArray(u0)
end