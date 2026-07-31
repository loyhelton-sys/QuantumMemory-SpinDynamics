# ============================================================
# STATE LAYOUT
# ============================================================

const IDX1_Sp_start = 1                  # 定义Sp=Sx+iSy的起始索引
idx1_Sz_start(M) = IDX1_Sp_start + M     # 计算Sz的起始索引，取决于M的值
state_length_1st_order(M) = 2M           # 整个状态向量的长度为2M

@views function unpack_state_1st_order_u(u, M)   # 从状态向量u中取出Sp和Sz
    idx = IDX1_Sp_start
    Sp = u[idx : idx + M - 1]   # 取出前M个元素作为Sp

    idx = idx1_Sz_start(M)
    Sz = u[idx : idx + M - 1]   # 取出接下来的M个元素作为Sz

    return Sp, Sz
end

@views function unpack_state_1st_order_du(du, M)  # 从导数向量du中取出dSp和dSz
    idx = IDX1_Sp_start
    dSp = du[idx : idx + M - 1]   # 取出前M个元素作为dSp

    idx = idx1_Sz_start(M)
    dSz = du[idx : idx + M - 1]   # 取出接下来的M个元素作为dSz

    return dSp, dSz
end