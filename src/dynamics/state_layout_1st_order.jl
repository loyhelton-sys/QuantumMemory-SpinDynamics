# ============================================================
# FIRST-ORDER STATE LAYOUT
#
# u = [Splus[1:M]; Sz[1:M]]
# ============================================================

function validate_state_bin_count_1st_order(M)
    M isa Integer && M > 0 || error("M must be a positive integer.")
    return M
end

function state_length_1st_order(M)
    validate_state_bin_count_1st_order(M)
    return 2M
end

function state_ranges_1st_order(M)
    validate_state_bin_count_1st_order(M)
    return (Sp = 1:M, Sz = M+1:2M)
end

@views function unpack_state_1st_order(state, M)
    ranges = state_ranges_1st_order(M)
    length(state) == 2M ||
        error("First-order state length must equal 2M = $(2M), but received $(length(state)).")

    return state[ranges.Sp], state[ranges.Sz]
end
