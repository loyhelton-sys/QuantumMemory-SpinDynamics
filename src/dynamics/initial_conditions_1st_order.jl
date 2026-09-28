# ============================================================
# FIRST-ORDER INITIAL CONDITIONS
# ============================================================
function initial_condition_kind(initial_condition)
    if initial_condition isa Symbol
        return initial_condition
    elseif hasproperty(initial_condition, :kind)
        return initial_condition.kind
    else
        error("initial_condition must be a Symbol or contain a kind field.")
    end
end

function validate_initial_populations(Nj)
    isempty(Nj) && error("Nj cannot be empty.")
    all(x -> x isa Real && isfinite(x) && x >= 0, Nj) ||
        error("All Nj values must be finite and non-negative.")
    return nothing
end

function validate_initial_components(Sp, Sz, M)
    length(Sp) == M || error("Custom Sp must have length M.")
    length(Sz) == M || error("Custom Sz must have length M.")
    all(isfinite, Sp) || error("All custom Sp values must be finite.")
    all(isfinite, Sz) || error("All custom Sz values must be finite.")
    return nothing
end

function build_initial_state_1st_order(Nj, initial_condition; delta_b=nothing)
    validate_initial_populations(Nj)
    M = length(Nj)
    kind = initial_condition_kind(initial_condition)

    u0 = zeros(ComplexF64, state_length_1st_order(M))
    Sp, Sz = unpack_state_1st_order(u0, M)

    if kind == :ground
        Sz .= -Float64.(Nj) .* 0.5

    elseif kind == :inverted
        Sz .= Float64.(Nj) .* 0.5

    elseif kind == :mixed
        # Sp = Sz = 0 is the center of the Bloch sphere.

    elseif kind == :rase
        initial_condition isa Symbol &&
            error(":rase requires delta_window, Sz_init, and optionally phase.")
        delta_b === nothing && error("delta_b is required for :rase.")
        length(delta_b) == M || error("length(delta_b) must equal length(Nj).")
        all(isfinite, delta_b) || error("All delta_b values must be finite.")

        hasproperty(initial_condition, :delta_window) ||
            error("RASE initial condition requires delta_window.")
        hasproperty(initial_condition, :Sz_init) ||
            error("RASE initial condition requires Sz_init.")

        delta_window = initial_condition.delta_window
        Sz_init = initial_condition.Sz_init
        phase = get(initial_condition, :phase, 0.0)

        delta_window isa Real && isfinite(delta_window) && delta_window > 0 ||
            error("RASE delta_window must be finite and positive.")
        Sz_init isa Real && isfinite(Sz_init) && -0.5 <= Sz_init <= 0.5 ||
            error("RASE Sz_init must lie between -0.5 and 0.5.")
        phase isa Real && isfinite(phase) || error("RASE phase must be finite.")

        inside = abs.(delta_b) .< delta_window
        Sp_single = sqrt(0.25 - Sz_init^2) * cis(phase)
        populations = Float64.(Nj)
        Sp .= ifelse.(inside, Sp_single .* populations, 0.0 + 0.0im)
        Sz .= ifelse.(inside, Sz_init .* populations, -0.5 .* populations)

    elseif kind == :custom
        initial_condition isa Symbol && error(":custom requires a function f.")
        hasproperty(initial_condition, :f) || error("Custom initial condition requires f.")
        initial_condition.f isa Function || error("Custom initial-condition f must be a function.")

        custom_Sp, custom_Sz = initial_condition.f(Float64.(Nj), delta_b)
        validate_initial_components(custom_Sp, custom_Sz, M)
        Sp .= custom_Sp
        Sz .= custom_Sz

    else
        error("Unknown initial condition kind = $(kind). Use :ground, :inverted, :mixed, :rase, or :custom.")
    end

    return u0
end
