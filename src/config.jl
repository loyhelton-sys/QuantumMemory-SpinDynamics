function validate_config(CONFIG)
    required_fields = (
        :simulation_order, :N, :M_delta, :M_g, :Ttotal, :Nt_save,
        :reltol, :abstol, :initial_condition, :saved_file_name,
        :freq_inhomogeneity, :g_inhomogeneity,
    )

    for field in required_fields
        hasproperty(CONFIG, field) || error("CONFIG must contain $(field).")
    end

    CONFIG.N isa Real && isfinite(CONFIG.N) && CONFIG.N > 0 || error("N must be finite and positive.")
    CONFIG.M_delta isa Integer && CONFIG.M_delta > 0 || error("M_delta must be a positive integer.")
    CONFIG.M_g isa Integer && CONFIG.M_g > 0 || error("M_g must be a positive integer.")
    CONFIG.Ttotal isa Real && isfinite(CONFIG.Ttotal) && CONFIG.Ttotal > 0 || error("Ttotal must be finite and positive.")
    CONFIG.Nt_save isa Integer && CONFIG.Nt_save > 1 || error("Nt_save must be an integer larger than 1.")
    CONFIG.reltol isa Real && isfinite(CONFIG.reltol) && CONFIG.reltol > 0 || error("reltol must be finite and positive.")
    CONFIG.abstol isa Real && isfinite(CONFIG.abstol) && CONFIG.abstol > 0 || error("abstol must be finite and positive.")
    (isnothing(CONFIG.saved_file_name) || CONFIG.saved_file_name isa AbstractString) ||
        error("saved_file_name must be a string or nothing.")

    validate_simulation_order(CONFIG)
    validate_frequency_inhomogeneity(CONFIG.freq_inhomogeneity)
    validate_coupling_inhomogeneity(CONFIG.g_inhomogeneity)
    validate_initial_condition_config(CONFIG.initial_condition)

    if CONFIG.freq_inhomogeneity.kind == :constant && CONFIG.M_delta != 1
        error("Constant frequency inhomogeneity requires M_delta = 1.")
    end

    if CONFIG.g_inhomogeneity.kind == :constant && CONFIG.M_g != 1
        error("Constant coupling requires M_g = 1.")
    end
    return nothing
end

function validate_pulse_config(PULSE_CONFIG)
    for (index, cfg) in pairs(PULSE_CONFIG)
        hasproperty(cfg, :kind) || error("Pulse $(index) must have a kind.")
        try
            build_drive_pulse(cfg)
        catch err
            error("Invalid pulse $(index): $(sprint(showerror, err))")
        end
    end
    return nothing
end

function validate_simulation_order(CONFIG)
    hasproperty(CONFIG, :simulation_order) || error("CONFIG must contain simulation_order.")
    order = CONFIG.simulation_order
    order in (:first_order, :order1, :first, 1) ||
        error("Unknown simulation_order = $(order). Only first-order dynamics is currently supported.")
    return nothing
end

function validate_initial_condition_config(cfg)
    if cfg isa Symbol
        cfg in (:ground, :inverted, :mixed) ||
            error("Unknown initial_condition = $(cfg). RASE and custom states require explicit configuration.")
        return nothing
    end

    hasproperty(cfg, :kind) || error("initial_condition configuration must contain kind.")
    if cfg.kind in (:ground, :inverted, :mixed)
        return nothing

    elseif cfg.kind == :rase
        hasproperty(cfg, :delta_window) || error("RASE initial condition requires delta_window.")
        cfg.delta_window isa Real && isfinite(cfg.delta_window) && cfg.delta_window > 0 ||
            error("RASE delta_window must be finite and positive.")
        hasproperty(cfg, :Sz_init) || error("RASE initial condition requires Sz_init.")
        cfg.Sz_init isa Real && isfinite(cfg.Sz_init) && -0.5 <= cfg.Sz_init <= 0.5 ||
            error("RASE Sz_init must be finite and lie in [-0.5, 0.5].")
        if hasproperty(cfg, :phase)
            cfg.phase isa Real && isfinite(cfg.phase) || error("RASE phase must be finite.")
        end

    elseif cfg.kind == :custom
        hasproperty(cfg, :f) || error("Custom initial condition requires function f.")
        cfg.f isa Function || error("Custom initial-condition f must be a function.")
    else
        error("Unknown initial-condition kind: $(cfg.kind).")
    end
    return nothing
end

function get_initial_condition(CONFIG)
    hasproperty(CONFIG, :initial_condition) || error("CONFIG must contain initial_condition.")
    validate_initial_condition_config(CONFIG.initial_condition)
    return CONFIG.initial_condition
end

function get_simulation_order(CONFIG)
    validate_simulation_order(CONFIG)
    return CONFIG.simulation_order
end
