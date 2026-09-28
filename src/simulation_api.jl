function run_simulation(SIM_SETTING, SYSTEM_CONFIG, PULSE_CONFIG;
    clean_gpu=false, verbose=true,)
    get_simulation_order(SIM_SETTING)
    verbose && println("Start running 1st-order spin-ensemble simulation...")
    return run_sim_1st_order(
        SIM_SETTING,
        SYSTEM_CONFIG,
        PULSE_CONFIG;
        clean_gpu = clean_gpu,
        verbose = verbose,
    )
end

function build_full_config(SIM_SETTING, SYSTEM_CONFIG)
    overlapping_fields = intersect(collect(keys(SIM_SETTING)), collect(keys(SYSTEM_CONFIG)))
    isempty(overlapping_fields) || error(
        "SIM_SETTING and SYSTEM_CONFIG contain duplicate fields: " *
        join(string.(overlapping_fields), ", "),
    )
    return merge(SIM_SETTING, SYSTEM_CONFIG)
end
