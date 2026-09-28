# ============================================================
# MAIN RUN FUNCTION FOR FIRST-ORDER SIMULATION
# ============================================================
function run_sim_1st_order(SIM_SETTING, SYSTEM_CONFIG, PULSE_CONFIG;
    clean_gpu=false, verbose=true,)
    # ---------------------------------------------------------
    # BUILD AND VALIDATE CONFIGURATION
    # ---------------------------------------------------------
    CONFIG = build_full_config(SIM_SETTING, SYSTEM_CONFIG,)

    validate_config(CONFIG)
    validate_pulse_config(PULSE_CONFIG)
    if CONFIG.saved_file_name !== nothing
        mkpath(dirname(CONFIG.saved_file_name))
    end
    
    d = prepare_derived(CONFIG)
    M = d.M
    timespan = (0.0, CONFIG.Ttotal)
    t_saved = collect(range(0.0, CONFIG.Ttotal; length=CONFIG.Nt_save))
    Nt = length(t_saved)

    E_of_t = build_E_of_t(PULSE_CONFIG)

    # ---------------------------------------------------------
    # INITIAL CONDITION AND GPU PARAMETERS
    # ---------------------------------------------------------
    initial_condition = get_initial_condition(CONFIG)

    delta_b_gpu = CuArray(Float64.(d.delta_b))
    g_b_gpu     = CuArray(Float64.(d.g_b))

    u0 = build_initial_state_1st_order(d.Nj, initial_condition;
        delta_b = d.delta_b,)
    u0_gpu = CuArray(u0)
    p_gpu = (delta_b = delta_b_gpu, g_b = g_b_gpu,
        M = M, E_of_t = E_of_t,)

    prob_gpu = ODEProblem(rhs_1st_order!, u0_gpu, timespan, p_gpu)

    # ---------------------------------------------------------
    # ENSEMBLE DIMENSIONS
    # ---------------------------------------------------------
    M_delta = length(d.delta_b_1d)
    M_g     = length(d.g_b_1d)
    @assert M == M_delta * M_g ("The flattened ensemble size is inconsistent. " *
        "M = $M, while M_delta × M_g = $(M_delta * M_g).")

    # ---------------------------------------------------------
    # MAIN SAVE ARRAYS
    # ---------------------------------------------------------

    # Collective spin values at every saved time.
    Σp_save = Vector{ComplexF64}(undef, Nt)
    Σz_save = Vector{ComplexF64}(undef, Nt)

    # Rows: flattened ensemble bins in g-fast order.
    # Columns: saved times.
    Sp_keep = Matrix{ComplexF64}(undef, M, Nt)
    Sz_keep = Matrix{ComplexF64}(undef, M, Nt)

    state_ranges = state_ranges_1st_order(M)
    range_Sp = state_ranges.Sp
    range_Sz = state_ranges.Sz

    # ---------------------------------------------------------
    # CALLBACK
    # ---------------------------------------------------------
    kref = Ref(0)

    function affect!(integrator)
        kref[] += 1
        k = kref[]
        if verbose && k % 100 == 0
            println("Progress: $k / $Nt  ($(round(100k/Nt, digits=1))%)")
        end
        # One GPU-to-CPU transfer per saved time point.
        u_cpu = Array(integrator.u)
        Sp_cpu = @view u_cpu[range_Sp]
        Sz_cpu = @view u_cpu[range_Sz]

        Sp_keep[:, k] .= Sp_cpu
        Sz_keep[:, k] .= Sz_cpu
        Σp_save[k] = sum(Sp_cpu)
        Σz_save[k] = sum(Sz_cpu)

        return nothing
    end

    cb = PresetTimeCallback(t_saved, affect!; save_positions = (false, false),)

    # ---------------------------------------------------------
    # SOLVE
    # ---------------------------------------------------------
    t0 = time_ns()
    sol_gpu = CUDA.allowscalar() do
        solve(prob_gpu, Tsit5();
            reltol = CONFIG.reltol, abstol = CONFIG.abstol, callback = cb,
            save_on = false, save_everystep = false, dense = false,)
    end

    CUDA.synchronize()

    elapsed_seconds = (time_ns() - t0) / 1e9
    verbose && println("Callback saved $(kref[]) / $Nt requested time points")

    kref[] == Nt || error("Callback saved $(kref[]) points, but expected $Nt.")
    verbose && println("Time taken: $elapsed_seconds seconds")

    # ---------------------------------------------------------
    # POST-PROCESS OBSERVABLES
    # ---------------------------------------------------------
    E_of_t_arr = E_of_t.(t_saved)

    Σx_save = real.(Σp_save)
    Σy_save = imag.(Σp_save)


    # ---------------------------------------------------------
    # SAVE DATA
    # ---------------------------------------------------------
    data = (SIM_SETTING = SIM_SETTING, SYSTEM_CONFIG = SYSTEM_CONFIG, PULSE_CONFIG = PULSE_CONFIG,
        t_saved = t_saved,
        Σp_sol = Σp_save, Σz_sol = Σz_save,
        Σx_sol = Σx_save, Σy_sol = Σy_save,
        
        E_of_t_arr = E_of_t_arr, M_delta = M_delta, M_g = M_g, M_total = M,
        edges_delta = d.edges_delta, delta_b_1d = d.delta_b_1d, p_delta = d.p_delta,
        edges_g = d.edges_g, g_b_1d = d.g_b_1d, p_g = d.p_g,
        delta_b = d.delta_b, g_b = d.g_b, Nj = d.Nj, Nj_2d = d.Nj_2d,

        Sp_keep = Sp_keep, Sz_keep = Sz_keep,

        N_total = d.N_total, elapsed_seconds = elapsed_seconds,)

    if CONFIG.saved_file_name !== nothing
        filename = CONFIG.saved_file_name
        @save filename data
        verbose && println("Saving to: ", filename)
    end

    # ---------------------------------------------------------
    # GPU CLEANUP
    # ---------------------------------------------------------
    if clean_gpu
        verbose && println("Cleaning GPU memory...")
        u0_gpu = nothing
        delta_b_gpu = nothing
        p_gpu = nothing
        prob_gpu = nothing
        sol_gpu = nothing
        cb = nothing
        GC.gc()
        CUDA.reclaim()
        verbose && println("GPU memory cleanup finished.")
    end

    return data
end