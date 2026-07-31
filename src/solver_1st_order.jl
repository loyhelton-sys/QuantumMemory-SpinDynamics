# MAIN RUN FUNCTION FOR FIRST-ORDER SIMULATION

function run_sim_1st_order(SIM_SETTING, SYSTEM_CONFIG, PULSE_CONFIG; clean_gpu = true,)
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
    t_saved = d.t_save
    Nt = d.Nt

    E_of_t = build_E_of_t(PULSE_CONFIG)
    println(E_of_t(0.0))
    println(E_of_t(50e-6))

    # ---------------------------------------------------------
    # INITIAL CONDITION AND GPU PARAMETERS
    # ---------------------------------------------------------
    initial_condition = get_initial_condition(CONFIG)
    println("Nj = ", d.Nj)
    u0_gpu = build_u0_gpu_1st_order(M, d.Nj, initial_condition,)
    delta_b_gpu = CuArray(Float64.(d.delta_b))
    p_gpu = (delta_b_gpu, M, E_of_t,)
    prob_gpu = ODEProblem(rhs_1st_order!, u0_gpu, d.timespan, p_gpu,)

    # ---------------------------------------------------------
    # ENSEMBLE DIMENSIONS
    # ---------------------------------------------------------
    M_delta = length(d.delta_b_1d)
    M_g     = length(d.g_b_1d)
    @assert M == M_delta * M_g ("The flattened ensemble size is inconsistent. " *
        "M = $M, while M_delta × M_g = $(M_delta * M_g).")

    # ---------------------------------------------------------
    # RESONANT-DELTA BIN FOR EVERY g VALUE
    # ---------------------------------------------------------

    # Select the detuning bin nearest delta = 0.
    idelta_res = argmin(abs.(d.delta_b_1d))
    delta_res  = d.delta_b_1d[idelta_res]

    # The flattened spin arrays follow Julia column-major order: 
    # flat_index = idelta + (ig - 1) * M_delta
    # Therefore, the resonant-delta indices form this strided range:
    keep_range = idelta_res:M_delta:M
    keep_bins  = collect(keep_range)
    @assert length(keep_bins) == M_g

    g_keep     = collect(d.g_b_1d)
    delta_keep = fill(delta_res, M_g)

    println("Selected resonant-detuning bin:")
    println("  idelta_res = $idelta_res")
    println("  delta_res / 2π = $(delta_res / (2π)) Hz")
    println("  number of g bins = $M_g")

    # ---------------------------------------------------------
    # MAIN SAVE ARRAYS
    # ---------------------------------------------------------

    # Collective spin values at every saved time.
    Σp_save = Vector{ComplexF64}(undef, Nt)
    Σz_save = Vector{ComplexF64}(undef, Nt)

    # Resonant-delta trajectories.
    # Rows:    g bins
    # Columns: saved times
    Sp_keep = Matrix{ComplexF64}(undef, M_g, Nt)
    Sz_keep = Matrix{ComplexF64}(undef, M_g, Nt)

    range_Sp = (IDX1_Sp_start : IDX1_Sp_start + M - 1)
    range_Sz = (idx1_Sz_start(M) : idx1_Sz_start(M) + M - 1)

    # ---------------------------------------------------------
    # CALLBACK
    # ---------------------------------------------------------
    kref = Ref(0)

    function affect!(integrator)
        kref[] += 1
        k = kref[]
        u = integrator.u

        # GPU views of all spin bins(no CPU transfer)
        Sp_gpu = @view u[range_Sp]
        Sz_gpu = @view u[range_Sz]

        # Collective spin
        Σp_save[k] = sum(Sp_gpu)
        Σz_save[k] = sum(Sz_gpu)

        # Save resonant-detuning trajectory (M_g CPU transfers only)
        Sp_keep[:, k] .= Array(@view Sp_gpu[keep_range])
        Sz_keep[:, k] .= Array(@view Sz_gpu[keep_range])

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
    println("Callback saved $(kref[]) / $Nt requested time points")

    kref[] == Nt || error("Callback saved $(kref[]) points, but expected $Nt.")
    println("Time taken: $elapsed_seconds seconds")

    # ---------------------------------------------------------
    # POST-PROCESS OBSERVABLES
    # ---------------------------------------------------------
    E_of_t_arr = E_of_t.(t_saved)

    Σx_save = real.(Σp_save)
    Σy_save = imag.(Σp_save)

    Sx_keep = real.(Sp_keep)
    Sy_keep = imag.(Sp_keep)

    # ---------------------------------------------------------
    # SAVE DATA
    # ---------------------------------------------------------
    data = (SIM_SETTING = SIM_SETTING, SYSTEM_CONFIG = SYSTEM_CONFIG, PULSE_CONFIG = PULSE_CONFIG,
        t_saved = t_saved,
        Σp_sol = Σp_save, Σz_sol = Σz_save,
        Σx_sol = Σx_save, Σy_sol = Σy_save,
        
        E_of_t_arr = E_of_t_arr,
        M_delta = M_delta, M_g = M_g, M_total = M,
        delta_b_1d = d.delta_b_1d, g_b_1d = d.g_b_1d, Nj_2d = d.Nj_2d,

        idelta_res = idelta_res, delta_res = delta_res,
        keep_bins = keep_bins, g_keep = g_keep, delta_keep = delta_keep,

        Sp_keep = Sp_keep, Sz_keep = Sz_keep,
        Sx_keep = Sx_keep, Sy_keep = Sy_keep,

        N_total = d.N_total, elapsed_seconds = elapsed_seconds,)

    if CONFIG.saved_file_name !== nothing
        filename = CONFIG.saved_file_name
        @save filename data
        println()
        println("Saving to: ", filename)
    end

    # ---------------------------------------------------------
    # GPU CLEANUP
    # ---------------------------------------------------------
    if clean_gpu
        println("Cleaning GPU memory...")
        u0_gpu = nothing
        delta_b_gpu = nothing
        p_gpu = nothing
        prob_gpu = nothing
        sol_gpu = nothing
        cb = nothing
        GC.gc()
        CUDA.reclaim()

        println("GPU memory cleanup finished.")
    end

    return data
end