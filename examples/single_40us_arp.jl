using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using CUDA, JLD2, SpinDynamics
include(joinpath(@__DIR__, "..", "src/plotting/PlotPulse.jl"))
include(joinpath(@__DIR__, "..", "src/plotting/PlotResults.jl"))

function main()
    source = joinpath(@__DIR__, "..", "Results/10-20-10/three_pulse_rase_optimization/adaptive_mean_smooth/best_feasible.jld2")
    m = load(source, "checkpoint_metadata")
    p = m.original_pulse_parameters
    duration = 40e-6
    pulse_parameters = (; t_center=duration/2, duration, amp=p.amp,
        bandwidth=p.bandwidth, n=p.n, omega0=p.omega0,
        chirp_sign=p.chirp_sign, phase0=p.phase0, edge_frac=p.edge_frac)
    drive = wurst_drive(; pulse_parameters...)
    t = [(k-0.5)*m.dt for k in 1:round(Int,duration/m.dt)]
    E = ComplexF64.(drive.(t))
    @assert length(E) == 1600
    @assert isapprox(maximum(abs,E),p.amp;rtol=1e-10)
    out = joinpath(@__DIR__, "..", "Results/40us_single_ARP")
    mkpath(out)
    PlotPulse.plot_pulse(E,m.dt;amplitude_threshold=0.01,
        filename=joinpath(out,"pulse.png"),label="Single 40 us ARP",
        title="Single 40 us WURST ARP — amplitude 4.5, bandwidth 1.6 MHz")
    ens = build_equal_weight_ensemble(m.freq_cfg,m.g_cfg,m.M_delta,m.M_g)
    delta,g,sw = CuArray(ens.delta_b),CuArray(ens.g_b),CuArray(ens.Nj ./ ens.N_total)
    # Match the earlier RASE Sz initial state and pre-pulse evolution exactly.
    n_sub = 32
    _,Jpop_inside,Jpop_outside,Jcoh,Jsmooth,Sp_echo,Sz_final = rase_objective(
        E,delta,g,sw,m.dt;delta_window=m.delta_window,t_dephase=m.t_dephase,
        t_echo_free=m.t_echo_free,Sz_init=m.Sz_init,Sz_target=m.Sz_target,n_sub)
    @assert all(isfinite,Array(Sz_final))
    @assert maximum(abs,Array(Sz_final)) <= 0.5+1e-8
    PlotResults.plot_sz_final_map(delta,g,Sz_final;M_delta=m.M_delta,M_g=m.M_g,
        delta_window_MHz=m.delta_window/(2pi*1e6),filename=joinpath(out,"final_sz.png"),
        title="Final Sz — single 40 us ARP (same RASE initial state)")
    simulation_metadata = (;dt=m.dt,n_sub,M_delta=m.M_delta,M_g=m.M_g,
        freq_cfg=m.freq_cfg,g_cfg=m.g_cfg,delta_window=m.delta_window,
        t_dephase=m.t_dephase,t_echo_free=m.t_echo_free,Sz_init=m.Sz_init,Sz_target=m.Sz_target)
    jldsave(joinpath(out,"single_arp_result.jld2");E,t,pulse_parameters,simulation_metadata,
        source_metadata_file=source,original_pulse_parameters=p,
        delta=Array(delta),g=Array(g),spin_weight=Array(sw),Sz_final=Array(Sz_final),
        Sp_echo=Array(Sp_echo),Jpop_inside,Jpop_outside,Jcoh,Jsmooth)
    open(joinpath(out,"pulse.csv"),"w") do io
        println(io,"time_s,E_real,E_imag,amplitude")
        for k in eachindex(E)
            println(io,join((t[k],real(E[k]),imag(E[k]),abs(E[k])),','))
        end
    end
    report = "Single 40 us WURST ARP; no optimization.\nParameters: $pulse_parameters\n" *
        "Frequency sweep: -0.8 to +0.8 MHz; dt=$(m.dt) s; samples=$(length(E)); n_sub=$n_sub.\n" *
        "Same ensemble and RASE initial state as original 10-20-10: inside +/-0.3 MHz Sz=-0.499 with positive Sx=sqrt(0.25-Sz^2), outside Sz=-0.5; pre-pulse free evolution 5 us.\n" *
        "Peak amplitude=$(maximum(abs,E)); Jpop_inside=$Jpop_inside; Jpop_outside=$Jpop_outside.\n" *
        "Same relative edge_frac=0.01 (0.4 us for this pulse); sweep slope=0.04 MHz/us.\n" *
        "Files: pulse.png, final_sz.png, single_arp_result.jld2, pulse.csv.\n"
    write(joinpath(out,"status.txt"),report)
    println(report)
end
main()
