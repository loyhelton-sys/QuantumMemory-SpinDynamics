# Final figures for the user's selected, stronger-smoothed control.
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using CUDA, JLD2, SpinDynamics, Printf
include(joinpath(@__DIR__, "..", "src", "plotting", "PlotPulse.jl"))
include(joinpath(@__DIR__, "..", "src", "plotting", "PlotResults.jl"))
function main()
    root=joinpath(@__DIR__,"..","Results","40us_single_ARP","smooth_edges_recovery")
    source=joinpath(root,get(ENV,"SPINDYNAMICS_PLOT_SOURCE","final_result.jld2"))
    out=joinpath(root,get(ENV,"SPINDYNAMICS_PLOT_DIR","recovered_figures"));mkpath(out)
    d=load(source); E=d["E"];m=d["checkpoint_metadata"]
    n_sub=max(8,m.n_sub)
    ens=build_equal_weight_ensemble(m.freq_cfg,m.g_cfg,m.M_delta,m.M_g)
    delta,g,sw=CuArray(ens.delta_b),CuArray(ens.g_b),CuArray(ens.Nj ./ ens.N_total)
    _,Jpop_inside,Jpop_outside,Jcoh,Jsmooth,Sp_echo,Sz_final=rase_objective(E,delta,g,sw,m.dt;
        delta_window=m.delta_window,t_dephase=m.t_dephase,t_echo_free=m.t_echo_free,
        Sz_init=m.Sz_init,Sz_target=m.Sz_target,n_sub,phase_weights=d["phase_weights"])
    PlotPulse.plot_pulse(E,m.dt;reference=d["E_before_cleanup"], reference_label="Before waveform repair",label="Repaired control",amplitude_threshold=0.01,
        filename=joinpath(out,"pulse.png"),title="Repaired 40 us RASE pulse")
    nzoom=min(length(E),round(Int,1e-6/m.dt)+1)
    for (name,indices) in (("head",1:nzoom),("tail",length(E)-nzoom+1:length(E)))
        PlotPulse.plot_pulse(E[indices],m.dt;reference=d["E_before_cleanup"][indices],
            reference_label="Before repair",label="Repaired",t_start=(first(indices)-1)*m.dt,
            amplitude_threshold=0.01,filename=joinpath(out,"$(name)_detail.png"),
            title="Amplitude edge and chirp: $name")
    end
    PlotResults.plot_sz_final_map(delta,g,Sz_final;M_delta=m.M_delta,M_g=m.M_g,
        delta_window_MHz=m.delta_window/(2pi*1e6),filename=joinpath(out,"final_sz.png"),
        title="Final Sz — repaired 40 us control")
    timeline_end=m.t_dephase+m.T_pulse+m.t_echo_free+10e-6
    t_full,C_full=rase_coherence_timeline(E,delta,g,sw,m.dt;delta_window=m.delta_window,
        t_dephase=m.t_dephase,t_free_after=timeline_end-m.t_dephase-m.T_pulse,
        Sz_init=m.Sz_init,n_sub,Nt_free=301)
    echo_time=m.t_dephase+m.T_pulse+m.t_echo_free
    echo_index=argmin(abs.(t_full .- echo_time))
    @assert isapprox(t_full[echo_index],echo_time;atol=1e-14,rtol=0)
    C_echo=C_full[echo_index]
    @assert isapprox(C_echo^2,Jcoh;atol=1e-8,rtol=1e-8)
    PlotResults.plot_coherence_timeline(t_full,C_full;pulse_start=m.t_dephase,
        pulse_end=m.t_dephase+m.T_pulse,echo_time,xtick_step_us=5.0,
        xlimits_us=(0.0,timeline_end*1e6),filename=joinpath(out,"coherence_timeline.png"),
        title=@sprintf("Repaired control: echo C = %.6f, Jcoh = %.6f",C_echo,Jcoh))
    jldsave(joinpath(out,"figure_data.jld2");source,E,checkpoint_metadata=m,n_sub,
        delta=Array(delta),g=Array(g),Sz_final=Array(Sz_final),Sp_echo=Array(Sp_echo),
        t_full,C_full,echo_time,C_echo,Jpop_inside,Jpop_outside,Jcoh,Jsmooth)
    report="Repaired 40 us RASE control\nSource: $(abspath(source))\nIntegration substeps: $n_sub\nJpop_inside=$Jpop_inside\nJpop_outside=$Jpop_outside\nJsmooth=$Jsmooth\nJcoh=$Jcoh\nC_echo=$C_echo (timeline plots C; Jcoh=C_echo^2)\nPulse phase/frequency visibility threshold: 1% of peak amplitude; propagation uses all controls.\n"
    write(joinpath(out,"README.txt"),report)
    println(report);println("Figures saved to ",out)
end
main()
