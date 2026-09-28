using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using CUDA, JLD2, SpinDynamics, Printf
include(joinpath(@__DIR__, "..", "src/plotting/PlotResults.jl"))

function main()
    root = joinpath(@__DIR__, "..", "Results/40us_single_ARP")
    source = joinpath(root, "single_arp_result.jld2")
    d = load(source)
    E = d["E"]; m = d["simulation_metadata"]
    delta, g, sw = CuArray(d["delta"]), CuArray(d["g"]), CuArray(d["spin_weight"])
    duration = d["pulse_parameters"].duration
    comparison_time = m.t_dephase + duration + m.t_echo_free
    timeline_end = comparison_time + 5e-6
    t_full, C_full = rase_coherence_timeline(E,delta,g,sw,m.dt;
        delta_window=m.delta_window,t_dephase=m.t_dephase,
        t_free_after=timeline_end-m.t_dephase-duration,Sz_init=m.Sz_init,
        n_sub=m.n_sub,Nt_free=301)
    idx = argmin(abs.(t_full .- comparison_time))
    C_at_comparison = C_full[idx]
    @assert isapprox(t_full[idx],comparison_time;atol=1e-14,rtol=0)
    @assert isapprox(C_at_comparison^2,d["Jcoh"];atol=1e-8)
    @assert all(isfinite,C_full)
    PlotResults.plot_coherence_timeline(t_full,C_full;
        pulse_start=m.t_dephase,pulse_end=m.t_dephase+duration,
        echo_time=comparison_time,xtick_step_us=5.,xlimits_us=(0.,timeline_end*1e6),
        filename=joinpath(root,"coherence_timeline.png"),
        title=@sprintf("Single 40 us ARP: C(50 us)=%.6f",C_at_comparison))
    jldsave(joinpath(root,"coherence_data.jld2");t_full,C_full,comparison_time,
        C_at_comparison,Jcoh=d["Jcoh"],simulation_metadata=m,source)
    open(joinpath(root,"coherence.csv"),"w") do io
        println(io,"time_s,coherence")
        for k in eachindex(t_full)
            println(io,t_full[k],",",C_full[k])
        end
    end
    report = "Coherence computed with n_sub=$(m.n_sub), target window +/-$(m.delta_window/(2pi*1e6)) MHz. Pulse from 5 to 45 us; plot from 0 to 55 us. C at comparison time 50 us=$C_at_comparison; Jcoh=$(d["Jcoh"]). The 50 us marker is the earlier three-pulse comparison time, not a detected single-ARP echo peak. Files: coherence_timeline.png, coherence_data.jld2, coherence.csv."
    open(joinpath(root,"status.txt"),"a") do io
        println(io,report)
    end
    println(report)
end
main()
