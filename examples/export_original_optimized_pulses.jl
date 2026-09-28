using JLD2, SpinDynamics
include(joinpath(@__DIR__,"..","src","plotting","PlotPulse.jl"))
function main()
    root=joinpath(@__DIR__,"..","Results","40us_single_ARP")
    original_source=joinpath(root,"single_arp_result.jld2")
    optimized_source=joinpath(root,"smooth_edges_recovery","final_result.jld2")
    a=load(original_source);b=load(optimized_source)
    original=ComplexF64.(a["E"]);optimized=ComplexF64.(b["E"])
    dt=b["checkpoint_metadata"].dt
    @assert a["simulation_metadata"].dt==dt && length(original)==length(optimized)
    out=joinpath(root,"smooth_edges_recovery","pulse_export");mkpath(out)
    for (name,E) in (("original",original),("optimized",optimized))
        phase=SpinDynamics.unwrap_phase(angle.(E));amp=abs.(E)
        phase[amp .<= 0.01maximum(amp)].=NaN
        open(joinpath(out,"$(name)_pulse.csv"),"w") do io
            println(io,"sample,interval_start_s,time_center_s,I,Q,amplitude,phase_rad")
            for k in eachindex(E)
                println(io,join((k,(k-1)*dt,(k-0.5)*dt,real(E[k]),imag(E[k]),amp[k],phase[k]),','))
            end
        end
        PlotPulse.plot_pulse(E,dt;label=name,amplitude_threshold=0.01,
            filename=joinpath(out,"$(name)_pulse.png"),title=name=="original" ? "Original 40 us WURST ARP (unoptimized)" : "Final optimized 40 us pulse (repaired edges)")
    end
    PlotPulse.plot_pulse(optimized,dt;reference=original,label="Optimized",reference_label="Original ARP",
        amplitude_threshold=0.01,filename=joinpath(out,"pulse_comparison.png"),title="Original vs optimized 40 us pulse")
    jldsave(joinpath(out,"pulses.jld2");original,optimized,dt,original_source,optimized_source,
        original_parameters=a["pulse_parameters"],optimized_metadata=b["checkpoint_metadata"],iter=b["iter"])
    write(joinpath(out,"README.txt"),"Original: unoptimized single 40 us WURST ARP from $original_source\nOptimized: $optimized_source, iteration $(b["iter"])\nSamples=$(length(original)), dt=$dt s. Same sample grid; no normalization or resampling.\nI/Q and amplitude are dimensionless control units. Samples apply over [interval_start_s, interval_start_s+dt).\nPhase is in radians and marked NaN at amplitudes <=1% of the pulse peak; I/Q remain unchanged.\nOriginal parameters: $(a["pulse_parameters"])\n")
    println("Exported $out; samples=$(length(original)), dt=$dt, iteration=$(b["iter"])")
end
main()
