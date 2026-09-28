using CUDA, JLD2, SpinDynamics, LinearAlgebra
include(joinpath(@__DIR__,"..","src","grape","RaseGPUAccelerated.jl"))
# Gaussian convolution with linear continuation, so a linear chirp is preserved.
function smooth_linear(x,sigma)
    r=ceil(Int,4sigma); w=exp.(-collect(-r:r).^2/(2sigma^2));w./=sum(w)
    n=length(x)
    extended(k)=k<1 ? x[1]+(k-1)*(x[2]-x[1]) : k>n ? x[end]+(k-n)*(x[end]-x[end-1]) : x[k]
    [sum(w[j+r+1]*extended(k+j) for j in -r:r) for k in 1:n]
end
function main()
    root=joinpath(@__DIR__,"..","Results","40us_single_ARP")
    source=joinpath(root,"coherence_until_plateau","final_result.jld2")
    out=joinpath(root,"smooth_edges_recovery");mkpath(out)
    isfile(joinpath(out,"input_pulse.jld2")) && error("Prepared input already exists; do not overwrite")
    d=load(source); E0=ComplexF64.(d["E"]);m=d["checkpoint_metadata"]
    phase0=SpinDynamics.unwrap_phase(angle.(E0)); A0=abs.(E0);pw=Float64.(d["phase_weights"])
    # Near-zero frozen endpoint phases are undefined; continue the neighboring chirp.
    for k in 2:-1:1;phase0[k]=phase0[3]+(k-3)*(phase0[4]-phase0[3]);end
    for k in length(E0)-1:length(E0);phase0[k]=phase0[end-2]+(k-length(E0)+2)*(phase0[end-2]-phase0[end-3]);end
    pw[1:2].=0;pw[end-1:end].=0
    ens=build_equal_weight_ensemble(m.freq_cfg,m.g_cfg,m.M_delta,m.M_g)
    delta,g,sw=CuArray(ens.delta_b),CuArray(ens.g_b),CuArray(ens.Nj./ens.N_total)
    physics=(;delta_window=m.delta_window,t_dephase=m.t_dephase,t_echo_free=m.t_echo_free,Sz_init=m.Sz_init,Sz_target=m.Sz_target,n_sub=8,phase_weights=pw)
    evaluate(E)=collect(RaseGPUAccelerated.fast_objective(E,delta,g,sw,m.dt;physics...,w_smooth=0.0)[2:5])
    visible_peak(E,p)=begin
        a=abs.(E);valid=(a[1:end-1].>0.01maximum(a)).&(a[2:end].>0.01maximum(a))
        maximum(abs.(diff(p)[valid]))/(2pi*m.dt*1e6)
    end
    baseline=evaluate(E0);peak0=visible_peak(E0,phase0)
    println("Original metrics=$baseline visible_frequency_peak_MHz=$peak0");flush(stdout)
    candidates=[]
    open(joinpath(out,"cleanup_candidates.csv"),"w") do io
        println(io,"sigma_samples,edge_us,pop_inside_mse,pop_outside_mse,coherence,smooth_penalty,visible_peak_MHz,max_frequency_step_MHz,score")
        for sigma in (0.5,1.0,1.5,2.0,3.0), edge_us in (0.1,0.15,0.2,0.3)
            smoothed=smooth_linear(phase0,sigma)
            # Repair only neighborhoods of large instantaneous-frequency excursions.
            freq=abs.(diff(phase0))./(2pi*m.dt*1e6)
            gate=zeros(length(phase0)); radius=12
            for k in findall(freq.>1.2), j in max(1,k-radius):min(length(gate),k+radius)
                gate[j]=max(gate[j],cospi((j-k)/(2radius))^2)
            end
            phi=phase0.+gate.*(smoothed.-phase0);A=copy(A0)
            ne=round(Int,edge_us*1e-6/m.dt)
            # Cubic Hermite edges match interior amplitude and slope, starting at zero with zero slope.
            for reverse_edge in (false,true)
                idx(k)=reverse_edge ? length(A)-k+1 : k
                endpoint=A0[idx(ne+1)]
                slope=ne*(A0[idx(ne+2)]-A0[idx(ne)])/2
                for k in 1:ne+1
                    u=(k-1)/ne
                    A[idx(k)]=clamp((-2u^3+3u^2)*endpoint+(u^3-u^2)*slope,0.0,maximum(A0))
                end
            end
            E=A.*cis.(phi);E[1]=0;E[end]=0
            v=evaluate(E);peak=visible_peak(E,phi)
            step=maximum(abs.(diff(diff(phi))))/(2pi*m.dt*1e6)
            score=sum(abs2,max.((-0.0099 .-v[1:2])./0.001,0.0))-v[3]
            println(io,join((sigma,edge_us,-v[1],-v[2],v[3],-v[4],peak,step,score),','))
            push!(candidates,(;sigma,edge_us,ne,E,phi,v,peak,step,score))
        end
    end
    # Require a real, visible reduction in spikes before ranking physical performance.
    eligible=filter(c->c.peak<=0.75peak0,candidates)
    isempty(eligible) && error("No candidate reduced visible frequency spikes by at least 25%")
    c=eligible[argmin([x.score for x in eligible])]
    saved=copy(d);saved["E"]=c.E;saved["phase_weights"]=pw;saved["cleanup_phase"]=c.phi
    saved["E_before_cleanup"]=E0;saved["edge_samples"]=c.ne+1
    saved["cleanup_sigma_samples"]=c.sigma;saved["cleanup_edge_us"]=c.edge_us
    saved["frequency_step_cap"]=1.05maximum(abs.(diff(diff(c.phi))))
    saved["frequency_cap"]=1.05maximum(abs.(diff(c.phi)))
    saved["cleanup_metrics"]=c.v;saved["original_metrics"]=baseline
    saved["cleanup_source"]=source
    for (key,val) in zip(("Jpop_inside","Jpop_outside","Jcoh","Jsmooth"),c.v);saved[key]=val;end
    jldsave(joinpath(out,"input_pulse.jld2");(Symbol(k)=>v for (k,v) in saved)...)
    println("Selected sigma=$(c.sigma) samples, edge=$(c.edge_us) us; metrics=$(c.v); visible_peak=$(c.peak) MHz (before $peak0)");flush(stdout)
end
main()
