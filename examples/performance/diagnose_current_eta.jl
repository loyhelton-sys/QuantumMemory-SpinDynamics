using CUDA,JLD2,SpinDynamics,LinearAlgebra,Dates
if get(ENV,"SPINDYNAMICS_ETA_LOW_MEMORY","false")=="true"
    include(joinpath(@__DIR__,"EtaLowMemoryGPU.jl"))
    const EtaBackend=EtaLowMemoryGPU
else
    include(joinpath(@__DIR__,"..","..","src/grape/RaseGPUAccelerated.jl"))
    const EtaBackend=RaseGPUAccelerated
end
function main()
    root=normpath(joinpath(@__DIR__,"..",".."))
    script=read(joinpath(root,"examples/original_three_pulse_rase_optimization.jl"),String)
    value(name)=parse(Float64,match(Regex("\\b"*name*"\\s*=\\s*([0-9.eE+\\-]+)"),script).captures[1])
    weights=map(value,["w_pop_inside","w_pop_outside","w_coh","w_smooth"])
    tolerances=map(value,["pop_inside_tol","pop_outside_tol","coh_tol","smooth_tol"])
    nsub=parse(Int,match(r"const N_SUB = env_int\(\"SPINDYNAMICS_N_SUB\",\s*(\d+)\)",script).captures[1])
    r=joinpath(root,"Results/40us_single_ARP/manual_from_iq_4336")
    runs=[(f,load(joinpath(r,f))) for f in ["checkpoint.jld2","optimized_single_arp.jld2"]]
    sort!(runs;by=x->(get(x[2],"iter",get(x[2],"completed_iterations",0)),mtime(joinpath(r,x[1]))))
    file,d=last(runs);m=d["checkpoint_metadata"];E=copy(d["E"])
    mask=load(joinpath(r,"input_pulse.jld2"),"phase_weights")
    ens=build_equal_weight_ensemble(m.freq_cfg,m.g_cfg,m.M_delta,m.M_g)
    delta,g,sw=CuArray(ens.delta_b),CuArray(ens.g_b),CuArray(ens.Nj./ens.N_total)
    p=(;delta_window=m.delta_window,t_dephase=m.t_dephase,t_echo_free=m.t_echo_free,
        Sz_init=m.Sz_init,Sz_target=m.Sz_target,n_sub=nsub,phase_weights=mask,
        w_pop_inside=weights[1],w_pop_outside=weights[2],w_coh=weights[3],w_smooth=weights[4])
    ev(x)=collect(EtaBackend.fast_objective(x,delta,g,sw,m.dt;p...)[1:5])
    v=ev(E);grad=EtaBackend.fast_gradient(E,delta,g,sw,m.dt;p...,n_edge=2)
    out=joinpath(r,"eta_diagnostic_latest.csv")
    println("Source=",file," iteration=",get(d,"iter",get(d,"completed_iterations",0))," saved_eta=",d["η"])
    println("Weights=",weights," tolerances [inside,outside,coherence,smooth]=",tolerances," n_sub=",nsub," baseline=",v);flush(stdout)
    open(out,"w") do io
        println(io,"eta,delta_total,delta_inside,delta_outside,delta_coherence,delta_smooth,rejected_by")
        steps=sort(unique(vcat([value("η")*2.0^k for k=-16:5],[d["η"],d["η"]/1.2,2d["η"]]));rev=true)
        for step in steps
            dv=ev(E+step*grad)-v
            reasons=String[]
            for (i,name) in enumerate(["inside","outside","coherence","smoothness"])
                i==2 && weights[2]==0 && continue
                dv[i+1]<-tolerances[i] && push!(reasons,name)
            end
            dv[1]<=0 && push!(reasons,"total_objective")
            row=join((step,dv...,isempty(reasons) ? "ACCEPT" : join(reasons,"+")),',')
            println(io,row);println(row);flush(stdout);flush(io)
        end
    end
    println("Saved ",out)
end
main()
