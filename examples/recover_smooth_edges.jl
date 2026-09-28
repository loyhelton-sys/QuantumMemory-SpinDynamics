using CUDA, JLD2, SpinDynamics, LinearAlgebra, Dates, Printf
include(joinpath(@__DIR__, "..", "src", "grape", "RaseGPUAccelerated.jl"))
include(joinpath(@__DIR__, "performance", "EtaLowMemoryGPU.jl"))

# Optimize amplitude and continuous phase: no division by |E| in the phase gradient.
function phase_penalty_gradient(phi, mask)
    q=diff(diff(phi)); g=zeros(length(phi)); w=mask[1:end-2].*mask[2:end-1].*mask[3:end]
    for k in eachindex(q)
        c=-2w[k]*q[k]; g[k]+=c; g[k+1]-=2c; g[k+2]+=c
    end
    return -sum(w.*q.^2),g
end
function atomic_save(path; kw...)
    jldsave(path*".tmp";kw...); mv(path*".tmp",path;force=true)
end
# Project a trial phase onto frequency/slew bounds while keeping endpoint phases fixed.
function project_phase!(p,frequency_cap,step_cap)
    n=length(p); movable(k)=2<k<n-1
    fc=frequency_cap*(1-1e-8);sc=step_cap*(1-1e-8)
    for sweep in 1:100
        for k in 1:n-1
            value=p[k+1]-p[k];excess=value-clamp(value,-fc,fc)
            denom=Int(movable(k))+Int(movable(k+1))
            if denom>0 && excess!=0
                c=excess/denom
                movable(k) && (p[k]+=c)
                movable(k+1) && (p[k+1]-=c)
            end
        end
        for k in 1:n-2
            value=p[k]-2p[k+1]+p[k+2];excess=value-clamp(value,-sc,sc)
            denom=Int(movable(k))+4Int(movable(k+1))+Int(movable(k+2))
            if denom>0 && excess!=0
                c=excess/denom
                movable(k) && (p[k]-=c)
                movable(k+1) && (p[k+1]+=2c)
                movable(k+2) && (p[k+2]-=c)
            end
        end
        maximum(abs.(diff(p)))<=frequency_cap && maximum(abs.(diff(diff(p))))<=step_cap && break
    end
    return p
end
function main()
    root=normpath(joinpath(@__DIR__,"..")); out=joinpath(root,"Results/40us_single_ARP/smooth_edges_recovery")
    mkpath(out)
    manual=joinpath(root,"Results/40us_single_ARP/manual_from_iq_4336")
    cp=joinpath(out,"checkpoint.jld2")
    input=load(joinpath(manual,"input_pulse.jld2"))
    if isfile(cp)
        d=load(cp); source=d["source"]; run_iteration=d["run_iteration"]; offset=d["iteration_offset"]
    else
        candidates=[(p,load(p)) for p in (joinpath(root,"Results/40us_single_ARP/smooth_edges_recovery/input_pulse.jld2"),) if isfile(p)]
        saved_iter(x)=Int(get(x,"iter",get(x,"completed_iterations",0)))
        sort!(candidates;by=x->(saved_iter(x[2]),mtime(x[1])))
        source,d=last(candidates); offset=saved_iter(d); run_iteration=0
        d=copy(d); d["best_score"]=Inf; d["best_feasible"]=Inf; d["accepted_count"]=0
        cp_src=joinpath(out,"input_snapshot.jld2"); atomic_save(cp_src;(Symbol(k)=>v for (k,v) in d)...)
    end
    # Continue until sustained negligible improvement or an explicit STOP request.
    E=ComplexF64.(d["E"]); seed=get(d,"seed",get(input,"seed",copy(E)))
    phase_weights=Float64.(d["phase_weights"])
    m=merge(d["checkpoint_metadata"],(;n_sub=4,phase_weights))
    @assert length(E)==length(phase_weights)
    # Aim slightly below the requested 0.01 population error to provide margin.
    targets=[-0.0099,-0.0099,-Inf,-0.1256690970883786]
    ens=build_equal_weight_ensemble(m.freq_cfg,m.g_cfg,m.M_delta,m.M_g)
    delta,g,sw=CuArray(ens.delta_b),CuArray(ens.g_b),CuArray(ens.Nj./ens.N_total)
    physics=(;delta_window=m.delta_window,t_dephase=m.t_dephase,t_echo_free=m.t_echo_free,Sz_init=m.Sz_init,Sz_target=m.Sz_target,n_sub=m.n_sub,phase_weights)
    evaluate(x)=collect(RaseGPUAccelerated.fast_objective(x,delta,g,sw,m.dt;physics...,w_smooth=0.0)[2:5])
    low_memory=Ref(get(d,"low_memory",false))
    function physical_gradient(x,w)
        kw=(;physics...,n_sub=64,n_edge=2,w_pop_inside=w[1],w_pop_outside=w[2],w_coh=w[3],w_smooth=0.0)
        if !low_memory[]
            try
                return RaseGPUAccelerated.fast_gradient(x,delta,g,sw,m.dt;kw...)
            catch err
                if occursin("memory",lowercase(sprint(showerror,err)))
                    GC.gc(true);CUDA.reclaim();low_memory[]=true
                    println("GPU memory limited; switching to exact atom-block gradient.");flush(stdout)
                else
                    rethrow()
                end
            end
        end
        EtaLowMemoryGPU.fast_gradient(x,delta,g,sw,m.dt;kw...)
    end
    v=evaluate(E); A=abs.(E); phi=Float64.(get(d,"phi",d["cleanup_phase"]))
    eta=Float64(get(d,"eta",1.0)); amplitude_cap=get(d,"amplitude_cap",max(4.5,maximum(A)))
    initial_smooth_allowance=get(d,"initial_smooth_allowance",max(1.256690970883786,1.5abs(v[4])))
    phase_scale=get(d,"phase_scale",max(sqrt(sum(abs2,A)/length(A)),0.5))
    best_score=get(d,"best_score",Inf);best_feasible=get(d,"best_feasible",Inf)
    accepted_count=get(d,"accepted_count",0);stalls=0
    # Physical deficits share explicit scales. The derivative gives adaptive weights.
    scales=[0.001,0.001,0.3]
    relaxed_smooth_allowance=max(1.10abs(targets[4]),1.05abs(d["cleanup_metrics"][4]))
    edge_samples=Int(d["edge_samples"])
    frequency_step_cap=Float64(d["frequency_step_cap"])
    frequency_cap=Float64(d["frequency_cap"])
    cleanup_phase=d["cleanup_phase"];cleanup_metrics=d["cleanup_metrics"]
    E_before_cleanup=d["E_before_cleanup"]
    cleanup_sigma_samples=d["cleanup_sigma_samples"];cleanup_edge_us=d["cleanup_edge_us"]
    function configuration(k)
        return abs(targets[4]),0.1
    end
    function loss_and_weights(x,allowance,smooth_weight)
        z=max.((targets[1:2].-x[1:2])./scales[1:2],0.0)
        zs=max(0.0,(-x[4]-allowance)/allowance)
        return sum(abs2,z)-x[3]+smooth_weight*zs^2, vcat(2z./scales[1:2],1.0,2smooth_weight*zs/allowance)
    end
    final_score(x)=loss_and_weights(x,abs(targets[4]),0.1)[1]
    window_size=500
    window_metrics=Float64.(get(d,"window_metrics",copy(v)))
    window_loss=Float64(get(d,"window_loss",final_score(v)))
    quiet_windows=Int(get(d,"quiet_windows",0))
    stop_reason="stopped by STOP file"
    function save(path)
        atomic_save(path;E,seed,phase_weights,checkpoint_metadata=m,source,targets,run_iteration,iteration_offset=offset,iter=offset+run_iteration,
            window_size,window_metrics,window_loss,quiet_windows,eta,Jpop_inside=v[1],Jpop_outside=v[2],Jcoh=v[3],Jsmooth=v[4],amplitude_cap,initial_smooth_allowance,phase_scale,
            best_score,best_feasible,accepted_count,low_memory=low_memory[],parameterization=:amplitude_phase,objective_mode=:maximize_coherence_population_priority,gradient_n_sub=64,scales,relaxed_smooth_allowance,phi,edge_samples,frequency_step_cap,frequency_cap,cleanup_phase,cleanup_metrics,E_before_cleanup,cleanup_sigma_samples,cleanup_edge_us)
    end
    function archive()
        score=final_score(v)
        if score<best_score
            best_score=score; save(joinpath(out,"best_balanced.jld2"))
        end
        if v[4]>=targets[4] && score<best_feasible
            best_feasible=score;save(joinpath(out,"best_smooth_feasible.jld2"))
        end
    end
    @assert E[1]==0 && E[end]==0
    @assert maximum(abs.(diff(phi))) <= frequency_cap
    @assert maximum(abs.(diff(diff(phi)))) <= frequency_step_cap
    # Check the new chain rule and phase regularizer at the actual starting pulse.
    ps,pg=phase_penalty_gradient(phi,phase_weights)
    @assert isapprox(ps,v[4];atol=1e-8,rtol=1e-8)
    probe=sin.(collect(eachindex(phi)));probe[1:2].=0;probe[end-1:end].=0;probe/=norm(probe)
    fd=(phase_penalty_gradient(phi+1e-5probe,phase_weights)[1]-phase_penalty_gradient(phi-1e-5probe,phase_weights)[1])/2e-5
    @assert isapprox(dot(pg,probe),fd;rtol=1e-5,atol=1e-8)
    wg=[1.0,1.0,1.0];cg=physical_gradient(E,wg)
    pred=dot(imag.(conj.(E).*cg),probe)
    fdphys=(sum(evaluate(A.*cis.(phi+1e-4probe))[1:3])-sum(evaluate(A.*cis.(phi-1e-4probe))[1:3]))/2e-4
    println("Gradient diagnostic: predicted=$pred finite_difference=$fdphys"); flush(stdout)
    @assert isapprox(pred,fdphys;rtol=0.02,atol=1e-6)
    println("Gradient checks passed: phase physical predicted=$pred finite_difference=$fdphys")
    println("Source=$source source_iteration=$offset start=$run_iteration stop=plateau metrics=$v amplitude_cap=$amplitude_cap");flush(stdout)
    save(cp);archive()
    history=joinpath(out,"history.csv")
    isfile(history)||write(history,"run_iteration,total_iteration,pop_inside,pop_outside,coherence,smooth,eta,accepted,trials,smooth_allowance,w_inside,w_outside,w_coh,w_smooth,elapsed_seconds\n")
    started=time()
    while !isfile(joinpath(out,"STOP"))
        run_iteration+=1
        allowance,smooth_weight=configuration(run_iteration)
        loss,w=loss_and_weights(v,allowance,smooth_weight)
        cg=physical_gradient(E,w)
        _,pg=phase_penalty_gradient(phi,phase_weights)
        ga=real.(cis.(-phi).*cg)
        gp=(imag.(conj.(E).*cg).+w[4].*pg)./phase_scale^2
        ga[1:edge_samples].=0;ga[end-edge_samples+1:end].=0;gp[1:2].=0;gp[end-1:end].=0
        # Project the amplitude direction onto its box; retain the existing peak bound.
        ga[(A.<=1e-10).&(ga.<0)].=0;ga[(A.>=amplitude_cap).&(ga.>0)].=0
        denom=max(maximum(abs.(ga))/0.15,maximum(abs.(gp))/0.08,1e-14)
        ga./=denom;gp./=denom
        accepted=false;trials=0
        for trial in 1:22
            trials=trial
            an=clamp.(A.+eta.*ga,1e-10,amplitude_cap);pn=phi.+eta.*gp
            an[1:edge_samples]=A[1:edge_samples];an[end-edge_samples+1:end]=A[end-edge_samples+1:end]
            project_phase!(pn,frequency_cap,frequency_step_cap)
            en=an.*cis.(pn);en[1:2]=E[1:2];en[end-1:end]=E[end-1:end]
            # Keep repaired frequency spikes from returning during recovery.
            if maximum(abs.(diff(pn))) > frequency_cap || maximum(abs.(diff(diff(pn)))) > frequency_step_cap
                eta*=0.5
                continue
            end
            vn=evaluate(en);ln,_=loss_and_weights(vn,allowance,smooth_weight)
            if all(isfinite,vn) && vn[4]>=-relaxed_smooth_allowance && ln<loss-1e-12*max(loss,1.0)
                E=en;A=an;phi=pn;v=vn
                accepted=true;eta=min(5.0,eta*1.25);break
            end
            eta*=0.5
        end
        accepted_count+=accepted
        stalls=accepted ? 0 : stalls+1
        if !accepted
            eta=0.25
        end
        archive()
        open(history,"a") do io
            println(io,join((run_iteration,offset+run_iteration,v...,eta,accepted,trials,allowance,w...,time()-started),','))
        end
        if run_iteration%10==0 || run_iteration==1 || !accepted
            save(cp)
            report="running; $(now()); new iterations=$run_iteration (plateau stop); accepted=$accepted_count; metrics=$v; temporary smooth floor=$(-allowance); weights=$w; eta=$eta; accepted_last=$accepted; low_memory=$(low_memory[])\n"
            write(joinpath(out,"status.txt"),report);print(report);flush(stdout)
        end
        if run_iteration % window_size == 0
            current_loss=final_score(v)
            loss_gain=window_loss-current_loss
            metric_change=abs.(v.-window_metrics)
            quiet=loss_gain < 1e-5 && metric_change[3] < 1e-5 &&
                  maximum(metric_change[1:2]) < 1e-6 && metric_change[4] < 1e-5
            quiet_windows=quiet ? quiet_windows+1 : 0
            open(joinpath(out,"plateau_checks.csv"),"a") do io
                println(io,join((run_iteration,loss_gain,metric_change...,quiet_windows),','))
            end
            println("Plateau check: iteration=$run_iteration loss_gain=$loss_gain changes=$metric_change quiet_windows=$quiet_windows/3");flush(stdout)
            window_loss=current_loss;window_metrics=copy(v)
            save(cp)
            if quiet_windows >= 3
                stop_reason="negligible improvement for three consecutive 500-iteration windows"
                break
            end
        end
        if stalls >= 50
            stop_reason="line search stalled for 50 consecutive iterations"
            break
        end
    end
    save(cp);save(joinpath(out,"final_result.jld2"))
    reason=stop_reason
    write(joinpath(out,"status.txt"),"$reason; new iterations=$run_iteration (plateau stop); accepted=$accepted_count; metrics=$v; validating saved results\n")
    for file in ("final_result.jld2","best_balanced.jld2","best_smooth_feasible.jld2")
        path=joinpath(out,file);isfile(path)||continue
        saved=load(path);x=saved["E"]
        refined=collect(RaseGPUAccelerated.fast_objective(x,delta,g,sw,m.dt;merge(physics,(;n_sub=8))...,w_smooth=0.0)[2:5])
        saved["validation_nsub8"]=refined
        atomic_save(path;(Symbol(k)=>v for (k,v) in saved)...)
        open(joinpath(out,"validation.txt"),"a") do io
            println(io,"$file n_sub=8 metrics=$refined targets_met=$(all(refined.>=targets))")
        end
        open(joinpath(out,replace(file,".jld2"=>"_pulse.csv")),"w") do io
            println(io,"time_s,I,Q,amplitude,phase_rad")
            ph=SpinDynamics.unwrap_phase(angle.(x))
            for k in eachindex(x)
                println(io,join(((k-1)*m.dt,real(x[k]),imag(x[k]),abs(x[k]),ph[k]),','))
            end
        end
    end
    write(joinpath(out,"status.txt"),"$reason; new iterations=$run_iteration (plateau stop); accepted=$accepted_count; final metrics (n_sub=4)=$v; n_sub=8 checks in validation.txt\n")
    println("Stopped automatically: $reason at $run_iteration (plateau stop)");flush(stdout)
end
try
    main()
catch err
    out=normpath(joinpath(@__DIR__,"..","Results/40us_single_ARP/smooth_edges_recovery"));mkpath(out)
    write(joinpath(out,"status.txt"),"ERROR $(now()): $(sprint(showerror,err)); see run.log; checkpoint preserved\n")
    rethrow()
end
