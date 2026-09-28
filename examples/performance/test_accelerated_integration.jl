using CUDA,JLD2,SpinDynamics,Test,LinearAlgebra
include(joinpath(@__DIR__,"..","..","src","grape","RaseGPUAccelerated.jl"))
function main()
    root=joinpath(@__DIR__,"..","..")
    script=read(joinpath(root,"examples","original_three_pulse_rase_optimization.jl"),String)
    Meta.parseall(script)
    d=load(joinpath(root,"Results/40us_single_ARP/manual_from_iq_4336/gpu_performance_test/input_snapshot.jld2"))
    E=d["E"];m=d["checkpoint_metadata"];p=d["parameters"]
    ens=build_equal_weight_ensemble(m.freq_cfg,m.g_cfg,101,101)
    delta,g,sw=CuArray(ens.delta_b),CuArray(ens.g_b),CuArray(ens.Nj./ens.N_total)
    tmp=mktempdir()
    base=joinpath(tmp,"original.jld2");fast=joinpath(tmp,"accelerated.jld2")
    opts=(;p...,η=.4,n_edge=2,coh_tol=1e-3,pop_inside_tol=1e-2,pop_outside_tol=1e-2,
        smooth_tol=.1,checkpoint_every=1,checkpoint_metadata=m,log_every=1)
    a=optimize_rase!(copy(E),delta,g,sw,m.dt;opts...,max_iter=3,iteration_offset=0,checkpoint_file=base)
    GC.gc(true);CUDA.reclaim()
    b=RaseGPUAccelerated.fast_optimize_rase!(copy(E),delta,g,sw,m.dt;opts...,max_iter=3,iteration_offset=0,checkpoint_file=fast)
    @testset "accelerated optimizer integration" begin
        @test isapprox(a[1],b[1];rtol=1e-9,atol=1e-10)
        @test all(isapprox(a[k],b[k];rtol=1e-9,atol=1e-10) for k in 2:6)
        ca,cb=load(base),load(fast)
        @test ca["iter"]==cb["iter"]==3
        @test ca["η"]==cb["η"]
        @test keys(ca)==keys(cb)
        # A second call continues from the accelerated checkpoint in the same format.
        c=RaseGPUAccelerated.fast_optimize_rase!(copy(cb["E"]),delta,g,sw,m.dt;
            opts...,max_iter=1,iteration_offset=cb["iter"],checkpoint_file=fast)
        @test load(fast,"iter")==4
        @test all(isfinite,c[1])
    end
    println("Integration passed; 3-step pulse max difference=",maximum(abs,a[1]-b[1]))
    println("Only temporary checkpoints written: ",tmp)
end
main()
