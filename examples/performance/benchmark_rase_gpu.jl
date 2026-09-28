using CUDA, JLD2, SpinDynamics, LinearAlgebra, Random, Statistics, Test
include(joinpath(@__DIR__, "..", "..", "src", "grape", "RaseGPUAccelerated.jl"))
using .RaseGPUAccelerated
cleanup()=(GC.gc(true);CUDA.reclaim())
function compare(E,delta,g,sw,dt,p;edge=2)
    a=rase_objective(E,delta,g,sw,dt;p...);CUDA.synchronize()
    ac=(collect(a[1:5]),Array(a[6]),Array(a[7]));a=nothing;cleanup()
    b=RaseGPUAccelerated.fast_objective(E,delta,g,sw,dt;p...);CUDA.synchronize()
    bc=(collect(b[1:5]),Array(b[6]),Array(b[7]));b=nothing;cleanup()
    @test isapprox(ac[1],bc[1];rtol=1e-9,atol=1e-10)
    @test isapprox(ac[2],bc[2];rtol=1e-9,atol=1e-10)
    @test isapprox(ac[3],bc[3];rtol=1e-9,atol=1e-10)
    ga=rase_gradient(E,delta,g,sw,dt;p...,n_edge=edge);cleanup()
    gb=RaseGPUAccelerated.fast_gradient(E,delta,g,sw,dt;p...,n_edge=edge);cleanup()
    err=norm(ga-gb)/max(norm(ga),1e-14)
    @test isapprox(ga,gb;rtol=1e-8,atol=1e-10)
    println("PARITY N=",length(E)," M=",length(delta)," n_sub=",p.n_sub," gradient relative error=",err," objective max error=",maximum(abs,ac[1]-bc[1]));flush(stdout)
end
function timing(f)
    cleanup()
    elapsed=@elapsed CUDA.@sync f()
    cleanup()
    elapsed
end
function main()
    Random.seed!(23)
    CUDA.functional()||error("CUDA unavailable")
    @testset "GPU accelerated backend parity" begin
        for (N,nsub,edge) in [(1,1,0),(33,4,2),(64,8,2)]
            M=35;delta=CuArray(collect(range(-2.,2.;length=M)));g=CuArray(fill(.8,M));sw=CuArray(fill(1/M,M))
            E=randn(ComplexF64,N)*.2
            p=(;delta_window=1.,t_dephase=.02,t_echo_free=.03,Sz_init=-.499,Sz_target=.499,
                n_sub=nsub,w_pop_inside=2.,w_pop_outside=1.,w_coh=1.,w_smooth=4e-6,phase_weights=rand(N))
            compare(E,delta,g,sw,.01,p;edge)
        end
    end
    root=joinpath(@__DIR__,"..","..","Results","40us_single_ARP","manual_from_iq_4336")
    d=load(joinpath(root,"checkpoint.jld2"));m=d["checkpoint_metadata"];E=ComplexF64.(d["E"])
    pm=load(joinpath(root,"input_pulse.jld2"),"phase_weights")
    p=(;delta_window=m.delta_window,t_dephase=m.t_dephase,t_echo_free=m.t_echo_free,
        Sz_init=m.Sz_init,Sz_target=m.Sz_target,n_sub=4,w_pop_inside=2.,w_pop_outside=1.,w_coh=1.,w_smooth=4e-6,phase_weights=pm)
    out=joinpath(root,"gpu_performance_test");mkpath(out)
    jldsave(joinpath(out,"input_snapshot.jld2");E,checkpoint_metadata=m,parameters=p,source_iter=get(d,"iter",0))
    println("Snapshot iteration=",get(d,"iter",0)," N=",length(E));flush(stdout)
    for (md,mg) in [(51,21),(101,101)]
        ens=build_equal_weight_ensemble(m.freq_cfg,m.g_cfg,md,mg)
        delta,g,sw=CuArray(ens.delta_b),CuArray(ens.g_b),CuArray(ens.Nj./ens.N_total)
        println("GRID ",md,"x",mg);flush(stdout)
        compare(E,delta,g,sw,m.dt,p)
        funcs=[()->rase_objective(E,delta,g,sw,m.dt;p...),
               ()->RaseGPUAccelerated.fast_objective(E,delta,g,sw,m.dt;p...),
               ()->rase_gradient(E,delta,g,sw,m.dt;p...,n_edge=2),
               ()->RaseGPUAccelerated.fast_gradient(E,delta,g,sw,m.dt;p...,n_edge=2)]
        # Parity calls above warm up every path before measurements.
        times=zeros(3,4)
        for repeat in 1:3, i in (isodd(repeat) ? (1,2,3,4) : (4,3,2,1))
            times[repeat,i]=timing(funcs[i])
            println("TIME grid=",md,"x",mg," repeat=",repeat," method=",i," seconds=",times[repeat,i]);flush(stdout)
        end
        med=vec(median(times;dims=1))
        println("SUMMARY grid=",md,"x",mg," medians=",med," objective_speedup=",med[1]/med[2]," gradient_speedup=",med[3]/med[4]);flush(stdout)
        jldsave(joinpath(out,"benchmark_$(md)x$(mg).jld2");times,medians=med,parameters=p,M_delta=md,M_g=mg)
        open(joinpath(out,"summary.txt"),"a") do io
            println(io,"grid=$md x $mg; seconds [old objective,new objective,old gradient,new gradient]=$med; objective speedup=$(med[1]/med[2]); gradient speedup=$(med[3]/med[4])")
        end
    end
end
main()
