using Test
using SpinDynamics
using CUDA

@testset "first-order solver data contract" begin
    sim = (simulation_order=:order1, N=10.0, M_delta=1, M_g=1,
        Ttotal=1e-7, Nt_save=3, initial_condition=:ground,
        reltol=1e-7, abstol=1e-9, saved_file_name=nothing)
    system = (
        freq_inhomogeneity=(kind=:constant, delta_value=0.0),
        g_inhomogeneity=(kind=:constant, g_value=1.0),
    )
    config = SpinDynamics.build_full_config(sim, system)
    derived = SpinDynamics.prepare_derived(config)
    @test !hasproperty(derived, :timespan)
    @test !hasproperty(derived, :t_save)
    @test derived.Nj == [10.0]

    if CUDA.functional()
        pulses = ((kind=:constant, amp=0.0),)
        data = run_simulation(sim, system, pulses; clean_gpu=false, verbose=false)
        @test data.t_saved == [0.0, 0.5e-7, 1e-7]
        @test data.Nj == [10.0]
        @test size(data.Sp_keep) == (1, 3)
        @test size(data.Sz_keep) == (1, 3)
        @test all(iszero, data.Sp_keep)
        @test real.(data.Sz_keep) ≈ fill(-5.0, 1, 3)
        @test !hasproperty(data, :Sx_keep)
        @test !hasproperty(data, :Sy_keep)
    else
        @test_skip CUDA.functional()
    end
end
