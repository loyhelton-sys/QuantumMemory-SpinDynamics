using Test
using SpinDynamics

valid_sim = (
    simulation_order=:order1,
    N=10.0,
    M_delta=1,
    M_g=1,
    Ttotal=1e-6,
    Nt_save=3,
    reltol=1e-7,
    abstol=1e-9,
    initial_condition=:ground,
    saved_file_name=nothing,
)
valid_system = (
    freq_inhomogeneity=(kind=:constant, delta_value=0.0),
    g_inhomogeneity=(kind=:constant, g_value=1.0),
)
valid_config = SpinDynamics.build_full_config(valid_sim, valid_system)

@testset "complete simulation config" begin
    @test SpinDynamics.validate_config(valid_config) === nothing
    @test SpinDynamics.get_simulation_order(valid_sim) == :order1
    @test SpinDynamics.get_initial_condition(valid_config) == :ground

    for bad in (
        merge(valid_config, (N=-1.0,)),
        merge(valid_config, (N=Inf,)),
        merge(valid_config, (M_delta=1.0,)),
        merge(valid_config, (M_g=0,)),
        merge(valid_config, (Ttotal=NaN,)),
        merge(valid_config, (Nt_save=1,)),
        merge(valid_config, (reltol=0.0,)),
        merge(valid_config, (abstol=Inf,)),
        merge(valid_config, (saved_file_name=42,)),
    )
        @test_throws ErrorException SpinDynamics.validate_config(bad)
    end

    incomplete = Base.structdiff(valid_config, NamedTuple{(:N,)}((valid_config.N,)))
    @test_throws ErrorException SpinDynamics.validate_config(incomplete)
    @test_throws ErrorException SpinDynamics.validate_simulation_order((simulation_order=:order2,))
end

@testset "initial-condition config" begin
    for initial in (
        :ground,
        :inverted,
        :mixed,
        (kind=:rase, delta_window=1.0, Sz_init=-0.499),
        (kind=:rase, delta_window=1.0, Sz_init=-0.499, phase=0.2),
        (kind=:custom, f=(Nj, delta_b) -> (zero.(Nj) .+ 0im, -0.5 .* Nj)),
    )
        @test SpinDynamics.validate_initial_condition_config(initial) === nothing
    end

    for bad in (
        :rase,
        :unknown,
        (kind=:rase, Sz_init=-0.499),
        (kind=:rase, delta_window=0.0, Sz_init=-0.499),
        (kind=:rase, delta_window=1.0, Sz_init=-0.6),
        (kind=:rase, delta_window=1.0, Sz_init=-0.499, phase=Inf),
        (kind=:custom, f=1),
        (kind=:unknown,),
    )
        @test_throws ErrorException SpinDynamics.validate_initial_condition_config(bad)
    end
end

@testset "config composition and pulse validation" begin
    @test_throws ErrorException SpinDynamics.build_full_config(
        (simulation_order=:order1, N=1.0),
        (N=2.0,),
    )

    @test SpinDynamics.validate_pulse_config((
        (kind=:constant, amp=0.0),
        (kind=:gaussian, t0=0.0, sigma=1.0, amp=1.0),
    )) === nothing
    @test_throws ErrorException SpinDynamics.validate_pulse_config(((kind=:unknown,),))
    @test_throws ErrorException SpinDynamics.validate_pulse_config(((kind=:constant,),))
end
