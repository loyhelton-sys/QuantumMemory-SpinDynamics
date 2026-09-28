using Test
using SpinDynamics

function unpack_initial(Nj, initial_condition; delta_b=nothing)
    u0 = SpinDynamics.build_initial_state_1st_order(
        Nj, initial_condition; delta_b=delta_b)
    return SpinDynamics.unpack_state_1st_order(u0, length(Nj))
end

@testset "basic first-order initial conditions" begin
    Nj = [10.0, 20.0]

    Sp, Sz = unpack_initial(Nj, :ground)
    @test Sp == zeros(ComplexF64, 2)
    @test Sz == ComplexF64[-5.0, -10.0]

    Sp, Sz = unpack_initial(Nj, :inverted)
    @test Sp == zeros(ComplexF64, 2)
    @test Sz == ComplexF64[5.0, 10.0]

    Sp, Sz = unpack_initial(Nj, :mixed)
    @test Sp == zeros(ComplexF64, 2)
    @test Sz == zeros(ComplexF64, 2)
end

@testset "RASE first-order initial condition" begin
    Nj = [10.0, 20.0, 30.0]
    delta = [-2.0, 1.0, 2.0]
    cfg = (kind=:rase, delta_window=2.0, Sz_init=-0.499, phase=π/2)
    Sp, Sz = unpack_initial(Nj, cfg; delta_b=delta)

    amplitude = sqrt(0.25 - 0.499^2)
    @test Sp[1] == 0im
    @test Sp[2] ≈ 20amplitude * cis(π/2)
    @test Sp[3] == 0im
    @test real.(Sz) ≈ [-5.0, -9.98, -15.0]

    for j in eachindex(Nj)
        @test abs2(Sp[j] / Nj[j]) + abs2(Sz[j] / Nj[j]) ≈ 0.25
    end
end

@testset "custom first-order initial condition" begin
    Nj = [10.0, 20.0]
    cfg = (kind=:custom, f=(populations, delta) ->
        (0.1im .* populations, -0.2 .* populations))
    Sp, Sz = unpack_initial(Nj, cfg; delta_b=[0.0, 1.0])
    @test Sp == ComplexF64[1im, 2im]
    @test Sz == ComplexF64[-2.0, -4.0]
end

@testset "initial-condition validation" begin
    @test_throws ErrorException SpinDynamics.build_initial_state_1st_order([], :ground)
    @test_throws ErrorException SpinDynamics.build_initial_state_1st_order([1.0, -1.0], :ground)
    @test_throws ErrorException SpinDynamics.build_initial_state_1st_order([1.0], :rase; delta_b=[0.0])
    @test_throws ErrorException SpinDynamics.build_initial_state_1st_order([1.0],
        (kind=:rase, delta_window=1.0, Sz_init=-0.6); delta_b=[0.0])
    @test_throws ErrorException SpinDynamics.build_initial_state_1st_order([1.0], :custom)
end
