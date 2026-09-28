using Test
using SpinDynamics

@testset "first-order state layout" begin
    M = 3
    ranges = SpinDynamics.state_ranges_1st_order(M)
    @test ranges.Sp == 1:3
    @test ranges.Sz == 4:6
    @test SpinDynamics.state_length_1st_order(M) == 6

    state = ComplexF64.(1:6)
    Sp, Sz = SpinDynamics.unpack_state_1st_order(state, M)
    @test Sp == ComplexF64[1, 2, 3]
    @test Sz == ComplexF64[4, 5, 6]

    Sp[1] = 10
    Sz[end] = 20
    @test state[1] == 10
    @test state[6] == 20
    @test Sp isa SubArray
    @test Sz isa SubArray

    @test_throws ErrorException SpinDynamics.state_length_1st_order(0)
    @test_throws ErrorException SpinDynamics.state_ranges_1st_order(2.0)
    @test_throws ErrorException SpinDynamics.unpack_state_1st_order(zeros(5), 3)
end
