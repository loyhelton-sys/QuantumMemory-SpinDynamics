using Test
using SpinDynamics

function evaluate_rhs(Sp, Sz; delta, g, E)
    M = length(Sp)
    u = vcat(ComplexF64.(Sp), ComplexF64.(Sz))
    du = similar(u)
    p = (delta_b=Float64.(delta), g_b=Float64.(g), M=M, E_of_t=t -> E)
    SpinDynamics.rhs_1st_order!(du, u, p, 0.0)
    return SpinDynamics.unpack_state_1st_order(du, M)
end

@testset "first-order RHS free evolution" begin
    Sp = ComplexF64[0.2 + 0.1im, -0.3 + 0.05im]
    Sz = [-0.4, 0.25]
    delta = [2.0, -3.0]
    dSp, dSz = evaluate_rhs(Sp, Sz; delta=delta, g=[1.0, 4.0], E=0im)

    @test dSp ≈ 1im .* delta .* Sp
    @test dSz ≈ zeros(ComplexF64, 2)
end

@testset "first-order RHS resonant drive" begin
    dSp, dSz = evaluate_rhs([0im], [-0.5]; delta=[0.0], g=[2.0], E=1.0 + 0im)
    @test dSp[1] ≈ 2im
    @test dSz[1] ≈ 0im
end

@testset "first-order RHS Bloch-length conservation" begin
    Sp = ComplexF64[0.2 + 0.1im, -0.1 + 0.25im]
    Sz = [-0.35, 0.4]
    dSp, dSz = evaluate_rhs(Sp, Sz; delta=[2.0, -3.0], g=[1.5, 0.7], E=0.3 + 0.4im)

    for j in eachindex(Sp)
        dlength2 = 2real(conj(Sp[j]) * dSp[j]) + 2Sz[j] * real(dSz[j])
        @test dlength2 ≈ 0.0 atol=1e-12
        @test imag(dSz[j]) ≈ 0.0 atol=1e-12
    end
end
