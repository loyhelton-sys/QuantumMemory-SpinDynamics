using Test
using JLD2
using SpinDynamics

@testset "custom coupling file validation" begin
    mktempdir() do dir
        valid = joinpath(dir, "valid.jld2")
        jldsave(valid; edges_g=[0.5, 1.5, 2.5], g_b_1d_tmp=[1.0, 2.0], p_g_tmp=[1.0, 3.0])
        edges, values, probs, mean_g, std_g, g2 = SpinDynamics.load_custom_g_distribution(valid)
        @test edges == [0.5, 1.5, 2.5]
        @test values == [1.0, 2.0]
        @test probs == [0.25, 0.75]
        @test mean_g ≈ 1.75
        @test std_g ≈ sqrt(0.1875)
        @test g2 ≈ 3.25

        missing = joinpath(dir, "missing.jld2")
        jldsave(missing; edges_g=[0.5, 1.5], g_b_1d_tmp=[1.0])
        @test_throws ErrorException SpinDynamics.load_custom_g_distribution(missing)

        cases = (
            (edges_g=[0.5, 0.5, 2.5], g_b_1d_tmp=[1.0, 2.0], p_g_tmp=[0.5, 0.5]),
            (edges_g=[0.5, 1.5, 2.5], g_b_1d_tmp=[1.0, -2.0], p_g_tmp=[0.5, 0.5]),
            (edges_g=[0.5, 1.5, 2.5], g_b_1d_tmp=[1.0, 2.0], p_g_tmp=[1.1, -0.1]),
            (edges_g=[0.5, Inf, 2.5], g_b_1d_tmp=[1.0, 2.0], p_g_tmp=[0.5, 0.5]),
        )
        for (i, data) in pairs(cases)
            file = joinpath(dir, "invalid_$(i).jld2")
            jldsave(file; data...)
            @test_throws ErrorException SpinDynamics.load_custom_g_distribution(file)
        end
        @test_throws ErrorException SpinDynamics.load_custom_g_distribution(joinpath(dir, "absent.jld2"))
        @test_throws ErrorException SpinDynamics.load_custom_g_distribution(valid; renormalize=1)
    end
end
