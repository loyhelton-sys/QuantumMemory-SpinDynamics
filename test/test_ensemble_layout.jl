using Test
using SpinDynamics

include(joinpath(@__DIR__, "..", "src", "ensemble", "ensemble.jl"))

@testset "g-fast ensemble layout" begin
    delta_vals = [-2.0, 3.0]
    g_vals = [10.0, 20.0, 30.0]
    p_delta = [0.25, 0.75]
    p_g = [0.2, 0.3, 0.5]
    N = 100.0

    Nj, delta_b, g_b, N_total, Nj_2d = build_2d_bins(
        N, delta_vals, p_delta, g_vals, p_g)

    @test delta_b == [-2.0, -2.0, -2.0, 3.0, 3.0, 3.0]
    @test g_b == [10.0, 20.0, 30.0, 10.0, 20.0, 30.0]
    @test Nj_2d == N .* (p_delta * transpose(p_g))
    @test Nj ≈ [5.0, 7.5, 12.5, 15.0, 22.5, 37.5]
    @test permutedims(reshape(Nj, length(g_vals), length(delta_vals))) == Nj_2d
    @test N_total ≈ N
end

@testset "bin edge validation" begin
    @test_throws ErrorException bin_means_and_probs(nothing, Float64[])
    @test_throws ErrorException bin_means_and_probs(nothing, [0.0])
    @test_throws ErrorException bin_means_and_probs(nothing, [0.0, Inf])
    @test_throws ErrorException bin_means_and_probs(nothing, [0.0, 0.0, 1.0])
    @test_throws ErrorException bin_means_and_probs(nothing, [1.0, 0.0])
end

@testset "unified ensemble builder" begin
    freq_cfg = (kind=:gaussian, FWHM=2.0, span_sigma=3.0)
    g_cfg = (kind=:gaussian, mean=10.0, FWHM=2.0, span_sigma=3.0)

    ensemble = SpinDynamics.build_ensemble(100.0, freq_cfg, g_cfg, 5, 3)

    @test ensemble.M_delta == 5
    @test ensemble.M_g == 3
    @test ensemble.M == 15
    @test length(ensemble.delta_b) == 15
    @test length(ensemble.g_b) == 15
    @test size(ensemble.Nj_2d) == (5, 3)
    @test ensemble.N_total ≈ 100.0
    @test ensemble.Nj ≈ vec(permutedims(ensemble.Nj_2d))
    @test ensemble.delta_b == repeat(ensemble.delta_b_1d; inner=3)
    @test ensemble.g_b == repeat(ensemble.g_b_1d; outer=5)
end


@testset "equal-weight quantile ensemble" begin
    freq_cfg = (kind=:gaussian, FWHM=2π * 0.4e6, span_sigma=3.0, renormalize=true)
    g_cfg = (kind=:gaussian, mean=2π * 100e3, FWHM=2π * 10e3, span_sigma=3.0, renormalize=true)
    ensemble = SpinDynamics.build_equal_weight_ensemble(freq_cfg, g_cfg, 8, 4)
    @test ensemble.sampling == :equal_weight_quantile
    @test ensemble.M == 32
    @test ensemble.N == 32
    @test ensemble.N_total ≈ 32
    @test length(ensemble.delta_b) == 32
    @test length(ensemble.g_b) == 32
    @test all(ensemble.Nj .≈ 1)
    @test all(ensemble.p_delta .≈ 1 / 8)
    @test all(ensemble.p_g .≈ 1 / 4)
    @test issorted(ensemble.delta_b_1d)
    @test issorted(ensemble.g_b_1d)
    @test ensemble.delta_b_1d[1] ≈ -ensemble.delta_b_1d[end]
    @test ensemble.g_b_1d[1] > 0
    @test_throws ErrorException SpinDynamics.build_equal_weight_ensemble(freq_cfg, g_cfg, 0, 4)
    @test_throws ErrorException SpinDynamics.build_equal_weight_ensemble((kind=:constant, delta_value=0.0), g_cfg, 2, 4)
end
