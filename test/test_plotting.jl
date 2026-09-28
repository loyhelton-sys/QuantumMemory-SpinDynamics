using Test
using CairoMakie

include(joinpath(@__DIR__, "..", "src", "plotting", "PlotResults.jl"))
include(joinpath(@__DIR__, "..", "src", "plotting", "PlotVideo.jl"))
using .PlotResults

@testset "plotting data boundaries" begin
    data = (
        M_delta=1, M_g=1, t_saved=[0.0, 1e-6],
        delta_b_1d=[0.0], g_b_1d=[2π * 1e3], delta_b=[0.0],
        Nj=[2.0], Sp_keep=reshape(ComplexF64[1.0, im], 1, 2),
        Sz_keep=reshape(ComplexF64[-1.0, -0.5], 1, 2),
    )
    @test PlotResults.compute_phase_coherence(data) ≈ [1.0, 1.0]
    @test PlotResults.plot_sz_final_map(data) isa Figure
    @test PlotResults.plot_coherence_timeline(data) isa Figure
    @test PlotResults.plot_sz_final_map([0.0], [2π * 1e3], [-0.5]; M_delta=1, M_g=1) isa Figure

    @test_throws ErrorException PlotResults.plot_coherence_timeline([0.0], [1.0, 0.5])
    @test_throws ErrorException PlotResults.plot_coherence_timeline(Float64[], Float64[])
    @test_throws ErrorException PlotResults.compute_phase_coherence(merge(data, (Nj=[0.0],)))
    @test_throws ErrorException PlotResults.compute_phase_coherence(data; detuning_max_MHz=0.0)
    @test_throws ErrorException PlotResults.plot_sz_final_map([0.0], [1.0], [-0.5, 0.5]; M_delta=1, M_g=1)

    @test PlotVideo._validate_video_data(data, :Sz_keep)[1:3] == (1, 1, 2)
    @test_throws ErrorException PlotVideo._validate_video_data(merge(data, (Nj=[0.0],)), :Sz_keep)
    @test_throws ErrorException PlotVideo._validate_video_data(merge(data, (Sz_keep=zeros(2, 2),)), :Sz_keep)
end
