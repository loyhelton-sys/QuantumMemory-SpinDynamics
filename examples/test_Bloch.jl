using CairoMakie

include(joinpath(@__DIR__, "../plots/PlotBloch.jl"))

struct TestData
    Sp_keep
    Sx_keep
    Sy_keep
    Sz_keep
    t_saved
end

N = 100

t = range(0, 1e-6, length=N)

Sx = zeros(N)
Sy = sin.(4π .* t ./ 1e-6)
Sz = cos.(4π .* t ./ 1e-6)

data = TestData(
    zeros(ComplexF64,1,N),
    reshape(Sx,1,:),
    reshape(Sy,1,:),
    reshape(Sz,1,:),
    collect(t)
)

plot_bloch(
    data;
    g_index=1,
    filename=joinpath(@__DIR__, "../Results/test_bloch.mp4"),
    fps=50
)