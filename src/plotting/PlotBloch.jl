using CairoMakie
using Dates
using Printf

# -----------------Structs----------------
struct BlochData
    t::Vector{Float64}
    Sx::Matrix{Vector{Float64}}
    Sy::Matrix{Vector{Float64}}
    Sz::Matrix{Vector{Float64}}
end

struct BlochLayout
    fig::Figure
    ax::Matrix{Axis3}
end

struct BlochObjects
    path::Matrix{Observable{Vector{Point3f}}}
    arrow::Matrix{Observable{Vector{Point3f}}}
    point::Matrix{Observable{Point3f}}
end

# ----------------- Build Data ----------------
function build_data(data; M_delta=1, M_g=1)
    @assert M_delta*M_g <= size(data.Sp_keep,1)

    t = data.t_saved .* 1e6
    Sx = Matrix{Vector{Float64}}(undef,M_delta,M_g)
    Sy = Matrix{Vector{Float64}}(undef,M_delta,M_g)
    Sz = Matrix{Vector{Float64}}(undef,M_delta,M_g)

    for j in 1:M_g, i in 1:M_delta
        ensemble = (i-1)*M_g + j
        norm = data.Nj[ensemble] * 0.5
        Sx[i,j] = real.(data.Sp_keep[ensemble,:]) ./ norm
        Sy[i,j] = imag.(data.Sp_keep[ensemble,:]) ./ norm
        Sz[i,j] = real.(data.Sz_keep[ensemble,:]) ./ norm
    end
    return BlochData(t,Sx,Sy,Sz)
end

# ----------------- Build Layout ----------------
function build_layout(data; M_delta=1, M_g=1, figsize=(1200,1200))
    fig = Figure(size=figsize)
    ax = Matrix{Axis3}(undef, M_delta, M_g)
    for i in 1:M_delta, j in 1:M_g
        ax[i,j] = Axis3(fig[i,j], aspect=(1,1,1), limits=(-1.3,1.3,-1.3,1.3,-1.3,1.3), protrusions=(0,0,0,0))
    end
    return fig, ax
end

# ----------------- Draw Background ----------------
function draw_background!(ax; sphere_color=(:lightskyblue,0.15),
    axis_color=:black, axis_length=1.2, show_labels=true)

    θ = range(0,π,length=50)
    ϕ = range(0,2π,length=100)

    xs = [sin(a)*cos(b) for a in θ, b in ϕ]
    ys = [sin(a)*sin(b) for a in θ, b in ϕ]
    zs = [cos(a) for a in θ, b in ϕ]

    M_delta, M_g = size(ax)

    for i in 1:M_delta, j in 1:M_g
        surface!(ax[i,j],xs,ys,zs,color=sphere_color)

        lines!(ax[i,j],[-axis_length,axis_length],[0,0],[0,0],color=axis_color,linewidth=1.5)
        lines!(ax[i,j],[0,0],[-axis_length,axis_length],[0,0],color=axis_color,linewidth=1.5)
        lines!(ax[i,j],[0,0],[0,0],[-axis_length,axis_length],color=axis_color,linewidth=1.5)

        if show_labels
            text!(ax[i,j],Point3f(axis_length,0,0),text="x",fontsize=20)
            text!(ax[i,j],Point3f(0,axis_length+0.05,0),text="y",fontsize=20)
            text!(ax[i,j],Point3f(0.05,0,axis_length),text="z",fontsize=20)
            text!(ax[i,j],Point3f(-0.4,0,1.4),text="|e⟩",fontsize=25)
            text!(ax[i,j],Point3f(-0.4,0,-1.4),text="|g⟩",fontsize=25)
        end
    end
end

function add_labels!(ax, data; M_delta=1, M_g=1)
    for i in 1:M_delta, j in 1:M_g
        Δ = round(data.delta_b_1d[i] / (2π) / 1e3)
        g = round(data.g_b_1d[j] / (2π) / 1e3)
        text!(ax[i,j], Point3f(-0.2, 0.0, -1.1), text="Δ = $(Δ) kHz\n g = $(g) kHz", fontsize=16, align=(:left, :top))
    end
end

# ----------------- Build Objects ----------------
function build_objects!(ax; M_delta=1, M_g=1)
    colors = [:royalblue,:crimson,:forestgreen,:orange,:purple]

    path = Matrix{Observable{Vector{Point3f}}}(undef,M_delta,M_g)
    arrow = Matrix{Observable{Vector{Point3f}}}(undef,M_delta,M_g)
    point = Matrix{Observable{Point3f}}(undef,M_delta,M_g)

    for i in 1:M_delta, j in 1:M_g

        color = colors[mod1((i-1)*M_g+j,length(colors))]

        path[i,j] = Observable(Point3f[])
        arrow[i,j] = Observable(Point3f[Point3f(0,0,0),Point3f(0,0,-1)])
        point[i,j] = Observable(Point3f(0,0,-1))

        lines!(ax[i,j],path[i,j],linewidth=3,color=color)
        lines!(ax[i,j],arrow[i,j],linewidth=3,color=color)
        scatter!(ax[i,j],point[i,j],markersize=15,color=color)

    end
    return BlochObjects(path,arrow,point)
end

# ----------------- Animate ----------------
function animate!(fig, objs::BlochObjects, pdata::BlochData;
     filename="bloch.mp4", fps=30, frame_step=1)

    Sx = pdata.Sx; Sy = pdata.Sy; Sz = pdata.Sz
    Nt = length(Sx[1,1])

    record(fig, filename, 1:frame_step:Nt; framerate=fps) do n
        for i in axes(Sx,1), j in axes(Sx,2)
            x = Sx[i,j]; y = Sy[i,j]; z = Sz[i,j]
            objs.path[i,j][] = Point3f.(x[1:n], y[1:n], z[1:n])
            objs.arrow[i,j][] = Point3f[Point3f(0,0,0), Point3f(x[n], y[n], z[n])]
            objs.point[i,j][] = Point3f(1.02*x[n], 1.02*y[n], 1.02*z[n])
        end
    end

end

# ----------------- Main Function ----------------
function plot_bloch(data; M_delta=1, M_g=1,
    filename=nothing, fps=30, frame_step=1,figsize=(1200,1200),)
    if filename === nothing
        timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
        filename = joinpath(@__DIR__, "..", "..", "Results", "bloch_$(timestamp).mp4")
    end

    pdata = build_data(data; M_delta=M_delta, M_g=M_g)
    fig, ax = build_layout(data; M_delta=M_delta, M_g=M_g, figsize=figsize)
    add_labels!(ax, data; M_delta=M_delta, M_g=M_g)
    draw_background!(ax, show_labels=(M_delta*M_g<=10))
    objs = build_objects!(ax; M_delta=M_delta, M_g=M_g)
    animate!(fig, objs, pdata; filename=filename, fps=fps, frame_step=frame_step)
    return fig

end