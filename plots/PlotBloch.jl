using CairoMakie
using Printf

function plot_bloch(data; g_index=1, filename="bloch.mp4", fps=30)
    # ----------------- Data ----------------
    @assert 1 ≤ g_index ≤ size(data.Sp_keep, 1)    
    Nj = data.Nj_2d[g_index]

    Sx = data.Sx_keep[g_index, :] ./ (Nj/2)
    Sy = data.Sy_keep[g_index, :] ./ (Nj/2)
    Sz = real(data.Sz_keep[g_index, :]) ./ (Nj/2)
    t = data.t_saved .* 1e6

    # ---------------- Figure----------------
    fig = Figure(size=(900,900))
    ax = Axis3(fig[2,1], aspect = (1,1,1),
        xlabel = "Sx", ylabel = "Sy", zlabel = "Sz",)

    # ---------------- Bloch sphere ----------------
    θ = range(0, π; length=50)
    ϕ = range(0, 2π; length=100)

    xs = [sin(a)*cos(b) for a in θ, b in ϕ]
    ys = [sin(a)*sin(b) for a in θ, b in ϕ]
    zs = [cos(a)        for a in θ, b in ϕ]

    surface!(ax, xs, ys, zs, color=(:lightskyblue,0.15)) 

    # ---------------- Coordinate Axes ----------------
    lines!(ax, [-1.2,1.2],[0,0],[0,0], color=:black,linewidth=1.5)
    lines!(ax, [0,0],[-1.2,1.2],[0,0], color=:black,linewidth=1.5)
    lines!(ax, [0,0],[0,0],[-1.2,1.2], color=:black,linewidth=1.5)

    text!(ax,Point3f(1.2,0,0),text="x",fontsize=20)
    text!(ax,Point3f(0,1.25,0),text="y",fontsize=20)
    text!(ax,Point3f(0.05,0,1.2),text="z",fontsize=20)

    text!(ax,Point3f(-0.3,0,1.2),text="|g⟩",fontsize=25)
    text!(ax,Point3f(-0.3,0,-1.2),text="|e⟩",fontsize=25)

    #----------------Observable----------------
    path = Observable(Point3f[])
    point=Observable(Point3f(Sx[1],Sy[1],Sz[1]))
    arrow = Observable(Point3f[
        Point3f(0,0,0),
        Point3f(Sx[1],Sy[1],Sz[1])
        ])
    
    lines!(ax, path, linewidth=4, color=:royalblue)
    lines!(ax, arrow, linewidth=4, color=:crimson)
    scatter!(ax, point, color=:crimson, markersize=22)

    #---------------- Labels ----------------
    θ0 = acos(clamp(Sz[1], -1, 1))
    ϕ0 = atan(Sy[1], Sx[1])

    info = Observable(@sprintf("t = %6.2f μs\nθ = %6.1f°\nφ = %7.1f°", t[1], rad2deg(θ0), rad2deg(ϕ0)))
    Label(fig[1,1], info, fontsize = 22, halign = :left, tellwidth = false)

    # ---------------- Animation ----------------
    record(fig,filename,eachindex(Sx);framerate=fps) do i
        path[]=Point3f.(Sx[1:i],Sy[1:i],Sz[1:i])
        point[]=Point3f(Sx[i],Sy[i],Sz[i])
        arrow[] = Point3f[
            Point3f(0,0,0),
            Point3f(Sx[i], Sy[i], Sz[i])
        ]

        θi = acos(clamp(Sz[i], -1, 1))
        ϕi = atan(Sy[i], Sx[i])

        info[] = @sprintf("t = %6.2f μs\nθ = %6.1f°\nφ = %7.1f°", t[i], rad2deg(θi), rad2deg(ϕi))
    
    end

return fig

end