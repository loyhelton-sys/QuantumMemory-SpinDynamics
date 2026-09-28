using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using JLD2
using SpinDynamics

include(joinpath(@__DIR__, "..", "src", "plotting", "PlotPulse.jl"))
using .PlotPulse
include(joinpath(@__DIR__, "..", "src", "plotting", "PlotResults.jl"))
using .PlotResults

# Original three-pulse baseline in control time. For a simulation with 5 μs
# of free evolution first, set t_start=5e-6 and plot over 5–35 μs instead.
durations = (7.5e-6, 15e-6, 7.5e-6)
t_start = 0.0
t_end = t_start + sum(durations)
dt = 0.01e-6
amp = 4.5
bandwidth = 2π * 1.6e6
n = 10.0
edge_frac = 0.01
junction_halfwidth = 0.1e-6
junctions = (t_start + durations[1], t_start + durations[1] + durations[2])
frequency_exclusion_windows = Tuple(
    (t - junction_halfwidth, t + junction_halfwidth) for t in junctions)

E_original = three_wurst_drive(
    t_start=t_start,
    durations=durations,
    amp=amp,
    bandwidth=bandwidth,
    n=n,
    omega0=0.0,
    chirp_sign=+1.0,
    phase0=0.0,
    edge_frac=edge_frac,
)

output_dir = joinpath(@__DIR__, "..", "Results", "original_three_wurst_7p5us_15us_7p5us")
mkpath(output_dir)
pulse_file = joinpath(output_dir, "pulse_7p5us_15us_7p5us.png")

fig = PlotPulse.plot_pulse(E_original;
    t_start=t_start,
    t_end=t_end,
    dt=dt,
    frequency_exclusion_windows=frequency_exclusion_windows,
    filename=pulse_file,
    label="Original 7.5–15–7.5 μs",
    title="Original three-WURST-pulse baseline",
)

display(fig)
println("Saved baseline pulse plot to: $pulse_file")


# -----------------------------------------------------------------------------
run_full_simulation = lowercase(get(ENV, "SPINDYNAMICS_RUN_FULL_SIMULATION", "false")) in ("1", "true", "yes")
if run_full_simulation
# Full RASE simulation: 5 μs free evolution, 30 μs pulse, 15 μs after it.
# -----------------------------------------------------------------------------
pulse_start = 5e-6
pulse_end = pulse_start + sum(durations)
echo_time = pulse_end + 5e-6

SIM_SETTING = (
    simulation_order=:order1,
    N=2000,
    M_delta=101,
    M_g=101,
    Ttotal=50e-6,
    Nt_save=1001,
    initial_condition=(
        kind=:rase,
        delta_window=2π * 0.30e6,
        Sz_init=-0.499,
        phase=0.0,
    ),
    reltol=1e-7,
    abstol=1e-7,
    saved_file_name=nothing,
)

SYSTEM_CONFIG = (
    freq_inhomogeneity=(
        kind=:gaussian,
        FWHM=2π * 0.4e6,
        span_sigma=2.9435,
        renormalize=true,
    ),
    g_inhomogeneity=(
        kind=:gaussian,
        mean=2π * 100e3,
        FWHM=2π * 10e3,
        span_sigma=2.35482,
        renormalize=true,
    ),
)

PULSE_CONFIG = ((
    name="Original three-WURST pulse",
    kind=:three_wurst,
    t_start=pulse_start,
    durations=durations,
    amp=amp,
    bandwidth=bandwidth,
    n=n,
    omega0=0.0,
    chirp_sign=+1.0,
    phase0=0.0,
    edge_frac=edge_frac,
),)

println("Running original three-pulse simulation...")
data = run_simulation(SIM_SETTING, SYSTEM_CONFIG, PULSE_CONFIG)

simulation_file = joinpath(output_dir, "simulation.jld2")
@save simulation_file data

sz_file = joinpath(output_dir, "final_sz.png")
sz_fig = PlotResults.plot_sz_final_map(data;
    delta_window_MHz=0.30,
    filename=sz_file,
    title="Final Sz: original 7.5–15–7.5 μs pulse",
)

coherence_file = joinpath(output_dir, "coherence_timeline.png")
coherence_fig = PlotResults.plot_coherence_timeline(data;
    detuning_max_MHz=0.30,
    pulse_start=pulse_start,
    pulse_end=pulse_end,
    echo_time=echo_time,
    filename=coherence_file,
    title="Coherence: original 7.5–15–7.5 μs pulse",
)

display(sz_fig)
display(coherence_fig)
println("Saved simulation data to: $simulation_file")
println("Saved final Sz map to: $sz_file")
println("Saved coherence timeline to: $coherence_file")
end
