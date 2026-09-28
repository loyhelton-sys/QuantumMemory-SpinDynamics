# Examples

Run from the repository root with `julia --project=. examples/<script>.jl`.
Plotting examples require CairoMakie in the active environment; GPU examples
require a working CUDA device. Several workflows need existing results.

## Optimization and simulation

| Entry | Purpose / required input |
| --- | --- |
| `grape_manual_optimization.jl` | Population-target GRAPE. |
| `rase_optimization.jl` | RASE optimization and pulse editing; requires `Results/E_rase_20μs.jld2`. |
| `three_pulse_rase_optimization.jl` | Three-pulse optimization and checkpoint continuation. |
| `original_three_pulse_rase_optimization.jl` | Current accelerated **single 40 μs ARP** continuation; historical filename retained. Requires `Results/40us_single_ARP/manual_from_iq_4336/input_pulse.jld2`. |
| `echo_simulation.jl` | Echo simulation and coherence plots. |
| `single_40us_arp.jl` | Single-ARP baseline using metadata from the saved 10–20–10 experiment. |
| `rase_spinmap.jl` | Spin evolution visualization from saved optimized controls. |
| `Bloch/`, `Pulse-Map/` | Bloch trajectories and pulse phase/population maps. |

## Plots and final recovery workflow

- `plot_original_three_pulse.jl`: analytic three-pulse baseline.
- `plot_single_40us_arp_coherence.jl`: coherence of the saved single-ARP baseline.
- `prepare_smooth_edges_recovery.jl`, `recover_smooth_edges.jl`: smooth-edge
  recovery from the saved `Results/40us_single_ARP/coherence_until_plateau/final_result.jld2`.
- `plot_smooth_edges_recovery.jl`, `export_original_optimized_pulses.jl`: plot and
  export the recovered pulse.
- `performance/`: backend parity, integration, benchmarking and learning-rate diagnostics.

## Cleanup

Completed experiment launchers and the duplicated GPU prototype were removed.
The benchmark now imports `src/grape/RaseGPUAccelerated.jl` directly.
`EtaLowMemoryGPU.jl` remains required by the recovery workflow. Core manual and
RASE implementations remain required by public APIs, tests and backend validation.

Existing result files are preserved. Historical reports in `docs/` retain their
numerical results; their old commands are no longer current entry points.
See [cleanup record](../docs/example_cleanup_proposal.md) for the removal list
and verified backup location.

The performance integration script requires the historical
`Results/40us_single_ARP/manual_from_iq_4336/gpu_performance_test/input_snapshot.jld2`.
Its absence prevents that snapshot-based test from running.
