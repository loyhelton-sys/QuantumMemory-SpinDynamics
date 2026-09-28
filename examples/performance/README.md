# RASE GPU performance and verification

Run from the repository root:

```sh
julia --project=. examples/performance/benchmark_rase_gpu.jl
```

The benchmark imports `src/grape/RaseGPUAccelerated.jl` directly; the duplicate
prototype has been removed. It does not run the manual optimization script.
The validated implementation is now integrated into
`examples/original_three_pulse_rase_optimization.jl` via
`src/grape/RaseGPUAccelerated.jl`. The reference SpinDynamics optimizer remains available for parity checks.
It uses the same Float64 RK4 arithmetic and discrete adjoint accumulation as the
existing implementation. The forward time/substep loops run inside a GPU kernel;
objective evaluation retains only the final state. Gradient history is retained,
with reverse propagation in 32-control-point chunks and on-device reductions.
Only the finished gradient is transferred back to the CPU.

Measured 2026-09-27 on RTX A4000, snapshot iteration 4700, 1600 controls,
101 x 101 ensemble, n_sub=4, weights (2,1,1,4e-6). Three alternating measurements
per function after warmup, with GPU synchronization. GC/reclaim between samples
is excluded from timing; function allocation costs are included.

| Function | Original median (s) | Prototype median (s) | Speedup |
|---|---:|---:|---:|
| Objective | 0.234784 | 0.033349 | 7.04x |
| Gradient | 0.691245 | 0.103712 | 6.67x |

All 12 small-case assertions passed (N=1,33,64; n_sub=1,4,8; endpoint masks;
fractional phase masks). The real 1600-point pulse passed parity on 51 x 21 and
101 x 101 grids. Full-grid objective differences were zero; gradient relative
error was 3.68e-16. This validates equivalence to the existing approximate
adjoint, not its agreement with an exact mathematical gradient.

These are component measurements, not an end-to-end optimization speedup.
Other jobs were present on the machine. Full-grid original gradient timings
ranged from 0.674 to 1.429 s, while prototype timings ranged from 0.0963 to
0.1049 s. Results and input snapshot are in
Results/40us_single_ARP/manual_from_iq_4336/gpu_performance_test/.

## Integration verification

`julia --project=. examples/performance/test_accelerated_integration.jl` compares
three accepted optimizer steps and checkpoint continuation. All 7 assertions passed;
maximum control difference after three steps was 4.44e-16. Checkpoint schemas and
accepted step sizes matched. Tests write only temporary checkpoints.

## Low-memory backend

`EtaLowMemoryGPU.jl` is still used by `recover_smooth_edges.jl` and
`diagnose_current_eta.jl`. The timings above are historical measurements.
