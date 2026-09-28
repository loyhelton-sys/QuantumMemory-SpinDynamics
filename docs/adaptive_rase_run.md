> Historical experiment record: the one-off scripts below were removed during
> example cleanup. Commands and service instructions are historical, not current
> run instructions. See [retained examples](../examples/README.md).

# Adaptive 5–10–5 μs RASE run

The original unattended run used a transient user service named
`spindynamics-adaptive-rase.service`. That run and both spike cleanup trials have now finished. Inspect the service with:

```sh
systemctl --user status spindynamics-adaptive-rase.service
```

Do not start a second optimizer while that service is active.

Run or resume manually from the repository root:

```sh
julia --project=. examples/adaptive_three_pulse_rase.jl
```

This run has a cumulative cap of 50,000 additional optimizer iterations,
including restoration and coherence refinement. Existing historical iterations
in the original result are not part of this cap. Backtracking objective
calculations and final validation are not optimizer iterations. An interrupted
iteration is counted conservatively on restart. Do not run multiple copies.
`SPINDYNAMICS_MAX_ITER` can lower the cumulative cap, but cannot exceed 50,000.

The original result is read from
`Results/5-10-5/original_three_pulse_rase_optimization_5us_10us_5us/optimized_three_pulse.jld2`.
All new files are in its sibling directory `adaptive_targets/`.

The acceptance criteria are strict:

- `Jpop_inside > -0.01`
- `Jpop_outside > -0.01`
- `Jsmooth > -0.4`

Before feasibility, normalized squared constraint violations determine dynamic
nonnegative objective weights. Once feasible, the optimizer projects the
coherence gradient onto linearized population/smoothness constraints; the
nonnegative multipliers are adaptive weights. Backtracking checks the actual
objectives, requires all three constraints and increasing coherence, and
requests a small inward component near constraint boundaries. No physical
ensemble settings, pulse duration, control grid, or objective definitions are
changed. The phase mask is fixed from the starting control for comparable
scores; consequently its starting smoothness can differ slightly from a
historical result evaluated with an earlier mask.

The run stops at the cap, after eight failed searches, or after coherence gains
less than `1e-7` over 1,000 feasible iterations. These are numerical stopping
criteria, not proof of a global optimum. The final candidate is re-evaluated
with 8, 16, and 32 integration substeps for the current saved configuration.

Files:

- `status.txt`: running/finished status, count, and final validation results.
- `run.log`: output from the detached run launched for this task.
- `history.csv`: each completed iteration and its actual objective components.
- `checkpoint.jld2`: resumable current state, saved every five iterations.
- `best_feasible.jld2`: highest coherence among accepted feasible candidates.
- `validated_result.jld2`: final selected waveform and finer-integration checks.
- `pulse_comparison.png`: final waveform compared with the starting waveform.

`examples/inspect_adaptive_rase.jl` prints metrics and visible frequency-jump
statistics and updates the waveform plot. `examples/validate_adaptive_rase.jl`
independently checks a snapshot with finer integration.

`Jsmooth` is a weighted sum of squared phase second differences, not a hard
bound on local instantaneous frequency, amplitude, or their jumps. Visible
local peaks may remain even when this numerical criterion passes. No additional
peak cutoff has been assumed; inspect the saved plot for that qualitative
requirement. Validation uses the original ensemble grid and finer time
integration; it does not establish robustness to ensemble-grid changes.

## Spike cleanup

The original adaptive run stopped at cumulative iteration 1820. Its fixed phase
mask allowed a new visible tail spike in a formerly low-amplitude region.
The accepted numerical objective by itself did not prevent that spike.

`examples/smooth_adaptive_rase.jl` evaluates phase smoothing on the saved best
control and writes candidates under `adaptive_targets/spike_cleanup/`.
Positive sigma values are Gaussian widths in samples (4 samples = 0.1 us).
Variant -4 repairs four samples at both ends; -1 repairs only four tail samples.
No accepted original waveform is overwritten.

`examples/optimize_spike_cleanup.jl` keeps the selected phase fixed and optimizes
nonnegative amplitudes. This prevents regenerated phase/frequency spikes.
The `edge4` trial reserves cumulative iterations 1821–4000. The `gaussian4`
trial reserves 4001–7000; unused slots are conservatively counted. Together these
remain below the authorized cumulative 50,000 cap. Each directory has its own
checkpoint, history, status, best feasible control, plot, and final validation.

The stronger candidate starts from `smoothed_sigma_4.jld2`:

```sh
SPINDYNAMICS_CLEANUP_VARIANT=gaussian4 \
SPINDYNAMICS_CLEANUP_SOURCE=smoothed_sigma_4.jld2 \
SPINDYNAMICS_ITERATION_FLOOR=4000 \
SPINDYNAMICS_MAX_ITER=7000 \
julia --project=. examples/optimize_spike_cleanup.jl
```

Do not launch another copy while an optimization is running. The fixed phase
is stored as `phase_unit` in checkpoints. Final integration checks and the
comparison plot are generated automatically on completion.

Final cleanup results are in each variant's `cleaned_result.jld2`,
`cleaned_pulse.png`, and `pulse_comparison.png`. The finalizer
`examples/finalize_spike_cleanup.jl` verifies fixed phase, checks 8/16/32
integration substeps, and leaves at least `1e-7` population margin. The stronger
variant needed one tiny amplitude correction for that margin. The conservative
cumulative budget count is 4763; both optimizers have stopped.

Final results (coarse integration; all pass finer checks):

| Variant | Jpop_inside | Jpop_outside | Jcoh | Jsmooth |
| --- | --- | --- | --- | --- |
| edge4 | -0.009999765508 | -0.009999039263 | 0.944232651214 | -0.333149249029 |
| gaussian4 | -0.009999268055 | -0.009999880970 | 0.920678982980 | -0.062834548544 |

Use gaussian4 for the stronger frequency smoothing, or edge4 to prioritize
coherence while removing the extreme endpoint spikes. Both retain real smooth
frequency extrema; neither is a flat-frequency waveform. Frequency statistics
use the existing 1% amplitude visibility threshold. Plot gaps mark regions below
that threshold; the stored physical controls and validation use every sample.
