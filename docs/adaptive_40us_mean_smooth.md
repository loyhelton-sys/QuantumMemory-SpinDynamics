> Historical experiment record: the one-off scripts below were removed during
> example cleanup. Commands and service instructions are historical, not current
> run instructions. See [retained examples](../examples/README.md).

# Joint amplitude and phase continuation for 10–20–10

Run: `julia --project=. examples/adaptive_10_20_10_mean_smooth.jl`.
Service: `spindynamics-adaptive-40us-mean.service`. Do not run concurrently.
Results: `Results/10-20-10/three_pulse_rase_optimization/adaptive_mean_smooth/`.

Starts directly from the historical optimized_three_pulse.jld2 (Jcoh approximately
0.999056), with no Gaussian smoothing and no phase projection. Reuses the
5–10–5 adaptive constraint strategy: complex control updates, backtracking,
increasing constraint penalty when needed, then feasible coherence ascent.
The score mask remains fixed for comparability; the pulse phase is free.
The first/last two control samples remain fixed as in the original strategy.

The raw smoothness lower bound is reference Jsmooth * N40/N20, approximately
-0.1256690971. Both population scores must exceed -0.0099999. Smoother
controls are allowed. Each recovery step may lose at most 5e-4 Jcoh; this
limits abrupt losses but is not a bound on cumulative loss or a guarantee
of convergence. Coherence has nonzero weight throughout recovery. Once
feasible, only feasible coherence improvements are accepted.

The cap is 50000 additional iterations, with checkpointing and early stopping.
The best feasible control is retained and validated at finer substeps at the end.
The previous fixed-phase result remains in comparison_5_10_5_mean_smooth/.

Recovery scaling fix: divide both merit and its gradient by rho. Legacy
checkpoints migrate to the verified effective step 1e-5. Preserve learned
step size when rho increases; do not increase rho merely on search failure.
The pre-fix checkpoint and history are archived in before_scaled_recovery/.

User-requested relaxation: per-recovery-step Jcoh loss limit raised from
1e-4 to 5e-4. Feasible ascent still requires increasing coherence.
