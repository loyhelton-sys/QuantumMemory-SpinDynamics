> Historical experiment record: the one-off scripts below were removed during
> example cleanup. Commands and service instructions are historical, not current
> run instructions. See [retained examples](../examples/README.md).

# 10–20–10 us comparable optimization

Run `julia --project=. examples/run_comparable_40us_rase.jl` to prepare or resume.
Do not run concurrently with the `spindynamics-comparable-40us.service` user service.

Reference: selected 5–10–5 gaussian4 cleaned_result, also used by 7.5–15–7.5.
Population objectives must both exceed -0.0099999; summed phase smoothness
must match -0.1256690970883786 within 1e-7 (twice the 20 us sum).
The target is calculated as reference Jsmooth × N40/N20, matching the
mean per control point at equal dt, not a mask-weight-normalized mean. The preparation checks shared
physical settings, ensemble and dt. Gaussian phase smoothing matches this
score before fixed-phase amplitude optimization. This is a local optimization,
not a proof of the global maximum. The fixed phase mask is saved with the seed.

Outputs are isolated in
`Results/10-20-10/three_pulse_rase_optimization/comparison_5_10_5_mean_smooth/`.
`history.csv`, `status.txt`, and `run.log` track progress. `best_feasible.jld2`
is created only after the constraints are met. The default cap is 50000 new
iterations; SPINDYNAMICS_MAX_ITER can set a smaller cap. Eight failed searches
or less than 1e-7 coherence gain over 1000 feasible iterations stop early.
Final evaluation uses n_sub=32 and asserts population and smoothness targets.
Final plots and comparison.csv are generated only for a feasible selected control.
Jcoh is squared coherence; the timeline shows C_echo=sqrt(Jcoh).

The previous soft-reference script fixed phase while trying to optimize its
smoothness through amplitudes, and used zero coherence weight. Its historical
outputs are preserved but are not used as feasible comparison results.
The older /tmp/spindynamics_continue task was stopped before this run.

The earlier equal-sum run remains paused and preserved in comparison_5_10_5/.
The new directory prevents resuming its fixed, more strongly smoothed phase.
On next run, preparation will build a new seed from the historical control
using the relaxed target. No optimization was restarted when changing the target.
