# SpinDynamics model conventions

This document records the physical and data conventions used by the
first-order semiclassical model and the GRAPE implementation. It is the
reference for future refactoring; it does not by itself change the model.

## Scope

The current optimization model describes an inhomogeneously broadened
ensemble of independent two-level spins driven by a prescribed classical
field. It does not include cavity dynamics, quantum field fluctuations, or
an ASE--RASE entanglement criterion.

The optimized coherence is a first-order collective rephasing proxy, not an
entanglement fidelity.

## Units

- Time is measured in seconds.
- Detuning `delta` is an angular frequency in rad/s.
- Coupling `g` is an angular frequency in rad/s per unit control amplitude.
- Frequencies reported in Hz or MHz are obtained by dividing angular
  frequencies by `2pi`.
- The complex control is `E(t) = Ex(t) + i Ey(t)`.
- The complex Rabi frequency for ensemble bin `j` is `Omega_j(t) = g_j E(t)`.

## Spin variables and normalization

For each ensemble bin,

```text
Splus = Sx + i Sy
```

The normalized single-spin Bloch variables satisfy

```text
Sx^2 + Sy^2 + Sz^2 = 1/4
```

for a pure state. Therefore the ground and inverted states have respectively
`Sz = -0.5` and `Sz = +0.5`.

In the full first-order solver, each bin stores collective variables scaled by
its population `Nj`. Plotted normalized population is therefore `Sz/Nj`, with
no additional factor of two.

## Initial conditions

The first-order solver supports `:ground`, `:inverted`, `:mixed`, `:rase`,
and `:custom`. Ground, inverted, and mixed conditions may be supplied as
Symbols. A RASE condition must explicitly provide its target window and
normalized population, for example:

```julia
initial_condition = (
    kind = :rase,
    delta_window = 2π * 0.30e6,
    Sz_init = -0.499,
    phase = 0.0,
)
```

The window is strict: `abs(delta) < delta_window`. A custom condition supplies
`f = (Nj, delta_b) -> (Sp, Sz)` and must return one value per ensemble bin.
Initial-state construction is performed on the CPU; the solver owns the final
conversion to `CuArray`.

## Equations of motion

The complex equations used by the first-order solver are

```text
dSplus/dt = i delta Splus - 2i g E Sz
dSz/dt    = -i g conj(E) Splus + i g E conj(Splus)
```

With `E = Ex + i Ey` and `Splus = Sx + i Sy`, the equivalent real equations
used by GRAPE are

```text
dSx/dt = -delta Sy + 2g Ey Sz
dSy/dt =  delta Sx - 2g Ex Sz
dSz/dt =  2g (Ex Sy - Ey Sx)
```

When `E = 0`, transverse free evolution is

```text
Sx(t) = Sx(0) cos(delta t) - Sy(0) sin(delta t)
Sy(t) = Sx(0) sin(delta t) + Sy(0) cos(delta t)
```

## Ensemble layout

The ensemble is a Cartesian grid over detuning and coupling.

The current model assumes that detuning and coupling are statistically
independent. Therefore their joint weight factorizes as

```text
p(delta, g) = p_delta(delta) p_g(g)
```

The intended matrix layout is

```text
(M_delta, M_g)
```

where rows correspond to detuning bins and columns correspond to coupling
bins. Flattened GRAPE arrays are coupling-fast within each detuning bin:

```text
(delta_1, g_1), (delta_1, g_2), ..., (delta_2, g_1), ...
```

Because Julia arrays are column-major, a `g`-fast flattened result is restored
to the `(M_delta, M_g)` matrix layout with

```julia
permutedims(reshape(flat, M_g, M_delta))
```

Directly calling `reshape(flat, M_delta, M_g)` would instead interpret the
flattened data as detuning-fast and is therefore incorrect.

Ensemble weights represent population fractions. Frequency and coupling
distributions are normalized by default, so their retained bin probabilities
sum to one. A caller may explicitly set `renormalize=false` when the omitted
tails should reduce the simulated population.
Consequently, with the default normalization `N_total = N`. If either
distribution uses `renormalize=false`, `N_total` is the population retained
inside the chosen finite windows and may be smaller than `N`.

## Pulse construction

Pulse constructors live in `src/pulses/pulses.jl`; operations that modify
discrete pulse samples live in `src/pulses/pulse_editing.jl`. Supported
configuration kinds are `:gaussian`, `:wurst`, `:three_wurst`, `:constant`,
and `:custom`. A `:three_wurst` pulse consists of three consecutive WURST
segments and requires three positive durations.

For a Gaussian pulse, `gaussian_pulse_amplitude` uses the Bloch-rotation area
convention

```text
area = integral 2 g E(t) dt
```

which matches the factor `2gE` in the first-order Bloch equations.

## RASE optimization timeline

The current RASE workflow consists of:

```text
initial weak coherence
-> free dephasing
-> piecewise-constant control pulse
-> free rephasing
-> terminal objective at the expected echo time
```

The objective has three roles:

1. collective phase alignment (`Jcoh`) is the primary target;
2. population (`Jpop`) is a guard rail;
3. instantaneous-frequency smoothness (`Jfreq`) is an experimental
   regularizer.

The optimizer performs gradient ascent.

The default normalized initial population inside the target window is
`Sz_init = -0.499`. Membership in the target window is defined strictly by

```text
abs(delta) < delta_window
```

so points exactly on the boundary are outside the target window.

## Solver output

The first-order solver stores full trajectories as `Sp_keep` and `Sz_keep`;
real and imaginary parts are derived on demand rather than saved as duplicate
`Sx_keep` and `Sy_keep` arrays. Flattened `delta_b`, `g_b`, and `Nj` are saved
explicitly in g-fast order. Plotting code must use these flat arrays instead
of calling `vec(Nj_2d)`. Time-grid construction belongs to the solver, not to
the ensemble builder. CUDA memory-pool reclamation is opt-in through
`clean_gpu=true`.

## Conventions still to be unified

The current source code is not yet consistent on the following values. These
must be resolved before the corresponding implementation is frozen:

- Frequency and coupling distributions are constructed separately for the
  full solver and for GRAPE.
- The full solver stores bin-scaled collective spins, while GRAPE stores
  normalized per-spin Bloch coordinates. Conversion between them needs one
  explicit, tested interface.

Until these points are decided, refactoring must preserve existing numerical
behavior rather than silently choosing a new convention.
