# Code overview

What each file in this project is for, and the reasoning behind non-obvious design
decisions. Source files keep only short comments; anything longer than a couple of
lines lives here instead, so the code stays easy to scan and the reasoning still has
somewhere to live. For the full build-and-debug narrative (what was tried, what went
wrong, how it was fixed), see `MANIFOLD_ANALYSIS_EXPLAINED.md` (local only, not part
of the repository).

## `mainsimulation.py`

Shared simulation setup used by all three entry-point scripts, plus the original
raw-EEG-waveform plotting script (`main()`).

- **`SIM_LENGTH_MS = 10000`**: the project originally ran for 10 ms, a leftover
  workaround for a since-fixed instability in an old, now-unused parameter set
  (see `ASTRO_PARAMS` below). Re-verified stable at the full 10 s with the current
  parameters before restoring this length — a real trajectory-shape analysis needs
  far more than ~10 samples.
- **`HEALTHY_PARAMS` / `ASTRO_PARAMS`**: the five Reduced Wong-Wang parameters for
  each condition, factored out as plain dicts (rather than left only inside TVB
  model objects) so `lesion.py` can build a *per-region* parameter array from the
  exact same numbers instead of duplicating them and risking drift. `HEALTHY_PARAMS`
  matches `models.ReducedWongWang()`'s own defaults exactly (verified directly) —
  spelled out here so that fact is visible rather than implicit. `ASTRO_PARAMS` is
  the third parameter set tried during early development; an earlier set (`a=0.5,
  w=0.8, I_o=0.2, tau_s=100.0, b=0.1`) produced `NaN`s over long runs and was
  dropped.
- **`build_coupling()`**: coupling strength `a=2.0`. The original value (`0.015`)
  is weak enough, relative to the noise level, that regions fluctuate almost
  independently — numerically fine, but with no real shared low-dimensional
  structure for a manifold analysis to find (participation ratio ends up near the
  full 192 regions, isotropic noise). `2.0` was chosen by sweeping coupling/noise
  combinations for the smallest effective dimensionality that stays numerically
  stable (no NaNs) for both conditions at the full 10 s duration.
- **`build_integrators(seed_healthy=42, seed_astro=43)`**: noise level `nsig=2**-8`,
  lowered from `2**-5` alongside the coupling increase (same reasoning — noise this
  large was drowning out the coupling-driven structure). The two seeds are explicit
  and always different on purpose: TVB's `Additive` noise defaults every instance to
  the same fixed seed (42) unless told otherwise, so two separately-constructed
  noise objects were silently producing bit-identical random draws — every
  "healthy" and "astrocytoma" run was driven by the exact same noise realization.
  That's mostly harmless for a global parameter shift (the shift itself is large
  and applies everywhere), but it actively breaks any analysis that isolates a
  *small* effect against a shared background (`lesion_growth_analysis.py`, where
  only one region differs between conditions) and would silently defeat any future
  multi-seed statistical comparison. Fixed once, centrally, here.
- **`build_population_monitor()`**: records the model's raw per-region state,
  not EEG. This is the right signal for population-dynamics analysis — EEG is a
  lead-field mixture across regions and loses information relative to it.
- **`run_simulation(..., monitor_list, ...)`**: takes a *list* of monitors (not a
  single one) so a single simulation run can record EEG and region-level activity
  together in one pass, rather than needing two separate runs.
- **`main()`**: the original script's behavior — plots raw EEG waveforms (overview,
  then split by hemisphere).
- **`map_channels_to_hemispheres`**: splits channels by sensor x-coordinate. In this
  sensor file, +x is the subject's left (F3, C3, P3 have x > 0) and -x is the
  right, matching the 10-20 convention of odd labels = left, even = right.
  Channels within `midline_tol` of x = 0 (Fpz, Fz, FCz, Cz, CPz, Pz, POz, Oz)
  belong to neither hemisphere and are returned separately.

## `manifold_analysis.py`

Static healthy-vs-astrocytoma comparison: builds a shared 6-component PCA embedding
of the region-level (population) activity, then runs every geometry/structure
method against that same embedding, plus a secondary EEG-space dimensionality
comparison. Method choices and their reasoning are documented in
`analysis/manifold.py` and `analysis/geometry.py`, not repeated here.

- **`BURN_IN_MS = 200`**: every region starts from the same fixed initial value,
  which isn't a sample of the system's real dynamics — discarded before any
  analysis touches the data. Found because it was dominating a smoothed-trajectory
  plot's scale (see the walkthrough doc for the full story).
- **`TRAJECTORY_WINDOW_MS = 2000`**: applies only to the smoothed-trajectory plot,
  not to any metric. A full 10 s trajectory plotted as one continuous line is
  illegible regardless of true dimensionality (a property of plotting a long noisy
  path this way, not a bug) — this windows *that one plot* to a readable excerpt.
- **Why the trajectory plot uses the GPFA-inspired smoothed factors, not the raw
  PCA trajectory**: a raw single trajectory at full time resolution looks tangled
  and fuzzy regardless of true dimensionality, because per-timestep noise
  dominates the picture at that resolution (compare `pca_trajectory.png` against
  `smoothed_trajectory.png`). The smoothed factors are the denoised version of the
  same population dynamics, so they're what actually shows the shape the
  dimensionality metrics are describing.
- **EEG vs. population space**: dimensionality is computed for both, primarily as
  a check on whether an effect visible at the region level survives being mixed
  through the EEG lead field. In the current runs it mostly doesn't (population PR
  ≈18 vs. ≈14 between conditions; EEG PR ≈61 for both) — a real, interesting
  secondary finding, not an oversight.

## `lesion.py`

A focal lesion, in one region only (not all 192 regions the way
`manifold_analysis.py`'s comparison works), that grows smoothly within a single
simulation run instead of jumping between two fixed endpoints.

- **`LESION_REGION_LABEL = "lPFCORB"`**: chosen by degree (sum of a region's
  connection weights to every other region) — the highest-degree "hub" region in
  this connectome. Reasoning: hub disruption is a well-documented real phenomenon
  (tumors at highly-connected regions cause disproportionate network-wide
  disruption) and gives the clearest test of whether this method can detect a
  focal effect at all; it's also anatomically identifiable (left orbital
  prefrontal cortex) and a clinically plausible glioma site, unlike an arbitrary
  index.
- **`sigmoid_schedule(...)`**: a logistic (S-curve) growth schedule, normalized so
  it hits exactly 0 at the start and exactly 1 at the end (a raw, un-normalized
  logistic only approaches those values asymptotically). `width_ms` is
  approximately the 10%-to-90% growth duration.
- **`run_growing_lesion(...)`**: runs the simulator in small chunks (default
  100 ms), and between chunks mutates the lesion region's model parameters and its
  connectome edge weights *in place* — interpolated from healthy toward
  astrocytoma values by the current severity `g(t)`. This relies on two things
  that were verified directly, by testing, before this function was written: that
  calling TVB's `Simulator.run()` repeatedly on the same configured object
  continues seamlessly (no gap, no need to reconfigure), and that mutating model
  parameters / connectivity weights between calls actually affects the next
  chunk's dynamics. Edge weights are recomputed from a pristine saved copy of the
  original weights on every chunk (not repeatedly rescaled in place), so
  floating-point error can't compound over the ~100 chunks in a full run.

## `lesion_growth_analysis.py`

Runs the healthy baseline and the growing-lesion condition from `lesion.py`, then
tracks region-level dimensionality *over time within* the growing-lesion run
(via `analysis.geometry.sliding_participation_ratio`) rather than only comparing
two static endpoints.

- **Null-comparison correlation check**: the script correlates lesion severity
  against windowed dimensionality for the growing-lesion condition, and — just as
  importantly — computes the *same* correlation for the healthy baseline against
  the same severity-vs-time curve, even though the baseline has no lesion at all.
  This exists because a single stochastic run's windowed participation-ratio
  estimate has its own sampling noise/drift even when nothing is actually
  changing (confirmed directly: the healthy baseline drifted by almost as much, in
  absolute terms, as the lesion condition did). Reporting the lesion condition's
  correlation alone would risk mistaking that baseline drift for a lesion effect;
  it's only meaningful to the extent it exceeds the null.
- **`WINDOW_SIZE = 1500` / `WINDOW_STEP = 250`** (samples, at 1 ms population
  sampling): a 1.5 s window gives a reasonably stable covariance estimate across
  192 channels; 250 ms steps give a reasonably smooth dimensionality-over-time
  curve without excessive recomputation.

## `analysis/geometry.py`

Pure trajectory-geometry math — takes a `(T, D)` array and a timestep, knows
nothing about brains, EEG, or simulations. Operates on an already-PCA-reduced
trajectory (except `sliding_participation_ratio`, which takes raw channels and
refits its own covariance per window).

- **`participation_ratio`**: `(sum(eig))^2 / sum(eig^2)` — effective dimensionality
  of a covariance spectrum; equals D if variance is spread evenly across D
  dimensions, approaches 1 if one eigenvalue dominates. Used throughout instead of
  an arbitrary variance-explained cutoff.
- **`trajectory_curvature`**: the n-dimensional generalization of the classic 3D
  Frenet curvature (`|v x a| / |v|^3`) — how much of the acceleration points
  sideways relative to the direction of travel, rather than just speeding up or
  slowing down along the same line.
- **`trajectory_tangling`**: after Russo et al. 2018 — at each timepoint, the
  worst-case ratio of "how differently is the trajectory moving" to "how close is
  it in state" against every other timepoint. High tangling means the near future
  is hard to predict from position alone. `eps` regularizes by a fraction of the
  trajectory's own variance (the original paper's approach), rather than needing
  pre-normalized units.
- **`sliding_participation_ratio`**: participation ratio recomputed fresh from
  each window's own covariance, for tracking dimensionality *changing* within one
  run (built specifically for `lesion_growth_analysis.py`).

## `analysis/manifold.py`

Dimensionality reduction and trajectory-structure methods. Method roles:

- **PCA is the quantitative backbone** for everything in this project: it's the
  only one of these that preserves true Euclidean distances/angles, which
  curvature, tangling, and jPCA all require to be meaningful. It also gives the
  eigenvalue spectrum the dimensionality metrics use.
- **`fit_smoothed_factors` is GPFA-*inspired*, not GPFA.** The original GPFA
  (Yu et al. 2009) is built for spike-count data via a Poisson-like observation
  model, which doesn't apply to this project's continuous region/EEG amplitude
  data. This is the natural continuous-data analogue — the same core idea (shared
  latent factors + an explicit temporal-smoothness prior) with factor analysis
  (PCA's cousin that separately models per-channel noise) plus a per-factor
  Gaussian process smoother, instead of a Poisson observation model. The GP fit is
  done on a subsample of the timebase (each fit costs O(T^3)) and interpolated
  back to full resolution; `variance_retained_fraction` is computed only at the
  subsampled points themselves (comparing the full-resolution raw series against
  the full-resolution *interpolated* curve isn't a like-for-like comparison and
  previously produced a nonsensical >100%-retained result for one condition).
- **jPCA fits the best-fit rotational (skew-symmetric) dynamics** in the PCA
  space, after Churchland et al. 2012: builds a design matrix from every possible
  pairwise-rotation basis matrix and solves a linear least-squares problem for the
  coefficients that reconstruct the best-fit skew-symmetric matrix `M`, then uses
  its eigendecomposition to find the plane where rotation is clearest.
- **Isomap and UMAP are qualitative visualization only** — used nowhere in any
  quantitative metric, since neither reliably preserves the distances those
  metrics depend on. `fit_umap` returns `None` if `umap-learn` isn't installed, so
  callers can skip that one panel without failing the whole run.

## `analysis/plotting.py`

Every figure the two analysis scripts produce. No non-obvious design decisions
beyond what's already covered above — mostly straightforward matplotlib.
`plot_pca_trajectories`'s `axis_labels` parameter exists so the same 3D-trajectory
plotting code can be reused for both a genuine PCA embedding and the GPFA-inspired
smoothed-factor trajectory, which aren't the same kind of space.

## `analysis/timeseries.py`

Two small helpers (`squeeze_monitor_data`, `drop_burn_in`) shared between
`manifold_analysis.py` and `lesion_growth_analysis.py`. Split out specifically so
the two scripts can't drift apart on this handling — e.g. one fixing the burn-in
cutoff and the other not.

- **`drop_burn_in`**: every region starts from the same fixed initial condition,
  which isn't a sample of the system's actual (noise-driven, coupled) dynamics.
  Found because it was dominating a smoothed-trajectory plot's scale — a single
  unusual early sample can get an outsized influence in a factor analysis / GP fit
  over the whole series, even when the raw signal's overall mean/variance looks
  fine at a glance. Standard practice for any stochastic simulation: drop the
  burn-in, analyze steady state.
