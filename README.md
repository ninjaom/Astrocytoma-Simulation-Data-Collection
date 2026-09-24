# Astrocytoma Simulation Data Collection

Simulates and compares EEG and region-level ("population") activity between a
healthy brain and an astrocytoma-affected brain using
[The Virtual Brain](https://www.thevirtualbrain.org/) (TVB). Runs a Reduced
Wong-Wang model over a 192-region connectivity, records simulated EEG on 62
channels plus the underlying region-level dynamics for both conditions, and
provides three analyses:

- `mainsimulation.py` — raw EEG waveform plots (overview + left/right
  hemisphere breakdown).
- `manifold_analysis.py` — a manifold / population-dynamics analysis
  comparing static healthy vs. astrocytoma conditions (see below).
- `lesion_growth_analysis.py` — a focal, *growing* lesion within a single
  run, rather than two static endpoints (see below).

Source files keep only short comments; see `CODE_OVERVIEW.md` for what each file
is for and the reasoning behind specific design decisions (parameter values,
tuning choices, why something is structured the way it is).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python mainsimulation.py           # raw EEG waveform plots
python manifold_analysis.py        # manifold analysis, writes figures to outputs/manifold_analysis/
python lesion_growth_analysis.py   # growing-lesion analysis, writes figures to outputs/lesion_growth/
```

## Model parameters

All three scripts share the model setup in `mainsimulation.py`. Two things
were tuned deliberately, both discovered by running the pipeline and looking
closely at results that didn't make sense rather than assumed up front:

- **Coupling strength (`build_coupling`) and noise level
  (`build_integrators`)** were tuned together: too little coupling relative
  to noise leaves each region fluctuating almost independently, which is
  numerically fine but has no real shared low-dimensional structure to
  analyze (participation ratio near the full 192 regions). The current
  values were chosen by sweeping coupling/noise combinations for the
  smallest effective dimensionality that stays numerically stable (no NaNs)
  for both conditions at the full 10 s simulation length.
- **`build_integrators` always uses two different, explicit noise seeds.**
  TVB's `Additive` noise defaults to the same fixed seed (42) for every
  instance unless told otherwise, so two separately-constructed noise
  objects were silently producing bit-identical random draws — i.e. every
  "healthy" and "astrocytoma" run shared the exact same noise realization.
  For a global parameter shift (`manifold_analysis.py`'s comparison) this
  probably doesn't change the conclusion, since the shift is large and
  applies to every region. It actively matters for `lesion_growth_analysis.py`,
  where only one region differs between conditions — with shared noise, the
  other 191 regions were behaving identically in both runs, which was
  masking the one region's real effect. Fixed once, centrally, so every
  caller benefits.

## Manifold analysis

`manifold_analysis.py` treats the simulated region-level activity as a
high-dimensional neural population recording and asks whether its
low-dimensional structure differs between the healthy and astrocytoma
conditions. Methodology, briefly (full reasoning in the module docstrings):

- **PCA** is the quantitative backbone (`analysis/manifold.py`,
  `analysis/geometry.py`): it's the only reduction used here that preserves
  true distances/angles, which the geometry metrics below require to be
  meaningful. Full-spectrum PCA gives the dimensionality metrics
  (participation ratio, # components for 90% variance); a shared top-6-PC
  embedding is used for jPCA, curvature, and tangling.
- **jPCA** fits the best-fit rotational (skew-symmetric) dynamics in that PCA
  space, after Churchland et al. 2012.
- **Trajectory curvature and tangling** (`analysis/geometry.py`) quantify the
  trajectory's shape directly, after Russo et al. 2018.
- **A GPFA-inspired smoothed-factor model** (factor analysis + per-factor
  Gaussian-process smoothing) is a second, independent structure estimate.
  This is *not* the original spike-count GPFA (Yu et al. 2009) — that method's
  Poisson-like observation model doesn't apply to continuous region/EEG
  amplitude data. This is the natural continuous-data analogue: same idea
  (shared latent factors + explicit temporal smoothness prior), Gaussian
  instead of Poisson observations.
- **Isomap and UMAP** are included for qualitative visualization only, not
  for any quantitative metric — neither reliably preserves the distances the
  metrics above depend on.
- The first 200 ms of every run is discarded before any analysis (burn-in):
  every region starts from the same fixed initial condition, which isn't a
  sample from the system's actual dynamics.
- All of the above is run primarily on the **192-region population-level
  signal** (from TVB's `TemporalAverage` monitor), not the 62-channel EEG.
  EEG is a lead-field mixture across regions and is included only as a
  secondary, sensor-space comparison — in the current runs, most of the
  dimensionality effect shows up at the region level and is largely washed
  out once projected through the EEG lead field.

Current result (single run per condition, 10 s @ the parameters above):
region-level participation ratio ≈ 18 (healthy) vs. ≈ 14 (astrocytoma) —
i.e. the astrocytoma condition's dynamics are measurably lower-dimensional.
The EEG-space participation ratio is ≈ 61 for both conditions. This is from a
single stochastic realization per condition; treat it as a demonstration of
the method, not a validated finding — multiple seeds and a statistical test
would be the natural next step before drawing a real conclusion.

## Focal, growing-lesion analysis

`manifold_analysis.py`'s astrocytoma condition is a parameter shift applied
uniformly to all 192 regions at once — not how a real tumor works. A tumor is
**focal** (affects one location, not the whole brain) and **develops over
time** (not two disconnected before/after snapshots). `lesion_growth_analysis.py`
addresses both:

- **Focal**: only one region is affected — `lesion.py` picks the
  highest-degree ("hub") region in this connectome (`lPFCORB`, left orbital
  prefrontal cortex; also an anatomically plausible glioma site), on the
  reasoning that hub disruption is both well-documented in the connectome
  literature and gives the clearest test of whether this method can detect a
  focal effect at all. Only that region's model parameters, and the
  connectome edges touching it, are affected; the rest of the network keeps
  the healthy parameters throughout.
- **Growing**: severity follows a logistic growth curve from 0 (fully
  healthy) to 1 (fully Attempt-3-astrocytoma-equivalent at that one region)
  over the course of a single run, rather than jumping between two fixed
  endpoints. Mechanically, this works by calling TVB's `Simulator.run()`
  repeatedly on the same configured object in small chunks (~100 ms), and
  mutating the lesion region's model parameters and connectome edge weights
  in place between chunks — verified directly, by testing, that this
  continues the simulation seamlessly (no discontinuity, no need to
  reconfigure) and that the mutations actually take effect on subsequent
  chunks (see `lesion.py`'s module docstring for the reasoning).

The analysis tracks **region-level dimensionality (sliding-window
participation ratio) over time** within the growing-lesion run, against a
static healthy-baseline control, and reports the correlation between lesion
severity and dimensionality for both — the healthy baseline's correlation is
the important part: since nothing changes in that condition, any correlation
it shows is pure single-run estimation noise, and the growing-lesion
condition's correlation is only meaningful to the extent it exceeds that
baseline.

Current result (single run per condition, 10 s, growth centered at 5 s with
a ~5 s 10%-90% growth window): growing-lesion correlation r ≈ -0.40 vs.
healthy-baseline (null) r ≈ -0.26 — same direction as expected, moderately
stronger than the null, but not a clean separation. The figure
(`outputs/lesion_growth/severity_and_dimensionality.png`) shows this more
clearly than the correlation number alone: the two conditions track closely
through most of the run and the growing-lesion condition visibly diverges
downward only in the last ~1.5 s, once severity is nearly saturated — as
before, a single-run demonstration of the method rather than a validated
result.

## Data

The `data/` directory contains the connectivity matrix, EEG sensor
definitions, and region mapping used by the simulation (sourced from TVB's
bundled datasets).
