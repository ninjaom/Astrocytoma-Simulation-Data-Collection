# Astrocytoma Simulation Data Collection

Simulates and compares EEG and region-level ("population") activity between a
healthy brain and an astrocytoma-affected brain using
[The Virtual Brain](https://www.thevirtualbrain.org/) (TVB). Runs a Reduced
Wong-Wang model over a 192-region connectivity, records simulated EEG on 62
channels plus the underlying region-level dynamics for both conditions, and
provides two analyses:

- `mainsimulation.py` — raw EEG waveform plots (overview + left/right
  hemisphere breakdown).
- `manifold_analysis.py` — a manifold / population-dynamics analysis of the
  same simulations (see below).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python mainsimulation.py       # raw EEG waveform plots
python manifold_analysis.py    # manifold analysis, writes figures to outputs/manifold_analysis/
```

## Model parameters

Both scripts share the model setup in `mainsimulation.py`. The coupling
strength (`build_coupling`) and noise level (`build_integrators`) were tuned
together: too little coupling relative to noise leaves each region
fluctuating almost independently, which is numerically fine but has no real
shared low-dimensional structure to analyze (participation ratio near the
full 192 regions). The current values were chosen by sweeping coupling/noise
combinations for the smallest effective dimensionality that stays
numerically stable (no NaNs) for both conditions at the full 10 s simulation
length.

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

## Data

The `data/` directory contains the connectivity matrix, EEG sensor
definitions, and region mapping used by the simulation (sourced from TVB's
bundled datasets).
