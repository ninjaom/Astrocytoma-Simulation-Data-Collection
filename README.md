# Astrocytoma Simulation Data Collection

Simulates and compares EEG activity between a healthy brain and an
astrocytoma-affected brain using [The Virtual Brain](https://www.thevirtualbrain.org/)
(TVB). Runs a Reduced Wong-Wang model over a 192-region connectivity, records
simulated EEG on 62 channels for both conditions, and plots the results
(overview + left/right hemisphere breakdown).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python mainsimulation.py
```

## Data

The `data/` directory contains the connectivity matrix, EEG sensor
definitions, and region mapping used by the simulation (sourced from TVB's
bundled datasets).