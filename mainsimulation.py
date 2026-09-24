"""Shared simulation setup (used by all three scripts) plus the original raw-EEG
plotting script. See CODE_OVERVIEW.md for design-decision reasoning."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tvb.datatypes.region_mapping import RegionMapping
from tvb.datatypes.sensors import SensorsEEG
from tvb.simulator.lab import connectivity, coupling, integrators, models, monitors, simulator
from tvb.simulator.noise import Additive

DATA_DIR = Path(__file__).parent / "data"
CONNECTIVITY_FILE = DATA_DIR / "connectivity_192.zip"
SENSORS_FILE = DATA_DIR / "eeg_unitvector_62.txt.bz2"
REGION_MAPPING_FILE = DATA_DIR / "regionMapping_16k_192.txt"

# See CODE_OVERVIEW.md for why these constants have the values they do.
SIM_LENGTH_MS = 10000
EEG_PERIOD_MS = 0.9765625  # 1024 Hz sampling
POPULATION_PERIOD_MS = 1.0  # region-level ("population") sampling, used for manifold analysis

# Reduced Wong-Wang parameters per condition, shared with lesion.py.
HEALTHY_PARAMS = {"a": 0.27, "w": 0.6, "I_o": 0.33, "tau_s": 100.0, "b": 0.108}
ASTRO_PARAMS = {"a": 0.6, "w": 0.9, "I_o": 0.1, "tau_s": 80.0, "b": 0.2}


def load_connectivity():
    # conn = connectivity.Connectivity.from_file()  # default connectivity (Large Scale)
    conn = connectivity.Connectivity.from_file(str(CONNECTIVITY_FILE))
    conn.configure()
    return conn


def build_models():
    healthy_model = models.ReducedWongWang(**{k: np.array([v]) for k, v in HEALTHY_PARAMS.items()})
    astrocytoma_model = models.ReducedWongWang(**{k: np.array([v]) for k, v in ASTRO_PARAMS.items()})
    return healthy_model, astrocytoma_model

def build_coupling():
    # a=2.0: see CODE_OVERVIEW.md for how this and the noise level below were tuned.
    return coupling.Linear(a=np.array([2.0]))


def build_integrators(seed_healthy=42, seed_astro=43):
    # nsig=2**-8; distinct seeds per condition are required, not cosmetic -- see
    # CODE_OVERVIEW.md.
    noise_healthy = Additive(nsig=np.array([2**-8]), noise_seed=seed_healthy)
    integrator_healthy = integrators.HeunStochastic(dt=0.1, noise=noise_healthy)

    noise_astro = Additive(nsig=np.array([2**-8]), noise_seed=seed_astro)
    integrator_astro = integrators.HeunStochastic(dt=0.1, noise=noise_astro)
    return integrator_healthy, integrator_astro

def build_eeg_monitor():
    sensors_eeg = SensorsEEG.from_file(str(SENSORS_FILE))
    rm = RegionMapping.from_file(str(REGION_MAPPING_FILE))
    print("Loaded RegionMapping shape:", rm.array_data.shape)
    return monitors.EEG(
        sensors=sensors_eeg,
        region_mapping=rm,
        period=EEG_PERIOD_MS,
        reference="average",  # Use average reference
    )

def build_population_monitor(period=POPULATION_PERIOD_MS):
    """Region-level (source-space) monitor -- see CODE_OVERVIEW.md."""
    return monitors.TemporalAverage(period=period)


def run_simulation(model, conn, cpl, integrator, monitor_list, duration=SIM_LENGTH_MS):
    """Run one condition with one or more monitors. Returns a list of (time, data)
    tuples in the same order as `monitor_list`."""
    sim = simulator.Simulator(
        model=model,
        connectivity=conn,
        coupling=cpl,
        integrator=integrator,
        monitors=monitor_list,
        simulation_length=duration,
    )
    sim.configure()
    return sim.run()

def clean_data(data, label):
    n_bad = np.count_nonzero(~np.isfinite(data))
    if n_bad:
        print(f"{label}: replacing {n_bad} non-finite values with 0")
    return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

def plot_overview(time_healthy, data_healthy, time_astro, data_astro):
    """All 62 channels per condition on one axis (original plot)."""
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time_healthy, data_healthy[:, 0, :, 0], "k", alpha=0.8)
    plt.title("Healthy Brain EEG")
    plt.xlabel("Time (ms)")
    plt.ylabel("EEG Amplitude")
    plt.subplot(2, 1, 2)
    plt.plot(time_astro, data_astro[:, 0, :, 0], "r", alpha=0.8)
    plt.title("Astrocytoma-Affected Brain EEG")
    plt.xlabel("Time (ms)")
    plt.ylabel("EEG Amplitude")
    plt.tight_layout()
    plt.show()


def map_channels_to_hemispheres(sensors, midline_tol=0.05):
    """Split channels by sensor x-coordinate -- convention in CODE_OVERVIEW.md."""
    x = sensors.locations[:, 0]
    left_channels = np.flatnonzero(x > midline_tol)
    right_channels = np.flatnonzero(x < -midline_tol)
    midline_channels = np.flatnonzero(np.abs(x) <= midline_tol)
    return left_channels, right_channels, midline_channels


def plot_eeg(time, data, title, color, left_channels, right_channels, midline_channels):
    groups = [
        ("Left Hemisphere", left_channels),
        ("Right Hemisphere", right_channels),
        ("Midline", midline_channels),
    ]
    plt.figure(figsize=(15, 14))
    for i, (name, channels) in enumerate(groups, start=1):
        plt.subplot(3, 1, i)
        for channel in channels:
            plt.plot(time, data[:, 0, channel, 0], color=color, alpha=0.5)
        plt.title(f"{title} - {name} ({len(channels)} channels)")
        plt.xlabel("Time (ms)")
        plt.ylabel("EEG Amplitude (μV)")
        plt.ylim([-1.5, 1.5])  # Adjust based on your data

    plt.tight_layout()
    plt.show()


def main():
    conn = load_connectivity()
    healthy_model, astrocytoma_model = build_models()
    cpl = build_coupling()
    integrator_healthy, integrator_astro = build_integrators()
    eeg_monitor = build_eeg_monitor()

    time_healthy, data_healthy = run_simulation(healthy_model, conn, cpl, integrator_healthy, [eeg_monitor])[0]
    time_astro, data_astro = run_simulation(astrocytoma_model, conn, cpl, integrator_astro, [eeg_monitor])[0]
    print("Simulations complete.")

    data_healthy = clean_data(data_healthy, "Healthy")
    data_astro = clean_data(data_astro, "Astrocytoma")

    plot_overview(time_healthy, data_healthy, time_astro, data_astro)

    channel_groups = map_channels_to_hemispheres(eeg_monitor.sensors)
    plot_eeg(time_healthy, data_healthy, "Healthy Brain EEG", "k", *channel_groups)
    plot_eeg(time_astro, data_astro, "Astrocytoma-Affected Brain EEG", "r", *channel_groups)


if __name__ == "__main__":
    main()
