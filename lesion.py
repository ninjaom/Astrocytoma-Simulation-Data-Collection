"""A focal lesion that grows within a single simulation run. See CODE_OVERVIEW.md
for the design reasoning."""

from __future__ import annotations

import numpy as np
from tvb.simulator.lab import models, simulator

import mainsimulation as sim_module

# Highest-degree hub region in this connectome -- see CODE_OVERVIEW.md for why.
LESION_REGION_LABEL = "lPFCORB"

EDGE_REDUCTION_MAX = 0.7  # at full severity, edges touching the lesion are cut to 30%
DEFAULT_CHUNK_MS = 100.0  # how often severity is updated during the run


def find_region_index(conn, label=LESION_REGION_LABEL):
    matches = np.flatnonzero(conn.region_labels == label)
    if len(matches) == 0:
        raise ValueError(f"Region label {label!r} not found in this connectivity's region_labels")
    return int(matches[0])


def sigmoid_schedule(t_ms, midpoint_ms, width_ms, total_duration_ms):
    """Logistic severity curve, normalized so g(0)=0 and g(total)=1 exactly.
    `width_ms` is approximately the 10%-to-90% growth duration."""
    rate = 4.394 / width_ms

    def raw(t):
        return 1.0 / (1.0 + np.exp(-rate * (t - midpoint_ms)))

    g0, g1 = raw(0.0), raw(total_duration_ms)
    return (raw(t_ms) - g0) / (g1 - g0)


def build_lesion_model(n_regions):
    """Per-node parameter arrays, all starting at healthy values (the lesion
    region's values get overwritten chunk-by-chunk in run_growing_lesion)."""
    return models.ReducedWongWang(**{
        key: np.full(n_regions, value) for key, value in sim_module.HEALTHY_PARAMS.items()
    })


def run_growing_lesion(conn, cpl, integrator, monitor_list, lesion_idx,
                        total_duration_ms, chunk_ms=DEFAULT_CHUNK_MS,
                        midpoint_ms=None, width_ms=4000.0,
                        edge_reduction_max=EDGE_REDUCTION_MAX):
    """Run one simulation with a focal lesion that grows smoothly over its course.
    Returns (monitor_results, (chunk_start_times, severities))."""
    n = conn.weights.shape[0]
    if midpoint_ms is None:
        midpoint_ms = total_duration_ms / 2

    model = build_lesion_model(n)
    healthy_weights = conn.weights.copy()  # pristine reference recomputed from each chunk

    n_chunks = int(np.ceil(total_duration_ms / chunk_ms))
    sim = simulator.Simulator(
        model=model, connectivity=conn, coupling=cpl, integrator=integrator,
        monitors=monitor_list, simulation_length=chunk_ms,
    )
    sim.configure()

    per_monitor_chunks = [[] for _ in monitor_list]
    chunk_start_times, severities = [], []

    for i in range(n_chunks):
        t_start = i * chunk_ms
        g = float(sigmoid_schedule(t_start, midpoint_ms, width_ms, total_duration_ms))
        g = float(np.clip(g, 0.0, 1.0))
        chunk_start_times.append(t_start)
        severities.append(g)

        for pname, healthy_v in sim_module.HEALTHY_PARAMS.items():
            astro_v = sim_module.ASTRO_PARAMS[pname]
            getattr(model, pname)[lesion_idx] = healthy_v + g * (astro_v - healthy_v)

        conn.weights[lesion_idx, :] = healthy_weights[lesion_idx, :] * (1 - g * edge_reduction_max)
        conn.weights[:, lesion_idx] = healthy_weights[:, lesion_idx] * (1 - g * edge_reduction_max)

        result = sim.run()
        for j, (t, d) in enumerate(result):
            per_monitor_chunks[j].append((t, d))

    monitor_results = []
    for chunks in per_monitor_chunks:
        times = np.concatenate([t for t, _ in chunks])
        data = np.concatenate([d for _, d in chunks], axis=0)
        monitor_results.append((times, data))

    conn.weights[:] = healthy_weights  # restore, so callers can reuse conn afterward

    return monitor_results, (np.array(chunk_start_times), np.array(severities))
