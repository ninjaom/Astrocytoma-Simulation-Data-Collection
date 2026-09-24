"""Tracks region-level dimensionality over time as the lesion.py lesion grows
within a single run, against a static healthy baseline. See CODE_OVERVIEW.md for
the methodology. Outputs go to outputs/lesion_growth/ as PNGs, plus a text summary
on stdout."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import lesion
import mainsimulation as sim_module
from analysis import geometry, plotting
from analysis.timeseries import drop_burn_in, squeeze_monitor_data

OUT_DIR = Path(__file__).parent / "outputs" / "lesion_growth"
COLORS = {"Healthy baseline": "k", "Growing lesion": "r"}

BURN_IN_MS = 200
WINDOW_SIZE = 1500  # samples (population period = 1 ms, so 1.5 s per window)
WINDOW_STEP = 250   # samples between window centers


def run_healthy_baseline(conn, cpl, integrator, duration):
    pop_monitor = sim_module.build_population_monitor()
    (t_pop, d_pop), = sim_module.run_simulation(
        sim_module.build_models()[0], conn, cpl, integrator, [pop_monitor], duration=duration
    )
    data = sim_module.clean_data(squeeze_monitor_data(d_pop), "Healthy baseline population")
    return drop_burn_in(t_pop, data, BURN_IN_MS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=sim_module.SIM_LENGTH_MS,
                         help="Simulation length in ms (default: %(default)s)")
    parser.add_argument("--width-ms", type=float, default=4000.0,
                         help="Approximate 10%%-to-90%% lesion growth duration in ms (default: %(default)s)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    conn = sim_module.load_connectivity()
    cpl = sim_module.build_coupling()
    integrator_healthy, integrator_lesion = sim_module.build_integrators()
    lesion_idx = lesion.find_region_index(conn)
    print(f"Lesion region: index {lesion_idx}, label {conn.region_labels[lesion_idx]!r} "
          f"(chosen as the highest-degree hub region in this connectome)")

    print(f"\nRunning healthy baseline ({args.duration:g} ms)...")
    t_healthy, d_healthy = run_healthy_baseline(conn, cpl, integrator_healthy, args.duration)
    print(f"  shape {d_healthy.shape}")

    print(f"\nRunning growing-lesion condition ({args.duration:g} ms, "
          f"growth width {args.width_ms:g} ms)...")
    pop_monitor = sim_module.build_population_monitor()
    monitor_results, (chunk_times, severities) = lesion.run_growing_lesion(
        conn, cpl, integrator_lesion, [pop_monitor], lesion_idx,
        total_duration_ms=args.duration, width_ms=args.width_ms,
    )
    t_lesion_raw, d_lesion_raw = monitor_results[0]
    d_lesion_clean = sim_module.clean_data(squeeze_monitor_data(d_lesion_raw), "Growing lesion population")
    t_lesion, d_lesion = drop_burn_in(t_lesion_raw, d_lesion_clean, BURN_IN_MS)
    print(f"  shape {d_lesion.shape}")

    # ---- Sliding-window dimensionality for both conditions ----
    print("\n=== Sliding-window participation ratio (population space) ===")
    pr_by_condition = {}
    for label, (t, d) in [("Healthy baseline", (t_healthy, d_healthy)),
                           ("Growing lesion", (t_lesion, d_lesion))]:
        centers_idx, pr = geometry.sliding_participation_ratio(d, WINDOW_SIZE, WINDOW_STEP)
        centers_time = t[centers_idx]
        pr_by_condition[label] = (centers_time, pr)
        print(f"  {label:18s} PR range [{pr.min():.2f}, {pr.max():.2f}], "
              f"start={pr[0]:.2f}, end={pr[-1]:.2f}")

    # Null-comparison check against the healthy baseline -- see CODE_OVERVIEW.md.
    lesion_times, lesion_pr = pr_by_condition["Growing lesion"]
    healthy_times, healthy_pr = pr_by_condition["Healthy baseline"]
    severity_at_lesion_times = np.interp(lesion_times, chunk_times, severities)
    severity_at_healthy_times = np.interp(healthy_times, chunk_times, severities)
    lesion_correlation = float(np.corrcoef(severity_at_lesion_times, lesion_pr)[0, 1])
    null_correlation = float(np.corrcoef(severity_at_healthy_times, healthy_pr)[0, 1])
    print(f"\nCorrelation of dimensionality with the severity-schedule's time course:")
    print(f"  growing lesion (real effect + noise): r = {lesion_correlation:.3f}")
    print(f"  healthy baseline (noise only, null):  r = {null_correlation:.3f}")
    print("  A lesion correlation only meaningfully stronger than the null one is "
          "good evidence of a real effect from a single run; comparable magnitudes "
          "mean it isn't distinguishable from this run's own estimation noise.")

    plotting.plot_severity_and_dimensionality(
        chunk_times, severities, pr_by_condition, COLORS,
        OUT_DIR / "severity_and_dimensionality.png",
    )

    print(f"\nAll figures written to {OUT_DIR}")


if __name__ == "__main__":
    main()
