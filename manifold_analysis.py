"""Manifold analysis of simulated healthy vs. astrocytoma-affected brain dynamics.

Pipeline (see README for the methodology reasoning):
  1. Simulate both conditions, capturing region-level ("population", 192-dim)
     activity as the primary signal, plus EEG (62-dim, sensor space) as a
     secondary comparison.
  2. PCA is the quantitative backbone: full-spectrum PCA gives the
     dimensionality metrics (participation ratio, # components for 90%
     variance); a top-6-PC embedding is the shared space for jPCA, curvature,
     and tangling, since those require distance/angle-preserving coordinates.
  3. jPCA looks for rotational structure in that PCA space.
  4. A GPFA-inspired smoothed-factor model (factor analysis + per-factor GP
     smoothing) is fit as a second, independent dimensionality/structure
     estimate.
  5. Isomap and UMAP embeddings are produced for qualitative visualization
     only -- not used for any quantitative metric.

Outputs go to outputs/manifold_analysis/ as PNGs, plus a text summary on stdout.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import mainsimulation as sim_module
from analysis import geometry, manifold, plotting

OUT_DIR = Path(__file__).parent / "outputs" / "manifold_analysis"
COLORS = {"Healthy": "k", "Astrocytoma": "r"}

N_PCA_COMPONENTS = 6      # shared space for jPCA / curvature / tangling
N_GPFA_FACTORS = 8
VIZ_SUBSAMPLE = 1200      # cap for Isomap/UMAP (both are expensive at T ~ 10^4)
BURN_IN_MS = 200          # discard from every series before any analysis -- see below
TRAJECTORY_WINDOW_MS = 2000  # smoothed-trajectory plot only: a full 10 s line at 1 ms
# resolution is illegible regardless of true dimensionality; metrics still use the full duration, only this one plot is windowed for readability.


def squeeze(data):
    """TVB monitor output is (T, state_vars, nodes, modes); collapse to (T, nodes)."""
    return np.asarray(data)[:, 0, :, 0]


def drop_burn_in(time, data, burn_in_ms=BURN_IN_MS):
    """Discard the initial transient before any analysis touches the data.

    The simulator starts every region from the same fixed initial condition, which is
    not a sample from the system's actual (noise-driven, coupled) dynamics. A factor
    analysis / GP fit over the whole series can otherwise assign the single unusual
    initial sample(s) an outsized influence -- this showed up as a spurious spike
    dominating the smoothed-trajectory plot before this cutoff was added. Standard
    practice for any stochastic simulation: drop the burn-in, analyze steady state.
    """
    time = np.asarray(time)
    cutoff = np.searchsorted(time, time[0] + burn_in_ms)
    return time[cutoff:], data[cutoff:]


def run_conditions(duration):
    conn = sim_module.load_connectivity()
    healthy_model, astro_model = sim_module.build_models()
    cpl = sim_module.build_coupling()
    integrator_healthy, integrator_astro = sim_module.build_integrators()
    eeg_monitor = sim_module.build_eeg_monitor()

    results = {}
    for label, model, integrator in [
        ("Healthy", healthy_model, integrator_healthy),
        ("Astrocytoma", astro_model, integrator_astro),
    ]:
        pop_monitor = sim_module.build_population_monitor()
        (t_eeg, d_eeg), (t_pop, d_pop) = sim_module.run_simulation(
            model, conn, cpl, integrator, [eeg_monitor, pop_monitor], duration=duration
        )
        eeg_clean = sim_module.clean_data(squeeze(d_eeg), f"{label} EEG")
        pop_clean = sim_module.clean_data(squeeze(d_pop), f"{label} population")
        results[label] = {
            "eeg": drop_burn_in(t_eeg, eeg_clean),
            "population": drop_burn_in(t_pop, pop_clean),
        }
        print(f"{label}: ran {duration} ms, population shape {d_pop.shape}, eeg shape {d_eeg.shape}")
    return results


def dimensionality_summary(data):
    """Full-spectrum PCA -> participation ratio + # components for 90% variance."""
    n = min(data.shape)
    _, eigvals, _ = manifold.fit_pca(data, n_components=n)
    return {
        "participation_ratio": geometry.participation_ratio(eigvals),
        "dims_90pct": geometry.cumulative_variance_dimensionality(eigvals, 0.90),
    }


def subsample_for_viz(X, cap=VIZ_SUBSAMPLE):
    stride = max(1, X.shape[0] // cap)
    return X[::stride]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=sim_module.SIM_LENGTH_MS,
                         help="Simulation length in ms (default: %(default)s)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conditions = run_conditions(args.duration)
    labels = list(conditions.keys())
    colors = [COLORS[label] for label in labels]

    # ---- Dimensionality, both spaces (secondary EEG comparison + primary population) ----
    dim_results = {"population": {}, "eeg": {}}
    for label in labels:
        for space in ["population", "eeg"]:
            _, data = conditions[label][space]
            dim_results[space][label] = dimensionality_summary(data)

    plotting.plot_dimensionality_bars(dim_results, OUT_DIR / "dimensionality.png")

    print("\n=== Dimensionality (participation ratio / # comps for 90% variance) ===")
    for space in ["population", "eeg"]:
        for label in labels:
            r = dim_results[space][label]
            print(f"  {space:10s} {label:12s} PR={r['participation_ratio']:.2f}  "
                  f"dims_90%={r['dims_90pct']}")

    # ---- Shared 6-PC embedding of population space, per condition ----
    pca_embeddings, pca_times = {}, {}
    for label in labels:
        t_pop, data = conditions[label]["population"]
        emb, _, _ = manifold.fit_pca(data, n_components=N_PCA_COMPONENTS)
        pca_embeddings[label] = emb
        pca_times[label] = t_pop

    dt = float(np.median(np.diff(pca_times[labels[0]])))

    plotting.plot_pca_trajectories(
        [pca_embeddings[l] for l in labels], labels, colors,
        "Population-space PCA trajectory (top 3 PCs)", OUT_DIR / "pca_trajectory.png",
    )

    # ---- Curvature & tangling on the shared PCA space ----
    curvature_by_cond, tangling_by_cond = {}, {}
    print("\n=== Trajectory geometry (top-6-PC population space) ===")
    for label in labels:
        emb = pca_embeddings[label]
        kappa = geometry.trajectory_curvature(emb, dt)
        Q = geometry.trajectory_tangling(emb, dt)
        curvature_by_cond[label] = kappa
        tangling_by_cond[label] = Q
        print(f"  {label:12s} curvature: mean={np.mean(kappa):.4g} median={np.median(kappa):.4g}  "
              f"tangling Q: mean={np.mean(Q):.4g} max={np.max(Q):.4g}")

    plotting.plot_geometry_timeseries(
        pca_times[labels[0]], curvature_by_cond, COLORS,
        "Trajectory curvature over time", "Curvature", OUT_DIR / "curvature.png",
    )
    plotting.plot_geometry_timeseries(
        pca_times[labels[0]], tangling_by_cond, COLORS,
        "Trajectory tangling Q(t) over time", "Tangling Q", OUT_DIR / "tangling.png",
    )

    # ---- jPCA (rotational structure) on the shared PCA space ----
    print("\n=== jPCA (rotational structure, top-6-PC population space) ===")
    jpca_projections = []
    for label in labels:
        jp = manifold.fit_jpca(pca_embeddings[label], dt)
        jpca_projections.append(jp["projection"])
        dom_eigval = jp["eigenvalues"][0]
        rotation_freq_hz = abs(dom_eigval.imag) / (2 * np.pi) * 1000  # rad/ms -> Hz
        print(f"  {label:12s} dominant eigenvalue={dom_eigval:.4g}  "
              f"~{rotation_freq_hz:.2f} Hz rotation in the dominant jPCA plane")

    plotting.plot_jpca_plane(jpca_projections, labels, colors,
                              "jPCA dominant rotation plane", OUT_DIR / "jpca_plane.png")

    # ---- GPFA-inspired smoothed factors ----
    # Also used for visualization: a raw single trajectory at full time resolution
    # looks tangled/fuzzy regardless of true dimensionality, because per-timestep noise
    # dominates the picture at that resolution (see pca_trajectory.png). The smoothed
    # factors are the denoised version of the same population dynamics, so they're what
    # actually shows the shape the dimensionality metrics above are describing.
    print("\n=== GPFA-inspired smoothed factors (population space) ===")
    smoothed_trajectories = []
    for label in labels:
        t_pop, data = conditions[label]["population"]
        gpfa = manifold.fit_smoothed_factors(data, t_pop, n_factors=N_GPFA_FACTORS)
        smoothed_trajectories.append(gpfa["factors_smoothed"][:, :3])
        print(f"  {label:12s} {N_GPFA_FACTORS} factors, "
              f"variance retained after smoothing: {gpfa['variance_retained_fraction']:.1%}")

    window_len = int(TRAJECTORY_WINDOW_MS / dt)
    plotting.plot_pca_trajectories(
        [traj[:window_len] for traj in smoothed_trajectories], labels, colors,
        f"GPFA-inspired smoothed population trajectory (first 3 factors, first {TRAJECTORY_WINDOW_MS/1000:g} s)",
        OUT_DIR / "smoothed_trajectory.png",
        axis_labels=("Factor 1", "Factor 2", "Factor 3"),
    )

    # ---- Isomap / UMAP, qualitative visualization only ----
    print("\n=== Isomap / UMAP (visualization only, subsampled to <= "
          f"{VIZ_SUBSAMPLE} points) ===")
    isomap_embeddings, umap_embeddings = [], []
    for label in labels:
        _, data = conditions[label]["population"]
        data_sub = subsample_for_viz(data)
        isomap_embeddings.append(manifold.fit_isomap(data_sub, n_components=2))
        umap_emb = manifold.fit_umap(data_sub, n_components=2)
        umap_embeddings.append(umap_emb)
        if umap_emb is None:
            print("  umap-learn not available; skipping UMAP panel")

    plotting.plot_embedding_2d(isomap_embeddings, labels, colors,
                                "Isomap embedding (population space, qualitative)",
                                OUT_DIR / "isomap.png", "Isomap 1", "Isomap 2")
    if all(e is not None for e in umap_embeddings):
        plotting.plot_embedding_2d(umap_embeddings, labels, colors,
                                    "UMAP embedding (population space, qualitative)",
                                    OUT_DIR / "umap.png", "UMAP 1", "UMAP 2")

    print(f"\nAll figures written to {OUT_DIR}")


if __name__ == "__main__":
    main()
