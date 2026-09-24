"""Figures for the manifold and lesion-growth analyses."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_pca_trajectories(embeddings, labels, colors, title, out_path,
                           axis_labels=("PC1", "PC2", "PC3")):
    """3D trajectory plot, one line per condition. `axis_labels` is overridable for
    non-PCA 3-dim embeddings (e.g. the smoothed-factor trajectory)."""
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    for emb, label, color in zip(embeddings, labels, colors):
        ax.plot(emb[:, 0], emb[:, 1], emb[:, 2], color=color, alpha=0.8, label=label, linewidth=0.8)
        ax.scatter(*emb[0, :3], color=color, marker="o", s=40)  # start
        ax.scatter(*emb[-1, :3], color=color, marker="x", s=40)  # end
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_embedding_2d(embeddings, labels, colors, title, out_path, xlabel="Dim 1", ylabel="Dim 2"):
    fig, ax = plt.subplots(figsize=(7, 6))
    for emb, label, color in zip(embeddings, labels, colors):
        if emb is None:
            continue
        ax.plot(emb[:, 0], emb[:, 1], color=color, alpha=0.6, label=label, linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_dimensionality_bars(results, out_path):
    """results: dict[space_label][condition_label] -> {'participation_ratio': x, 'dims_90pct': y}"""
    spaces = list(results.keys())
    conditions = list(next(iter(results.values())).keys())
    metrics = ["participation_ratio", "dims_90pct"]
    metric_titles = ["Participation ratio (effective dims)", "# components for 90% variance"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    width = 0.8 / len(conditions)
    x = np.arange(len(spaces))
    for ax, metric, mtitle in zip(axes, metrics, metric_titles):
        for i, cond in enumerate(conditions):
            vals = [results[space][cond][metric] for space in spaces]
            ax.bar(x + i * width, vals, width, label=cond)
        ax.set_xticks(x + width * (len(conditions) - 1) / 2)
        ax.set_xticklabels(spaces)
        ax.set_title(mtitle)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_geometry_timeseries(time, series_by_condition, colors, title, ylabel, out_path):
    fig, ax = plt.subplots(figsize=(9, 4))
    for cond, series in series_by_condition.items():
        ax.plot(time, series, label=cond, color=colors[cond], alpha=0.85, linewidth=0.9)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_severity_and_dimensionality(severity_time, severity, pr_by_condition, colors, out_path):
    """Two-panel figure: lesion severity g(t) on top, sliding-window participation
    ratio below, sharing a time axis."""
    fig, (ax_sev, ax_pr) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)

    ax_sev.plot(severity_time, severity, color="tab:purple", linewidth=1.2)
    ax_sev.set_ylabel("Lesion severity g(t)")
    ax_sev.set_title("Lesion growth schedule")
    ax_sev.set_ylim(-0.05, 1.05)

    for cond, (times, pr) in pr_by_condition.items():
        ax_pr.plot(times, pr, label=cond, color=colors[cond], marker="o", markersize=3, linewidth=1.0)
    ax_pr.set_xlabel("Time (ms)")
    ax_pr.set_ylabel("Participation ratio\n(sliding window)")
    ax_pr.set_title("Region-level effective dimensionality over time")
    ax_pr.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_jpca_plane(projections, labels, colors, title, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for proj, label, color in zip(projections, labels, colors):
        ax.plot(proj[:, 0], proj[:, 1], color=color, alpha=0.7, label=label, linewidth=0.8)
    ax.set_xlabel("jPCA dim 1")
    ax.set_ylabel("jPCA dim 2")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
