"""Comparison plots for the manifold analysis (healthy vs. astrocytoma)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_pca_trajectories(embeddings, labels, colors, title, out_path,
                           axis_labels=("PC1", "PC2", "PC3")):
    """3D trajectory plot (top-3 dims of whatever embedding is passed in), one line
    per condition. `axis_labels` defaults to PCA's but is overridable for other
    3-dim embeddings (e.g. the smoothed-factor trajectory)."""
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
