"""Trajectory-geometry metrics: pure math on a (T, D) array + timestep `dt`.
See CODE_OVERVIEW.md for what each metric means and why."""

from __future__ import annotations

import numpy as np


def participation_ratio(eigenvalues):
    """Effective dimensionality: (sum(eig))^2 / sum(eig^2). See CODE_OVERVIEW.md."""
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    eigenvalues = eigenvalues[eigenvalues > 0]
    if eigenvalues.size == 0:
        return 0.0
    return float(eigenvalues.sum() ** 2 / np.sum(eigenvalues ** 2))


def cumulative_variance_dimensionality(eigenvalues, threshold=0.90):
    """Number of components needed to explain `threshold` fraction of variance."""
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    eigenvalues = np.clip(eigenvalues, 0, None)
    total = eigenvalues.sum()
    if total <= 0:
        return 0
    cumulative = np.cumsum(eigenvalues) / total
    return int(np.searchsorted(cumulative, threshold) + 1)


def sliding_participation_ratio(data, window_size, step):
    """Participation ratio recomputed in a sliding window over time (`data` is
    (T, D) raw channels). Returns (center_indices, pr_values)."""
    data = np.asarray(data, dtype=float)
    T = data.shape[0]
    if window_size > T:
        raise ValueError(f"window_size ({window_size}) exceeds series length ({T})")

    starts = np.arange(0, T - window_size + 1, step)
    centers = starts + window_size // 2
    pr_values = np.empty(len(starts))
    for i, start in enumerate(starts):
        window = data[start:start + window_size]
        window = window - window.mean(axis=0)  # PCA's mean-subtraction step
        cov = np.cov(window, rowvar=False)
        eigvals = np.linalg.eigvalsh(cov)
        pr_values[i] = participation_ratio(eigvals)

    return centers, pr_values


def trajectory_velocity(X, dt):
    """Central-difference velocity dX/dt, shape (T, D)."""
    return np.gradient(np.asarray(X, dtype=float), dt, axis=0)


def trajectory_curvature(X, dt, eps=1e-8):
    """n-dimensional curvature at every timepoint: kappa(t) = |a_perp(t)| / |v(t)|^2.
    See CODE_OVERVIEW.md. Returns an array of shape (T,)."""
    X = np.asarray(X, dtype=float)
    v = trajectory_velocity(X, dt)
    a = trajectory_velocity(v, dt)
    v_sq = np.sum(v ** 2, axis=1)
    v_sq_safe = np.clip(v_sq, eps, None)
    proj_scale = np.sum(a * v, axis=1) / v_sq_safe
    a_perp = a - proj_scale[:, None] * v
    kappa = np.linalg.norm(a_perp, axis=1) / v_sq_safe
    return kappa


def trajectory_tangling(X, dt, eps_frac=0.1):
    """Trajectory tangling Q(t) = max_t' [|v(t)-v(t')|^2 / (|x(t)-x(t')|^2 + eps)],
    after Russo et al. 2018. See CODE_OVERVIEW.md."""
    X = np.asarray(X, dtype=float)
    v = trajectory_velocity(X, dt)
    T = X.shape[0]

    eps = eps_frac * np.mean(np.sum((X - X.mean(axis=0)) ** 2, axis=1))
    eps = max(eps, 1e-12)

    Q = np.empty(T)
    for t in range(T):
        num = np.sum((v[t] - v) ** 2, axis=1)
        den = np.sum((X[t] - X) ** 2, axis=1) + eps
        Q[t] = np.max(num / den)
    return Q
