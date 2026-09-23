"""Trajectory-geometry metrics for low-dimensional neural population dynamics.

These operate on an already-reduced trajectory X of shape (T, D) (e.g. the top D
principal components of a population recording over time), sampled at a fixed
timestep `dt`. They quantify the *shape* of the trajectory rather than its raw
values, which is what makes them comparable across conditions with otherwise
different amplitudes/scales.
"""

from __future__ import annotations

import numpy as np


def participation_ratio(eigenvalues):
    """Effective dimensionality of a covariance spectrum.

    PR = (sum(eig))^2 / sum(eig^2). Equals D if all D eigenvalues are equal
    (activity spread evenly across D dimensions), and approaches 1 if a single
    eigenvalue dominates (activity effectively 1-dimensional). Standard measure
    in the population-dynamics literature for "how many dimensions are really
    being used" without picking an arbitrary variance-explained cutoff.
    """
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


def trajectory_velocity(X, dt):
    """Central-difference velocity dX/dt, shape (T, D)."""
    return np.gradient(np.asarray(X, dtype=float), dt, axis=0)


def trajectory_curvature(X, dt, eps=1e-8):
    """Generalized (n-dimensional) curvature of a trajectory at every timepoint.

    kappa(t) = |a_perp(t)| / |v(t)|^2, where a_perp is the acceleration
    component orthogonal to velocity. This is the natural n-D generalization
    of the classic 3D Frenet curvature |v x a| / |v|^3: a straight (even if
    accelerating) trajectory has kappa = 0; a tightly turning one has large
    kappa. Returns an array of shape (T,).
    """
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
    """Trajectory tangling Q(t), after Russo et al. 2018.

    Q(t) = max_{t'} [ |v(t) - v(t')|^2 / (|x(t) - x(t')|^2 + eps) ]

    High Q at a timepoint means there exists another point on the trajectory
    that is nearby in state (small denominator) but moving in a very
    different direction (large numerator) -- i.e. the local future is hard to
    predict from position alone. `eps` is set as `eps_frac` times the mean
    squared distance from the trajectory mean, following the original paper's
    approach of regularizing by a fraction of the data's overall variance
    (prevents division blow-up near self-intersections without needing units
    to be pre-normalized).
    """
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
