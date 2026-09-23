"""Dimensionality reduction and trajectory-structure methods.

Method roles (see project notes for the reasoning):
  - PCA is the quantitative backbone: it's the only one of these that preserves
    true Euclidean distances/angles, which curvature, tangling, and jPCA all
    require to be meaningful. It also gives the eigenvalue spectrum used by
    the dimensionality metrics in `geometry.py`.
  - `fit_smoothed_factors` is a GPFA-*inspired* method for continuous-valued
    data: factor analysis (like PCA, but models per-channel noise) followed by
    per-factor Gaussian-process smoothing. This is NOT the original GPFA
    (Yu et al. 2009), which is built for spike counts via a Poisson-like
    observation model; that doesn't apply to continuous region/EEG amplitude
    data. This is the natural continuous-data analogue -- same idea (shared
    latent factors + explicit temporal smoothness), different observation
    model -- and is labeled as such throughout rather than called "GPFA".
  - jPCA is run on top of the PCA space and specifically looks for rotational
    (oscillatory) structure.
  - Isomap / UMAP are kept for qualitative visualization only -- not used for
    any of the quantitative geometry metrics, since neither reliably
    preserves the distances those metrics depend on.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.manifold import Isomap


def fit_pca(X, n_components):
    """Returns (embedding [T, n_components], explained_variance, pca_object)."""
    pca = PCA(n_components=n_components)
    embedding = pca.fit_transform(np.asarray(X, dtype=float))
    return embedding, pca.explained_variance_, pca


def fit_isomap(X, n_components=3, n_neighbors=15):
    iso = Isomap(n_components=n_components, n_neighbors=n_neighbors)
    return iso.fit_transform(np.asarray(X, dtype=float))


def fit_umap(X, n_components=3, n_neighbors=15, random_state=0):
    """Optional -- returns None if umap-learn isn't installed, so callers can
    skip the (visualization-only) UMAP panel without failing the whole run."""
    try:
        import umap
    except ImportError:
        return None
    reducer = umap.UMAP(
        n_components=n_components, n_neighbors=n_neighbors, random_state=random_state
    )
    return reducer.fit_transform(np.asarray(X, dtype=float))


def fit_smoothed_factors(X, times, n_factors, length_scale=20.0, noise_level=1e-2,
                          max_points=1000):
    """GPFA-inspired: factor analysis + per-factor GP smoothing (see module docstring).

    `times` are in the same units as `length_scale` (ms here). `max_points`
    subsamples the timebase before the GP fit (each fit is O(T^3)); the
    smoothed curve is then linearly interpolated back to the full timebase.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel

    X = np.asarray(X, dtype=float)
    times = np.asarray(times, dtype=float)

    fa = FactorAnalysis(n_components=n_factors, random_state=0)
    factors_raw = fa.fit_transform(X)

    T = X.shape[0]
    stride = max(1, T // max_points)
    sub_idx = np.arange(0, T, stride)
    t_sub = times[sub_idx].reshape(-1, 1)

    kernel = RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_level)
    smoothed = np.empty_like(factors_raw)
    smoothed_at_subsample = np.empty((len(sub_idx), n_factors))
    for k in range(n_factors):
        # optimizer=None: fixed kernel, single posterior-mean solve (no repeated
        # log-marginal-likelihood optimization), which is what keeps this
        # tractable at T in the thousands.
        gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=True)
        gp.fit(t_sub, factors_raw[sub_idx, k])
        mean_sub = gp.predict(t_sub)
        smoothed_at_subsample[:, k] = mean_sub
        smoothed[:, k] = np.interp(times, t_sub.ravel(), mean_sub)

    # Variance-retained is computed at the subsampled points only, comparing the GP's
    # posterior mean there against the *same* raw points it was fit on. Comparing full-
    # resolution raw variance against the full-resolution *interpolated* curve instead
    # (as an earlier version of this function's caller did) isn't a like-for-like
    # comparison -- piecewise-linear reconstruction from a subsample can legitimately
    # have higher or lower variance than the original full-resolution noisy series
    # regardless of how much the GP itself smoothed, which showed up as a nonsensical
    # >100%-retained result for one condition. This version is a proper shrinkage
    # ratio and is guaranteed <= 100% (up to floating-point slack) by construction.
    raw_var_sub = np.var(factors_raw[sub_idx], axis=0).sum()
    smoothed_var_sub = np.var(smoothed_at_subsample, axis=0).sum()
    variance_retained_fraction = float(smoothed_var_sub / raw_var_sub) if raw_var_sub > 0 else 0.0

    return {
        "loadings": fa.components_,
        "factors_raw": factors_raw,
        "factors_smoothed": smoothed,
        "variance_retained_fraction": variance_retained_fraction,
    }


def fit_jpca(X, dt):
    """Fit the best-fit linear rotational dynamics M (skew-symmetric) such
    that dX/dt ~= X @ M, following Churchland et al. 2012. Returns the fitted
    M, its eigendecomposition, the dominant rotation plane, and the
    trajectory projected into that plane.
    """
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0)
    Xdot = np.gradient(Xc, dt, axis=0)

    k = Xc.shape[1]
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]

    # Build the design matrix: each column is X @ E_ij for a basis
    # skew-symmetric matrix E_ij (1 at (i,j), -1 at (j,i)); solving the
    # resulting linear least-squares problem gives the coefficients that
    # reconstruct the best-fit skew-symmetric M.
    A = np.stack([_apply_skew_basis(Xc, i, j).reshape(-1) for i, j in pairs], axis=1)
    b = Xdot.reshape(-1)
    coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)

    M = np.zeros((k, k))
    for c, (i, j) in zip(coeffs, pairs):
        M[i, j] = c
        M[j, i] = -c

    eigvals, eigvecs = np.linalg.eig(M)
    order = np.argsort(-np.abs(eigvals.imag))
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    plane = np.stack([eigvecs[:, 0].real, eigvecs[:, 0].imag], axis=1)
    plane, _ = np.linalg.qr(plane)
    projection = Xc @ plane

    return {"M": M, "eigenvalues": eigvals, "plane": plane, "projection": projection}


def _apply_skew_basis(X, i, j):
    """X @ E_ij for the skew-symmetric basis matrix E_ij, without building E_ij
    explicitly: (X @ E_ij)[:, i] = -X[:, j], (X @ E_ij)[:, j] = X[:, i], zero elsewhere.
    """
    out = np.zeros_like(X)
    out[:, i] = -X[:, j]
    out[:, j] = X[:, i]
    return out
