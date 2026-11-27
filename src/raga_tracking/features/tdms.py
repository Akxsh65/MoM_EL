"""W2 features: time-delayed melody surface (TDMS), after Gulati et al., ISMIR 2016.

For each window: 2-D histogram over (pc(t), pc(t+tau)) using only pairs where BOTH frames are voiced,
element-wise power compression, circular Gaussian smoothing (the surface is octave-folded, so it wraps),
flatten, L1-normalise. Bin 0 is centred on Sa, as in W1.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def _bin_index(cents: np.ndarray, n_bins: int) -> np.ndarray:
    w = 1200.0 / n_bins
    return np.floor(np.mod(np.where(np.isfinite(cents), cents, 0.0) + w / 2, 1200.0) / w).astype(int) % n_bins


def tdms_windows(cents_windows: np.ndarray, tau_s: float = 0.3, n_bins: int = 40, alpha: float = 0.75,
                 sigma: float = 1.0, fs: float = 100.0, batch: int = 128) -> np.ndarray:
    """(n_windows, n_frames) cents (NaN = unvoiced) -> (n_windows, n_bins**2) float32 L1-normalised surfaces.

    sigma is in bins (0 disables smoothing); alpha is the compression exponent (1 disables it).
    A window with no voiced pair returns a uniform surface (callers should skip it via the validity flag).
    """
    c = np.asarray(cents_windows)
    tau = int(round(tau_s * fs))
    n, F = c.shape
    B2 = n_bins * n_bins
    out = np.empty((n, B2), dtype=np.float32)
    for s in range(0, n, batch):
        cb = c[s:s + batch].astype(float)
        m = cb.shape[0]
        a, b = cb[:, : F - tau], cb[:, tau:]
        ok = np.isfinite(a) & np.isfinite(b)
        cell = _bin_index(a, n_bins) * n_bins + _bin_index(b, n_bins)
        flat = (cell + np.arange(m)[:, None] * B2)[ok]
        S = np.bincount(flat, minlength=m * B2).reshape(m, n_bins, n_bins).astype(float)
        if alpha != 1.0:
            S = S ** alpha
        if sigma > 0:
            S = gaussian_filter(S, sigma=(0, sigma, sigma), mode="wrap", truncate=4.0)
        S = S.reshape(m, B2)
        tot = S.sum(1, keepdims=True)
        S = np.where(tot > 0, S / np.where(tot > 0, tot, 1), 1.0 / B2)
        out[s:s + m] = S
    return out
