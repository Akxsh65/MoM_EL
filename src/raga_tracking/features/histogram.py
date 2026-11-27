"""W1 features: octave-folded pitch-class histogram per window."""
from __future__ import annotations

import numpy as np


def window_histograms(cents_windows: np.ndarray, n_bins: int = 24, smooth: float = 1e-6) -> np.ndarray:
    """(n_windows, n_frames) cents with NaN for unvoiced -> (n_windows, n_bins) L1-normalised histograms.

    Bins are centred so that Sa (0 cents) sits in the middle of bin 0 (bin 0 covers [-w/2, w/2)).
    Masked (NaN) frames are never counted. `smooth` is added to every bin before renormalising,
    which also gives an all-masked window a uniform histogram (callers should skip those via `valid`).
    """
    w = 1200.0 / n_bins
    c = np.asarray(cents_windows, dtype=float)
    ok = np.isfinite(c)
    idx = np.floor(np.mod(np.where(ok, c, 0.0) + w / 2, 1200.0) / w).astype(int) % n_bins
    n = c.shape[0]
    flat = (idx + np.arange(n)[:, None] * n_bins)[ok]
    h = np.bincount(flat, minlength=n * n_bins).reshape(n, n_bins).astype(float)
    s = h.sum(axis=1, keepdims=True)
    h = np.divide(h, s, out=np.zeros_like(h), where=s > 0)  # L1
    h = h + smooth
    return h / h.sum(axis=1, keepdims=True)
