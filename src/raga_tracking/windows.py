"""Sliding windows over cached cents/voiced arrays."""
from __future__ import annotations

import numpy as np


def n_windows(n_frames: int, fs: float, L: float, hop: float) -> int:
    """floor((dur - L)/hop) + 1, or 0 if the recording is shorter than L."""
    dur = n_frames / fs
    if dur < L:
        return 0
    return int(np.floor((dur - L) / hop + 1e-9)) + 1


def make_windows(cents: np.ndarray, voiced: np.ndarray, fs: float, L: float,
                 hop: float, t0: float = 0.0, min_voiced: float = 0.4) -> dict:
    """Return window cents (n, L*fs) with unvoiced frames NaN, masks, centre times,
    voiced fractions and a validity flag (voiced fraction >= min_voiced)."""
    wl, wh = int(round(L * fs)), int(round(hop * fs))
    n = n_windows(len(cents), fs, L, hop)
    starts = np.arange(n) * wh
    v = np.asarray(voiced, dtype=bool)
    c = np.where(v, cents, np.nan)
    idx = starts[:, None] + np.arange(wl)[None, :]
    wc, wv = c[idx], v[idx]
    frac = wv.mean(axis=1) if n else np.zeros(0)
    centres = t0 + (starts + wl / 2) / fs
    return {"cents": wc, "voiced": wv, "centres": centres,
            "voiced_frac": frac, "valid": frac >= min_voiced}
