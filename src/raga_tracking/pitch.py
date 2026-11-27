"""Hz -> cents conversion, resampling to a uniform grid, voiced mask."""
from __future__ import annotations

import numpy as np

FS = 100.0  # cached grid rate (Hz)


def hz_to_cents(f_hz: np.ndarray, tonic_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (cents, voiced). Frames with f <= 0 (or NaN) are unvoiced; cents there are NaN."""
    f = np.asarray(f_hz, dtype=float)
    voiced = np.isfinite(f) & (f > 0)
    cents = np.full(f.shape, np.nan)
    cents[voiced] = 1200.0 * np.log2(f[voiced] / tonic_hz)
    return cents, voiced


def pitch_class(cents: np.ndarray) -> np.ndarray:
    """Cents modulo one octave, in [0, 1200)."""
    return np.mod(cents, 1200.0)


def resample_to_grid(t: np.ndarray, cents: np.ndarray, voiced: np.ndarray,
                     fs: float = FS) -> dict:
    """Nearest-neighbour resample onto a uniform grid starting at t[0].

    A grid frame is voiced only if its nearest source frame is voiced and lies
    within one source hop of the grid time (so gaps in the source stay unvoiced).
    """
    t = np.asarray(t, dtype=float)
    if len(t) < 2:
        raise ValueError("need at least 2 pitch frames")
    hop = float(np.median(np.diff(t)))
    t0 = float(t[0])
    n = int(np.floor((t[-1] - t0) * fs)) + 1
    grid = t0 + np.arange(n) / fs
    idx = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
    idx = np.where(np.abs(t[idx - 1] - grid) <= np.abs(t[idx] - grid), idx - 1, idx)
    near = np.abs(t[idx] - grid) <= hop
    v = np.asarray(voiced, dtype=bool)[idx] & near
    c = np.where(v, np.asarray(cents)[idx], np.nan)
    return {"cents": c, "voiced": v, "t0": t0, "fs": float(fs)}
