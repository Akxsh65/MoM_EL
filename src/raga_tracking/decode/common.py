"""Shared helpers: window labels -> (intervals, labels) segments."""
from __future__ import annotations

import numpy as np


def ffill_labels(labels: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Carry the last valid label through silent windows (leading silence takes the first valid one)."""
    out = labels.copy()
    valid = np.asarray(valid, bool)
    if not valid.any():
        return out
    last = labels[np.argmax(valid)]
    for t in range(len(out)):
        if valid[t]:
            last = labels[t]
        out[t] = last
    return out


def labels_to_segments(labels: np.ndarray, centres: np.ndarray, hop: float):
    """Run-length encode per-window labels. Window t covers [c_t - hop/2, c_t + hop/2].
    Returns (intervals (n,2) seconds, labels (n,))."""
    labels = np.asarray(labels)
    if len(labels) == 0:
        return np.zeros((0, 2)), np.zeros(0, dtype=int)
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate([[0], change]).astype(int)
    ends = np.concatenate([change, [len(labels)]]).astype(int)
    t_lo = centres[starts] - hop / 2
    t_hi = centres[ends - 1] + hop / 2
    return np.stack([t_lo, t_hi], axis=1), labels[starts]
