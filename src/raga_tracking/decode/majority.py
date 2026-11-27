"""D1: argmax + majority filter (width 5)."""
from __future__ import annotations

import numpy as np

from .argmax import window_labels_d0
from .common import labels_to_segments


def majority_filter(labels: np.ndarray, width: int = 5) -> np.ndarray:
    """Sliding mode; ties keep the centre label if tied, else the smallest tied label.
    Edges use truncated windows."""
    T, h = len(labels), width // 2
    out = labels.copy()
    for t in range(T):
        w = labels[max(0, t - h): t + h + 1]
        vals, cnt = np.unique(w, return_counts=True)
        best = vals[cnt == cnt.max()]
        out[t] = labels[t] if labels[t] in best else best[0]
    return out


def decode_d1(post, valid, centres, hop, width: int = 5):
    """Returns (intervals, labels)."""
    return labels_to_segments(majority_filter(window_labels_d0(post, valid), width), centres, hop)
