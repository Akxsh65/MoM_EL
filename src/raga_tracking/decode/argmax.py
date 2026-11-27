"""D0: per-window argmax."""
from __future__ import annotations

import numpy as np

from .common import ffill_labels, labels_to_segments


def window_labels_d0(post: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per-window argmax; silent windows carry the previous label."""
    return ffill_labels(post.argmax(1), valid)


def decode_d0(post, valid, centres, hop):
    """post (T,K) posteriors; returns (intervals, labels)."""
    return labels_to_segments(window_labels_d0(post, valid), centres, hop)
