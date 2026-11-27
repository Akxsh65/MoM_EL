"""Segmentation metrics shared by tuning (T10) and the main evaluation (T11).

Boundaries are the interior segment edges. Matching uses mir_eval's one-to-one window matching, the same routine
mir_eval.segment.detection(trim=True) uses (tests/test_metrics.py asserts the two agree), but returns raw counts so
results can be pooled and bootstrapped over sequences.
"""
from __future__ import annotations

import numpy as np
from mir_eval.util import match_events


def boundaries(intervals: np.ndarray) -> np.ndarray:
    """Interior boundaries (s): the end of every segment except the last."""
    iv = np.asarray(intervals, float).reshape(-1, 2)
    return iv[:-1, 1] if len(iv) > 1 else np.zeros(0)


def boundary_counts(ref_iv: np.ndarray, est_iv: np.ndarray, window: float) -> tuple[int, int, int]:
    """(true positives, n_estimated, n_reference) within +-window seconds."""
    r, e = boundaries(ref_iv), boundaries(est_iv)
    tp = len(match_events(r, e, window)) if len(r) and len(e) else 0
    return tp, len(e), len(r)


def pooled_prf(counts: list[tuple[int, int, int]]) -> tuple[float, float, float]:
    """Micro-pooled precision, recall, F over sequences."""
    tp, ne, nr = (float(sum(c[i] for c in counts)) for i in range(3))
    p, r = (tp / ne if ne else 0.0), (tp / nr if nr else 0.0)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def label_at(intervals: np.ndarray, labels, t: np.ndarray) -> np.ndarray:
    """Label of the segment covering each time t (clamped to first/last segment)."""
    iv = np.asarray(intervals, float).reshape(-1, 2)
    idx = np.clip(np.searchsorted(iv[:, 0], t, side="right") - 1, 0, len(iv) - 1)
    return np.asarray(labels)[idx]


def frame_accuracy(ref_iv, ref_lab, est_iv, est_lab, step: float = 1.0, skip: tuple = ("other",)) -> tuple[int, int]:
    """(correct, total) on a `step`-second grid over the reference span, ignoring reference frames labelled in `skip`.
    Estimated labels must be raga names (strings), like the reference."""
    t = np.arange(ref_iv[0][0] + step / 2, ref_iv[-1][1], step)
    r, e = label_at(ref_iv, ref_lab, t), label_at(est_iv, est_lab, t)
    keep = ~np.isin(r, skip)
    return int((r[keep] == e[keep]).sum()), int(keep.sum())


def false_switches_per_10min(ref_iv, est_iv, window: float, duration_s: float) -> float:
    """Estimated boundaries with no reference boundary within +-window, per 10 minutes."""
    tp, ne, _ = boundary_counts(ref_iv, est_iv, window)
    return (ne - tp) / (duration_s / 600.0)
