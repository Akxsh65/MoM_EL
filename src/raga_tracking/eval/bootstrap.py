"""Bootstrap confidence intervals over evaluation units (sequences, or concerts for B2).

Every unit contributes a vector of raw counts COLS; every reported metric is a ratio of sums of those counts, so a
resample is just a sum of the drawn units' count vectors. Paired comparisons reuse the same draws for both systems.
"""
from __future__ import annotations

import numpy as np

COLS = ("tp5", "ne5", "nr5", "tp10", "ne10", "nr10", "fa_c", "fa_t", "seg_c", "seg_t", "dur_s")
METRICS = ("P5", "R5", "F5", "F10", "frame_acc", "seg_acc", "false_sw_10min")


def metrics_from_sums(s: np.ndarray) -> dict[str, np.ndarray]:
    """s: (..., len(COLS)) summed counts -> metric arrays (NaN where a denominator is 0)."""
    c = {k: s[..., i] for i, k in enumerate(COLS)}
    div = lambda a, b: np.divide(a, b, out=np.full(np.shape(a), np.nan), where=np.asarray(b) > 0)  # noqa: E731
    p5, r5 = div(c["tp5"], c["ne5"]), div(c["tp5"], c["nr5"])
    p10, r10 = div(c["tp10"], c["ne10"]), div(c["tp10"], c["nr10"])
    f = lambda p, r: np.where((p + r) > 0, 2 * p * r / np.where((p + r) > 0, p + r, 1), 0.0)  # noqa: E731
    return {"P5": p5, "R5": r5, "F5": f(np.nan_to_num(p5), np.nan_to_num(r5)), "F10": f(np.nan_to_num(p10), np.nan_to_num(r10)),
            "frame_acc": div(c["fa_c"], c["fa_t"]), "seg_acc": div(c["seg_c"], c["seg_t"]),
            "false_sw_10min": div(c["ne5"] - c["tp5"], c["dur_s"] / 600.0)}


def draws(n_units: int, n_boot: int, rng: np.random.Generator, clusters: np.ndarray | None = None) -> np.ndarray:
    """(n_boot, n_units) multiplicity matrix. With `clusters`, whole clusters (e.g. concerts) are resampled."""
    if clusters is None:
        idx = rng.integers(0, n_units, size=(n_boot, n_units))
        return np.stack([np.bincount(r, minlength=n_units) for r in idx])
    ids = np.unique(clusters)
    pick = rng.integers(0, len(ids), size=(n_boot, len(ids)))
    cm = np.stack([np.bincount(r, minlength=len(ids)) for r in pick])          # (n_boot, n_clusters)
    return cm[:, np.searchsorted(ids, clusters)]                               # same multiplicity for every unit in a cluster


def ci_table(counts: np.ndarray, mult: np.ndarray, level: float = 0.95) -> dict[str, tuple[float, float, float]]:
    """point estimate and percentile CI for every metric. counts (n_units, d); mult (n_boot, n_units)."""
    point = metrics_from_sums(counts.sum(0))
    boot = metrics_from_sums(mult @ counts)
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    return {m: (float(point[m]), float(np.nanpercentile(boot[m], lo)), float(np.nanpercentile(boot[m], hi))) for m in METRICS}


def paired_diff(counts_a: np.ndarray, counts_b: np.ndarray, mult: np.ndarray, metric: str = "F5",
                level: float = 0.95) -> dict[str, float]:
    """A minus B on the same resamples. Returns difference, CI, and the share of resamples with A <= B."""
    d0 = metrics_from_sums(counts_a.sum(0))[metric] - metrics_from_sums(counts_b.sum(0))[metric]
    d = metrics_from_sums(mult @ counts_a)[metric] - metrics_from_sums(mult @ counts_b)[metric]
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    return {"diff": float(d0), "lo": float(np.nanpercentile(d, lo)), "hi": float(np.nanpercentile(d, hi)),
            "p_a_not_better": float(np.nanmean(d <= 0))}
