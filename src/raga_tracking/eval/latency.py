"""Online detection latency (CLAUDE.md section 5).

For every reference boundary b whose new label is a real raga (not `other`): latency is the time the online
prediction first equals the new raga for `run` consecutive hops, minus b. A hop's prediction is only available when
its window has ended (centre + L/2), so `avail` is the window END time; using centres would hide the cost of long windows.
The first hop of the run must become available in [b, min(b + max_s, next reference boundary)); otherwise the
boundary is a miss (NaN). Misses are returned, never dropped.
"""
from __future__ import annotations

import numpy as np


def detection_latencies(ref_iv: np.ndarray, ref_lab, pred_names: np.ndarray, avail: np.ndarray,
                        max_s: float = 60.0, run: int = 2, skip: tuple = ("other",)) -> np.ndarray:
    """Latency (s) per qualifying reference boundary, NaN where missed. pred_names/avail are per window, time-ordered."""
    ref_iv = np.asarray(ref_iv, float).reshape(-1, 2)
    pred_names, avail = np.asarray(pred_names), np.asarray(avail, float)
    out = []
    for i in range(len(ref_iv) - 1):
        new, b = ref_lab[i + 1], ref_iv[i, 1]
        if new in skip:
            continue
        hi = min(b + max_s, ref_iv[i + 1, 1])
        hit = pred_names == new
        lat = np.nan
        for t in np.flatnonzero((avail >= b) & (avail < hi)):
            if t + run - 1 < len(hit) and hit[t:t + run].all():
                lat = float(avail[t] - b)
                break
        out.append(lat)
    return np.asarray(out, float)


def summarize(lat: np.ndarray) -> dict[str, float]:
    """Median and IQR over detected boundaries, miss rate over all."""
    ok = lat[~np.isnan(lat)]
    q = np.percentile(ok, [25, 50, 75]) if len(ok) else [np.nan] * 3
    return {"n": int(len(lat)), "miss_rate": float(np.isnan(lat).mean()) if len(lat) else np.nan,
            "median_s": float(q[1]), "q25_s": float(q[0]), "q75_s": float(q[2])}
