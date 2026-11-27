"""D2: label-free novelty change points from W1 histograms, labelled by mean posterior."""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def sym_kl(p: np.ndarray, q: np.ndarray) -> float:
    """Symmetric KL between two histograms (1e-6 smoothing, renormalised)."""
    p = (p + 1e-6) / (p + 1e-6).sum()
    q = (q + 1e-6) / (q + 1e-6).sum()
    return float(np.sum((p - q) * (np.log(p) - np.log(q))))


def novelty_curve(hists: np.ndarray, k: int = 3) -> np.ndarray:
    """Novelty at point t (between windows t-1 and t): sKL(mean hist[t-k:t], mean hist[t:t+k]).
    Points lacking k windows on a side are 0."""
    T = len(hists)
    nov = np.zeros(T)
    for t in range(k, T - k + 1):
        nov[t] = sym_kl(hists[t - k:t].mean(0), hists[t:t + k].mean(0))
    return nov


def decode_d2(hists, post, valid, centres, hop, k: int = 3, thr: float = 0.1):
    """hists (T,B) per-window W1 histograms, post (T,K). Peaks above thr split the sequence;
    each segment is labelled by its mean posterior over valid windows. Returns (intervals, labels)."""
    T = len(post)
    valid = np.asarray(valid, bool)
    nov = novelty_curve(np.asarray(hists), k)
    peaks, _ = find_peaks(nov, height=thr, distance=max(1, k))
    cuts = np.concatenate([[0], peaks, [T]]).astype(int)
    ivs, labs = [], []
    for a, b in zip(cuts[:-1], cuts[1:]):
        v = valid[a:b]
        seg = post[a:b][v] if v.any() else post[a:b]
        ivs.append([centres[a] - hop / 2, centres[b - 1] + hop / 2])
        labs.append(int(seg.mean(0).argmax()))
    return np.array(ivs), np.array(labs)
