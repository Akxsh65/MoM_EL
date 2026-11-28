"""Temperature scaling and expected calibration error (T7)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature T > 0 minimising the val negative log-likelihood of softmax(logits / T)."""
    def nll(logT: float) -> float:
        return -log_softmax(logits / np.exp(logT), axis=1)[np.arange(len(y)), y].mean()
    r = minimize_scalar(nll, bounds=(-3.0, 4.0), method="bounded")
    return float(np.exp(r.x))


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """Calibrated posteriors softmax(logits / T)."""
    return softmax(logits / T, axis=1)


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error on the max-probability prediction, equal-width bins."""
    conf, pred = probs.max(1), probs.argmax(1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    tot = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(tot)
