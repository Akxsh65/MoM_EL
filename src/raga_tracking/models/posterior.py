"""Window posteriors of a trained W1/W2 model on a pitch sequence (no temperature: T7 backup, beta lives in D3)."""
from __future__ import annotations

import joblib
import numpy as np
from scipy.special import softmax

from ..features.histogram import window_histograms
from ..features.tdms import tdms_windows
from ..io import ROOT
from ..windows import make_windows


def load_model(name: str) -> dict:
    return joblib.load(f"{ROOT}/models/{name}.joblib")


def _features(m: dict, name: str, cents: np.ndarray) -> np.ndarray:
    if name.startswith("w1"):
        return window_histograms(cents, m["n_bins"])
    return tdms_windows(cents, m["tau_s"], m["n_bins"], m["alpha"], m["sigma"]) * (m["n_bins"] ** 2)


def sequence_posteriors(m: dict, name: str, cents: np.ndarray, voiced: np.ndarray, min_voiced: float = 0.4,
                        hist_bins: int = 12) -> dict:
    """post (T,K) softmax posteriors, valid (T,), centres (T,) seconds, hists (T,B) 12-bin W1 histograms (for D2)."""
    w = make_windows(cents, voiced, 100.0, m["L"], m["hop"], 0.0, min_voiced)
    post = softmax(m["clf"].decision_function(_features(m, name, w["cents"])), axis=1)
    return {"post": post.astype(np.float32), "valid": w["valid"], "centres": w["centres"],
            "hists": window_histograms(w["cents"], hist_bins).astype(np.float32)}
