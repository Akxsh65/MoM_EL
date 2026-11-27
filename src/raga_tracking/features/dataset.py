"""Split-aware access to cached recordings and window sampling. Only the splits you ask for are read."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import yaml

from ..io import ROOT, safe_name
from ..windows import make_windows


def load_cfg() -> dict:
    return yaml.safe_load(open(f"{ROOT}/configs/default.yaml"))


def raga_list() -> list[str]:
    return yaml.safe_load(open(f"{ROOT}/configs/raga_set.yaml"))["ragas"]


def recordings(split: str) -> pd.DataFrame:
    """In-set, non-duplicate, usable recordings of one split (train / val / test)."""
    d = pd.read_csv(f"{ROOT}/data/processed/splits.csv", keep_default_na=False)
    return d[(d.split == split) & d.in_set & ~d.dup].sort_values("rec_id").reset_index(drop=True)


def load_cached(rec_id: str) -> tuple[np.ndarray, np.ndarray, float]:
    """(cents with NaN unvoiced, voiced mask, t0) from the 100 Hz cache."""
    z = np.load(f"{ROOT}/data/interim/pitch/{safe_name(rec_id)}.npz")
    return z["cents"].astype(float), z["voiced"], float(z["t0"])


def sample_windows(rec_id: str, L: float, hop: float, max_windows: int | None, rng: np.random.Generator,
                   min_voiced: float = 0.4) -> dict:
    """Valid windows of a recording, at most `max_windows` chosen uniformly at random (seeded)."""
    cents, voiced, t0 = load_cached(rec_id)
    w = make_windows(cents, voiced, 100.0, L, hop, t0, min_voiced)
    keep = np.flatnonzero(w["valid"])
    if max_windows is not None and len(keep) > max_windows:
        keep = np.sort(rng.choice(keep, max_windows, replace=False))
    return {"cents": w["cents"][keep], "centres": w["centres"][keep], "idx": keep}
