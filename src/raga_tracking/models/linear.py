"""Shared helpers for the linear window classifiers (W1/W2): window sampling per split and metrics."""
from __future__ import annotations

import subprocess

import numpy as np

from ..features.dataset import recordings, sample_windows
from ..io import ROOT

SPLITS_USED = ("train", "val")  # training scripts never touch "test" (CLAUDE.md rule 2)


def sample_split(split: str, ragas: list[str], L: float, hop: float, cap: int, seed: int):
    """Sampled valid windows (float32 cents) with labels and recording ids for one of train/val."""
    assert split in SPLITS_USED
    rng = np.random.default_rng(seed)
    C, y, rid = [], [], []
    for _, r in recordings(split).iterrows():
        w = sample_windows(r.rec_id, L, hop, cap, rng)
        if len(w["idx"]) == 0:
            continue
        C.append(w["cents"].astype(np.float32))
        y += [ragas.index(r.raga)] * len(w["idx"])
        rid += [r.rec_id] * len(w["idx"])
    return np.concatenate(C), np.array(y), np.array(rid)


def majority_by_recording(pred: np.ndarray, y: np.ndarray, rid: np.ndarray) -> float:
    """Fraction of recordings whose most-predicted window label is the true raga."""
    return float(np.mean([np.bincount(pred[rid == r]).argmax() == y[rid == r][0] for r in np.unique(rid)]))


def git_hash() -> str:
    try:
        h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
        return h + ("+uncommitted" if dirty else "")
    except Exception:
        return "unknown"
