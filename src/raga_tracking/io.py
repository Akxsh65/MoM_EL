"""Loaders for CMD Carnatic features and Saraga Carnatic (formats documented in docs/data_formats.md)."""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CMD = os.path.join(ROOT, "data/raw/cmd/RagaDataset/Carnatic/features")
SAR = os.path.join(ROOT, "data/raw/saraga/saraga1.5_carnatic")


def safe_name(rec_id: str) -> str:
    """Filesystem-safe cache name for a rec_id (':' is illegal on Windows)."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", rec_id)


def read_pitch_file(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (t_s, f_hz) from a 2-column whitespace file. Unvoiced frames are 0 Hz in the source."""
    a = pd.read_csv(path, sep=r"\s+", header=None, dtype=float).to_numpy()
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError(f"{path}: expected 2 columns, got {a.shape}")
    return a[:, 0], a[:, 1]


def read_tonic_file(path: str) -> float:
    """Tonic in Hz (single float)."""
    return float(open(path).read().strip())


def recording_paths() -> dict[str, tuple[str, str]]:
    """rec_id -> (pitch_path, tonic_path), same rec_id scheme as scripts/01_inventory.py."""
    out: dict[str, tuple[str, str]] = {}
    for p in glob.glob(f"{CMD}/*/*/*/*/*.pitch"):
        rel = os.path.relpath(p, CMD).replace("\\", "/").split("/")
        ragaid, artist, concert, item = rel[:4]
        base = p[: -len(".pitch")]
        out[f"cmd:{ragaid[:8]}:{artist}:{concert}:{item}"] = (p, base + ".tonicFine")  # fine-tuned tonic, see decisions.md
    for j in glob.glob(f"{SAR}/*/*/*.json"):
        base = j[: -len(".json")]
        mbid = json.load(open(j, encoding="utf8"))["mbid"]
        if os.path.exists(base + ".pitch.txt") and os.path.exists(base + ".ctonic.txt"):
            out[f"sar:{mbid}"] = (base + ".pitch.txt", base + ".ctonic.txt")
    return out
