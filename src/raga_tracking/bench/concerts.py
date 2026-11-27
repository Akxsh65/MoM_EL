"""B2: real concerts, items concatenated in performance order (T9).

Per test-split Saraga concert: items sorted by MusicBrainz track position, each trimmed to its melodic regions
(Saraga sections minus tani/unlabelled; without sections, leading/trailing unvoiced stretches), then concatenated on the
100 Hz cache. Label = raga if in the set, `other` for out-of-set and ragamalika items. Neighbouring items with the same
label merge into one reference segment (no boundary). Items with no known raga or with an unreliable tonic are dropped.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from ..features.dataset import load_cached
from ..io import ROOT
from .synthetic import FS, melodic_intervals

OTHER = "other"


def trim_unvoiced(voiced: np.ndarray, win_s: float = 10.0, min_frac: float = 0.5) -> list[tuple[float, float]]:
    """Whole item minus leading/trailing stretches where the 10 s rolling voiced fraction is below min_frac."""
    w = int(win_s * FS)
    if len(voiced) <= w:
        return [(0.0, len(voiced) / FS)]
    r = np.convolve(voiced.astype(float), np.ones(w) / w, mode="valid")
    ok = np.flatnonzero(r >= min_frac)
    if len(ok) == 0:
        return [(0.0, len(voiced) / FS)]
    return [(float(ok[0]) / FS, float(ok[-1] + w) / FS)]


def item_regions(rec_id: str, cents: np.ndarray, voiced: np.ndarray, t0: float, has_sections: bool) -> list[tuple[float, float]]:
    if has_sections:
        return melodic_intervals(rec_id, len(cents), t0)
    return trim_unvoiced(voiced)


def test_concert_items() -> dict[str, list[dict]]:
    """concert -> candidate items (performance order) with labels; drops unknown-raga and excluded-tonic items."""
    inv = pd.read_csv(f"{ROOT}/data/processed/inventory.csv", keep_default_na=False)
    sp = pd.read_csv(f"{ROOT}/data/processed/splits.csv", keep_default_na=False)[["rec_id", "in_set", "usable", "split"]]
    d = inv[inv.source == "saraga"].merge(sp, on="rec_id")
    d = d[d.split == "test"].copy()
    d["order"] = d.item_order.astype(float).astype(int)
    out: dict[str, list[dict]] = {}
    dropped: list[dict] = []
    for r in d.sort_values(["concert", "order"]).itertuples():
        rm = bool(r.ragamalika_flag)
        if r.in_set:
            lab = r.raga
        elif rm or (r.raga != ""):
            lab = OTHER  # ragamalika, or a known raga outside the 40-raga set
        else:
            dropped.append({"rec_id": r.rec_id, "concert": r.concert, "order": r.order, "why": "no raga in metadata"})
            continue
        if not r.in_set and not rm and not r.usable:
            dropped.append({"rec_id": r.rec_id, "concert": r.concert, "order": r.order, "why": "excluded recording (tonic)"})
            continue
        out.setdefault(r.concert, []).append({"rec_id": r.rec_id, "order": int(r.order), "label": lab, "ragamalika": rm,
                                              "raga_raw": r.raga_raw, "has_sections": bool(r.has_sections)})
    out["__dropped__"] = dropped  # type: ignore[assignment]
    return out


def render(entry: dict) -> dict:
    """Rebuild a concert from its manifest entry -> cents, voiced, merged reference intervals/labels, item intervals."""
    C, V, items, t = [], [], [], 0.0
    for it in entry["items"]:
        cents, voiced, _ = load_cached(it["rec_id"])
        a_ = []
        for s, e in it["regions_s"]:
            a, b = int(round(s * FS)), int(round(e * FS))
            C.append(cents[a:b]); V.append(voiced[a:b]); a_.append(b - a)
        n = sum(a_) / FS
        items.append((t, t + n, it["label"]))
        t += n
    iv, lab = [], []
    for s, e, l in items:  # merge neighbouring items with the same label
        if lab and lab[-1] == l:
            iv[-1][1] = e
        else:
            iv.append([s, e]); lab.append(l)
    return {"cents": np.concatenate(C).astype(np.float32), "voiced": np.concatenate(V).astype(bool),
            "intervals": np.array(iv), "labels": np.array(lab), "item_intervals": np.array([[s, e] for s, e, _ in items])}


def content_hash(seq: dict) -> str:
    h = hashlib.sha256()
    for k in ("cents", "voiced", "intervals", "item_intervals"):
        h.update(np.ascontiguousarray(seq[k]).tobytes())
    h.update("|".join(seq["labels"]).encode())
    return h.hexdigest()


def n_boundaries(seq: dict) -> int:
    return max(len(seq["labels"]) - 1, 0)


def save_manifest(entries: list[dict], path: str, meta: dict) -> None:
    json.dump({"meta": meta, "concerts": entries}, open(path, "w"), indent=1)
