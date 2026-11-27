"""B1: synthetic same-concert splices (T8).

A sequence is 2-5 excerpts (60-240 s) cut from different in-set items of ONE Saraga concert, each item in a distinct raga,
concatenated on the 100 Hz cents/voiced cache with re-based time. Everything needed to rebuild a sequence lives in the
manifest entry (source rec_ids, cut times in seconds of cache time, label), so `render(entry)` is deterministic.
Only val and test concerts are used (never train).
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os

import numpy as np
import pandas as pd
import yaml

from ..features.dataset import load_cached
from ..io import ROOT, recording_paths

FS = 100
NON_MELODIC = ("tani", "-")  # section labels never used for excerpts (percussion solo, unlabelled)
EASY_MAX, HARD_MIN = 4, 6    # shared svaras between neighbouring ragas (CLAUDE.md T8; backup 1: HARD_MIN = 5)


def load_scales() -> dict[str, set[int]]:
    s = yaml.safe_load(open(f"{ROOT}/configs/scales.yaml"))
    return {k: set(v["svaras"]) for k, v in s.items() if isinstance(v, dict) and "svaras" in v}


def shared_svaras(a: str, b: str, scales: dict[str, set[int]]) -> int:
    return len(scales[a] & scales[b])


def label_sequence(ragas: list[str], scales: dict[str, set[int]], hard_min: int = HARD_MIN) -> tuple[str, list[int]]:
    """'easy' (all neighbouring pairs share <= 4), 'hard' (any pair shares >= hard_min), else 'mid'."""
    sh = [shared_svaras(a, b, scales) for a, b in zip(ragas[:-1], ragas[1:])]
    if max(sh) >= hard_min:
        return "hard", sh
    return ("easy" if max(sh) <= EASY_MAX else "mid"), sh


def sections_path(rec_id: str) -> str | None:
    pitch, _ = recording_paths()[rec_id]
    p = pitch[: -len(".pitch.txt")] + ".sections-manual.txt"
    return p if os.path.exists(p) else None


def melodic_intervals(rec_id: str, n_frames: int, t0: float) -> list[tuple[float, float]]:
    """Allowed excerpt regions in cache time (s): merged melodic sections, or the whole item if none are annotated."""
    sp = sections_path(rec_id)
    full = [(0.0, n_frames / FS)]
    if sp is None:
        return full
    iv = []
    for line in open(sp, encoding="utf8"):
        p = line.rstrip("\n").split("\t")
        if len(p) < 4:
            continue
        s, d, lab = float(p[0]) - t0, float(p[2]), p[3].strip().lower()
        if lab.startswith(NON_MELODIC):
            continue
        iv.append((max(s, 0.0), min(s + d, n_frames / FS)))
    iv.sort()
    merged: list[list[float]] = []
    for s, e in iv:
        if merged and s <= merged[-1][1] + 0.5:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged if e > s] or full


def pick_excerpt(rng: np.random.Generator, cents: np.ndarray, voiced: np.ndarray, ivs: list[tuple[float, float]],
                 min_len: float, max_len: float, min_voiced: float, tries: int = 25) -> tuple[float, float] | None:
    """Random (start, end) inside one allowed interval, length in [min_len, max_len], voiced fraction >= min_voiced."""
    ok = [(s, e) for s, e in ivs if e - s >= min_len]
    if not ok:
        return None
    for _ in range(tries):
        s0, e0 = ok[rng.integers(len(ok))]
        d = round(float(rng.uniform(min_len, min(max_len, e0 - s0))), 2)
        st = round(float(rng.uniform(s0, e0 - d)), 2)
        a, b = int(round(st * FS)), int(round((st + d) * FS))
        if voiced[a:b].mean() >= min_voiced:
            return st, round(st + d, 2)
    return None


def render(entry: dict) -> dict:
    """Rebuild a sequence from its manifest entry: concatenated cents/voiced and reference segments."""
    C, V, segs, t = [], [], [], 0.0
    for e in entry["excerpts"]:
        cents, voiced, _ = load_cached(e["rec_id"])
        a, b = int(round(e["start_s"] * FS)), int(round(e["end_s"] * FS))
        C.append(cents[a:b]); V.append(voiced[a:b])
        segs.append({"start": t, "end": t + (b - a) / FS, "raga": e["raga"]})
        t += (b - a) / FS
    cents = np.concatenate(C).astype(np.float32)
    voiced = np.concatenate(V).astype(bool)
    return {"cents": cents, "voiced": voiced, "intervals": np.array([[s["start"], s["end"]] for s in segs]),
            "labels": np.array([s["raga"] for s in segs])}


def content_hash(seq: dict) -> str:
    """sha256 over the array bytes. (np.savez embeds zip timestamps, so the data, not the file, is hashed.)"""
    h = hashlib.sha256()
    for k in ("cents", "voiced", "intervals"):
        h.update(np.ascontiguousarray(seq[k]).tobytes())
    h.update("|".join(seq["labels"]).encode())
    return h.hexdigest()


def concert_pool(split: str) -> dict[str, list[dict]]:
    """concert -> usable in-set Saraga items of this split that have a cached pitch track."""
    assert split in ("val", "test")  # train concerts are never spliced
    d = pd.read_csv(f"{ROOT}/data/processed/splits.csv", keep_default_na=False)
    d = d[(d.split == split) & d.in_set & d.usable & ~d.dup & (d.source == "saraga")]
    pool: dict[str, list[dict]] = {}
    for r in d.itertuples():
        cents, voiced, t0 = load_cached(r.rec_id)
        pool.setdefault(r.concert, []).append({
            "rec_id": r.rec_id, "raga": r.raga, "artist": r.artist, "group": r.group,
            "cents": cents, "voiced": voiced, "ivs": melodic_intervals(r.rec_id, len(cents), t0)})
    return {c: v for c, v in pool.items() if len({i["raga"] for i in v}) >= 2}


def build_manifest(split: str, quotas: dict[str, int], seed: int, min_len: float = 60.0, max_len: float = 240.0,
                   min_voiced: float = 0.6, hard_min: int = HARD_MIN, max_attempts: int = 200000) -> list[dict]:
    """Draw sequences until each label ('easy', 'hard', 'mid') reaches its quota (or attempts run out)."""
    rng = np.random.default_rng(seed)
    scales, pool = load_scales(), concert_pool(split)
    concerts = sorted(pool)
    need, out, seen = dict(quotas), [], set()
    for _ in range(max_attempts):
        if all(v <= 0 for v in need.values()):
            break
        c = concerts[rng.integers(len(concerts))]
        items = pool[c]
        ragas = sorted({i["raga"] for i in items})
        n = int(rng.integers(2, 6))
        if len(ragas) < n:
            continue
        chosen = [ragas[j] for j in rng.permutation(len(ragas))[:n]]
        exc = []
        for r in chosen:
            cand = [i for i in items if i["raga"] == r]
            it = cand[rng.integers(len(cand))]
            span = pick_excerpt(rng, it["cents"], it["voiced"], it["ivs"], min_len, max_len, min_voiced)
            if span is None:
                break
            exc.append({"rec_id": it["rec_id"], "raga": r, "start_s": span[0], "end_s": span[1], "artist_group": it["group"]})
        if len(exc) < n:
            continue
        lab, sh = label_sequence(chosen, scales, hard_min)
        key = tuple((e["rec_id"], e["start_s"], e["end_s"]) for e in exc)
        if need.get(lab, 0) <= 0 or key in seen:
            continue
        seen.add(key); need[lab] -= 1
        out.append({"split": split, "concert": c, "label": lab, "shared_svaras": sh, "excerpts": exc})
    for i, e in enumerate(out):
        e["seq_id"] = f"b1_{split}_{i:04d}"
    return out


def save_manifest(entries: list[dict], path: str, meta: dict) -> None:
    json.dump({"meta": meta, "sequences": entries}, open(path, "w"), indent=1)
