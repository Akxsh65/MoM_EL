"""T2: choose the raga set and artist-disjoint train/val/test splits (seeded greedy).

Command: python scripts/02_splits.py
Writes: configs/raga_set.yaml, data/processed/splits.csv, results/splits_summary.md
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = yaml.safe_load(open(f"{ROOT}/configs/default.yaml"))["splits"]


def norm(s: str) -> str:
    return "".join(c for c in str(s).lower() if c.isalnum())


def load() -> pd.DataFrame:
    d = pd.read_csv(f"{ROOT}/data/processed/inventory.csv", keep_default_na=False)
    groups = yaml.safe_load(open(f"{ROOT}/configs/artist_groups.yaml")) or {}
    gm = {norm(k): norm(v) for k, v in groups.items()}
    d["group"] = d.artist_key.map(lambda k: gm.get(k, k))
    d["usable"] = d.has_pitch & d.has_tonic & (d.raga != "")
    d["dup"] = False
    # CMD copy of a recording that also exists in Saraga is the duplicate
    sar = set(d[(d.source == "saraga") & (d.mbid != "")].mbid)
    d.loc[(d.source == "cmd") & d.mbid.isin(sar), "dup"] = True
    return d


def choose_ragas(d: pd.DataFrame, n_rec: int, n_art: int) -> list[str]:
    u = d[d.usable & ~d.dup]
    g = u.groupby("raga").agg(n=("rec_id", "size"), a=("group", "nunique"))
    return sorted(g[(g.n >= n_rec) & (g.a >= n_art)].index)


def assign(d: pd.DataFrame, ragas: list[str], seed: int) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    u = d[d.usable & ~d.dup & d.raga.isin(ragas)]
    total = len(u)
    art_raga = u.groupby(["group", "raga"]).size().unstack(fill_value=0)
    n_art_raga = (art_raga > 0).sum(axis=0)  # artists per raga
    taken = {r: 0 for r in ragas}       # artists removed from train per raga
    split: dict[str, str] = {}
    cover = {"val": set(), "test": set()}
    size = {"val": 0, "test": 0}
    target = {"test": CFG["target_test_frac"] * total, "val": CFG["target_val_frac"] * total}
    # Saraga concerts with enough in-set items, keyed by artist group
    sar = u[u.source == "saraga"].groupby(["group", "concert"]).size()
    good = {g for (g, c), n in sar.items() if n >= CFG["min_inset_items_per_concert"]}
    n_conc = {"val": 0, "test": 0}
    need = {"test": CFG["min_test_concerts"], "val": CFG["min_val_concerts"]}

    def safe(g: str) -> bool:
        rs = art_raga.columns[art_raga.loc[g] > 0]
        return all(n_art_raga[r] - taken[r] - 1 >= 1 for r in rs)

    def take(g: str, s: str) -> None:
        split[g] = s
        rs = art_raga.columns[art_raga.loc[g] > 0]
        for r in rs:
            taken[r] += 1
        cover[s] |= set(rs)
        size[s] += int(art_raga.loc[g].sum())
        n_conc[s] += int(sum(1 for (gg, c), n in sar.items() if gg == g and n >= CFG["min_inset_items_per_concert"]))

    def gain(g: str, s: str) -> int:
        return len(set(art_raga.columns[art_raga.loc[g] > 0]) - cover[s])

    for s in ("test", "val"):
        # phase A: Saraga artists with good concerts, until enough concerts
        while n_conc[s] < need[s]:
            cands = [g for g in good if g not in split and safe(g)]
            if not cands:
                break
            sc = [(gain(g, s), rng.random()) for g in cands]
            take(cands[max(range(len(cands)), key=lambda i: sc[i])], s)
        # phase B: cover remaining ragas / reach size target, prefer artists with no Saraga concerts
        while size[s] < 1.25 * target[s]:  # may overshoot the target only to cover a missing raga
            cands = [g for g in art_raga.index if g not in split and safe(g)]
            if not cands:
                break
            sc = [(gain(g, s), g not in good, rng.random()) for g in cands]
            best = max(range(len(cands)), key=lambda i: sc[i])
            if size[s] >= target[s] and sc[best][0] == 0:
                break
            take(cands[best], s)
    return split


def main() -> None:
    d = load()
    n_rec, n_art = CFG["min_recordings"], CFG["min_artists"]
    ragas = choose_ragas(d, n_rec, n_art)
    note = "strict thresholds"
    if len(ragas) < 15:
        ragas, note = choose_ragas(d, 6, 2), "relaxed to >=6 recs / >=2 artists (T2 backup 1)"
    split = assign(d, ragas, CFG["seed"])
    d["split"] = d.group.map(split).fillna("train")
    d["in_set"] = d.raga.isin(ragas) & d.usable
    cols = ["rec_id", "source", "artist", "group", "concert", "item_order", "raga", "in_set", "usable", "dup", "split"]
    d[cols].to_csv(f"{ROOT}/data/processed/splits.csv", index=False)
    yaml.safe_dump({"K": len(ragas), "thresholds": note, "seed": CFG["seed"], "ragas": ragas},
                   open(f"{ROOT}/configs/raga_set.yaml", "w"), sort_keys=False)
    report(d, ragas, note)


def report(d: pd.DataFrame, ragas: list[str], note: str) -> None:
    u = d[d.in_set & ~d.dup]
    L = ["# Splits summary", "", "Produced by: `python scripts/02_splits.py`", "",
         f"K = {len(ragas)} ({note}). Artist groups: {d.group.nunique()}.", "",
         "| split | artists | in-set recs | frac | ragas covered | saraga recs | test-style concerts (>=3 in-set) |", "|---|---|---|---|---|---|---|"]
    for s in ("train", "val", "test"):
        x = u[u.split == s]
        sc = x[x.source == "saraga"].groupby("concert").size()
        L.append(f"| {s} | {x.group.nunique()} | {len(x)} | {len(x)/len(u):.1%} | {x.raga.nunique()}/{len(ragas)} | "
                 f"{int((x.source == 'saraga').sum())} | {int((sc >= 3).sum())} |")
    L += ["", "Ragas missing from a split (cannot be covered given artist counts):"]
    for s in ("val", "test"):
        miss = sorted(set(ragas) - set(u[u.split == s].raga))
        L.append(f"- {s}: {miss}")
    miss = sorted(set(ragas) - set(u[u.split == "train"].raga))
    L.append(f"- train: {miss}")
    open(f"{ROOT}/results/splits_summary.md", "w", encoding="utf8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
