"""T9: build the B2 concert benchmark (test-split Saraga concerts in performance order).

Command: python scripts/09_build_b2.py
Writes: data/processed/b2/{concert}.npz (gitignored), manifests/b2_manifest.json, results/b2_summary.md,
        results/figures/t9_timelines.png. Test split only; no model involved.
Two ragamalika items lack a pitch cache (the T3 cache covered raga-labelled recordings), so they are cached here.
"""
from __future__ import annotations

import os
import re
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/src")
sys.path.insert(0, f"{ROOT}/scripts")
from raga_tracking.bench.concerts import (OTHER, content_hash, item_regions, n_boundaries, render,  # noqa: E402
                                          save_manifest, test_concert_items)
from raga_tracking.features.dataset import load_cached  # noqa: E402
from raga_tracking.io import recording_paths, safe_name  # noqa: E402
from raga_tracking.models.linear import git_hash  # noqa: E402

import importlib  # noqa: E402

c3 = importlib.import_module("03_cache_pitch")


def ensure_cached(rec_id: str) -> None:
    if not os.path.exists(f"{c3.CACHE}/{safe_name(rec_id)}.npz"):
        c3.cache_one(rec_id, *recording_paths()[rec_id])


def main() -> None:
    items = test_concert_items()
    dropped = items.pop("__dropped__")
    os.makedirs(f"{ROOT}/data/processed/b2", exist_ok=True)
    os.makedirs(f"{ROOT}/manifests", exist_ok=True)
    entries, rows, seqs = [], [], {}
    for concert in sorted(items):
        its = items[concert]
        for it in its:
            ensure_cached(it["rec_id"])
            if it["ragamalika"] and c3.pc_stats(it["rec_id"])["sa_mass"] < c3.SA_MIN:
                it["drop"] = True  # unreliable tonic
        its = [i for i in its if not i.get("drop")]
        for it in its:
            cents, voiced, t0 = load_cached(it["rec_id"])
            it["regions_s"] = [[round(s, 2), round(e, 2)] for s, e in item_regions(it["rec_id"], cents, voiced, t0, it["has_sections"])]
        e = {"concert": concert, "items": [{k: i[k] for k in ("rec_id", "order", "label", "ragamalika", "has_sections", "regions_s")} for i in its]}
        s = render(e)
        e["content_sha256"], e["n_boundaries"] = content_hash(s), n_boundaries(s)
        e["duration_s"] = float(len(s["cents"]) / 100)
        e["reference"] = [{"start": round(float(a), 2), "end": round(float(b), 2), "label": str(l)} for (a, b), l in zip(s["intervals"], s["labels"])]
        np.savez_compressed(f"{ROOT}/data/processed/b2/{re.sub(r'[^A-Za-z0-9]+', '_', concert)}.npz", **s)
        entries.append(e); seqs[concert] = s
        rows.append((concert, len(its), len(s["labels"]), e["n_boundaries"], sum(l != OTHER for l in s["labels"]), e["duration_s"] / 60,
                     min(b - a for a, b in s["intervals"])))
        print(rows[-1], flush=True)
    meta = {"_header": f"Produced by: python scripts/09_build_b2.py | commit {git_hash()}", "split": "test",
            "dropped_items": dropped, "order_source": "MusicBrainz release track position (docs/decisions.md)",
            "total_boundaries": sum(e["n_boundaries"] for e in entries)}
    save_manifest(entries, f"{ROOT}/manifests/b2_manifest.json", meta)
    L = ["# B2 summary (T9)", "", f"Produced by: `python scripts/09_build_b2.py` | commit {git_hash()} | test split only", "",
         "| concert | items kept | reference segments | boundaries | in-set segments | duration (min) | shortest segment (s) |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]:.1f} | {r[6]:.0f} |")
    L += ["", f"Concerts: {len(rows)}. Total boundaries: {meta['total_boundaries']}. `other` = out-of-set or ragamalika item (excluded from frame accuracy).",
          f"Dropped items ({len(dropped)}): " + "; ".join(f"{d['concert']} #{d['order']} ({d['why']})" for d in dropped)]
    open(f"{ROOT}/results/b2_summary.md", "w", encoding="utf8").write("\n".join(L) + "\n")
    # timelines for two concerts
    cols = {}
    fig, axs = plt.subplots(2, 1, figsize=(11, 3.6), sharex=False)
    for ax, concert in zip(axs, ("Sanjay at Sastri Hall", "Live at Kamarajar Hall")):
        s = seqs[concert]
        for (a, b), l in zip(s["intervals"], s["labels"]):
            c = "lightgrey" if l == OTHER else cols.setdefault(l, plt.cm.tab20(len(cols) % 20))
            ax.barh(0, (b - a) / 60, left=a / 60, color=c, edgecolor="k", linewidth=0.5)
            if l != OTHER and (b - a) > 420:
                ax.text((a + b) / 120, 0, l, ha="center", va="center", fontsize=7)
        ax.set_yticks([]); ax.set_title(concert + " (grey = other)", fontsize=9); ax.set_xlabel("concert time after trimming (min)")
    fig.tight_layout(); fig.savefig(f"{ROOT}/results/figures/t9_timelines.png", dpi=130)


if __name__ == "__main__":
    main()
