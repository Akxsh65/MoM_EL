"""T8: build the B1 synthetic benchmark (val + test). Command: python scripts/08_build_b1.py [--hard_min 6]

Writes data/processed/b1/{split}/{seq_id}.npz (gitignored) and data/processed/b1/b1_manifest.json (source rec_ids, cut
times, seed, label, content hash). The manifest is the released artifact; the npz files rebuild from it.
Never touches the train split. No model is involved; nothing here is tuned on test.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/src")
from raga_tracking.bench.synthetic import (HARD_MIN, build_manifest, content_hash, render,  # noqa: E402
                                           save_manifest)
from raga_tracking.features.dataset import load_cfg  # noqa: E402
from raga_tracking.models.linear import git_hash  # noqa: E402

QUOTAS = {"test": {"easy": 200, "hard": 200, "mid": 100}, "val": {"easy": 80, "hard": 80, "mid": 40}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hard_min", type=int, default=HARD_MIN)
    a = ap.parse_args()
    seed = load_cfg()["seed"]
    out_dir = f"{ROOT}/data/processed/b1"
    os.makedirs(out_dir, exist_ok=True)
    allseq = []
    for k, split in enumerate(("val", "test")):
        ents = build_manifest(split, QUOTAS[split], seed + 100 + k, hard_min=a.hard_min)
        os.makedirs(f"{out_dir}/{split}", exist_ok=True)
        for e in ents:
            s = render(e)
            e["content_sha256"] = content_hash(s)
            e["duration_s"] = float(len(s["cents"]) / 100)
            np.savez_compressed(f"{out_dir}/{split}/{e['seq_id']}.npz", **s)
        c = collections.Counter(e["label"] for e in ents)
        print(split, len(ents), dict(c), "concerts", len({e["concert"] for e in ents}), flush=True)
        allseq += ents
    sc = open(f"{ROOT}/configs/scales.yaml", "rb").read()
    meta = {"_header": f"Produced by: python scripts/08_build_b1.py --hard_min {a.hard_min} | commit {git_hash()}",
            "seed": seed, "quotas": QUOTAS, "hard_min": a.hard_min, "easy_max": 4, "excerpt_len_s": [60, 240],
            "min_voiced": 0.6, "scales_sha256": hashlib.sha256(sc).hexdigest(),
            "n_scales_unverified": sum(1 for v in yaml.safe_load(open(f"{ROOT}/configs/scales.yaml")).values() if isinstance(v, dict) and v.get("verify"))}
    save_manifest(allseq, f"{out_dir}/b1_manifest.json", meta)
    # summary for results/
    L = ["# B1 summary (T8)", "", f"Produced by: `python scripts/08_build_b1.py --hard_min {a.hard_min}` | commit {git_hash()}", "",
         "| split | label | sequences | mean excerpts | mean duration (min) | concerts |", "|---|---|---|---|---|---|"]
    for split in ("val", "test"):
        for lab in ("easy", "mid", "hard"):
            x = [e for e in allseq if e["split"] == split and e["label"] == lab]
            if x:
                L.append(f"| {split} | {lab} | {len(x)} | {np.mean([len(e['excerpts']) for e in x]):.2f} | "
                         f"{np.mean([e['duration_s'] for e in x]) / 60:.1f} | {len({e['concert'] for e in x})} |")
    L += ["", "easy = all neighbouring pairs share <= 4 svaras; hard = at least one pair shares >= %d; mid = neither." % a.hard_min,
          "Labels rest on configs/scales.yaml, which still has unverified entries; they are re-derivable from the manifest's raga list."]
    open(f"{ROOT}/results/b1_summary.md", "w", encoding="utf8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
