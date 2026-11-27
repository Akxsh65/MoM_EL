"""T10: tune decoders on B1-VAL only (never test).

Command: python scripts/10_tune_decoders.py [--models w1,w2]
For each classifier: posteriors on every B1-val sequence (no temperature, T7 backup), then
  D0 argmax, D1 majority (width 5), D2 novelty (threshold grid), D3 (eps x min_dur x beta grid).
Selection metric: pooled boundary F@5 s on B1-val (ties: higher frame accuracy). Frame accuracy uses a 1 s grid.
Writes results/d3_tuning.csv (all configs), results/d2_tuning.csv, results/t10_summary.md, configs/decoders_frozen.yaml.
Test sequences are never loaded.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/src")
from raga_tracking.decode import decode_d0, decode_d1, decode_d2, decode_d3  # noqa: E402
from raga_tracking.eval.metrics import boundary_counts, frame_accuracy, pooled_prf  # noqa: E402
from raga_tracking.features.dataset import load_cfg  # noqa: E402
from raga_tracking.models.linear import git_hash  # noqa: E402
from raga_tracking.models.posterior import load_model, sequence_posteriors  # noqa: E402

SPLIT = "val"  # hard-coded: this script must never read test
BETAS = [0.1, 0.25, 0.5, 1.0, 2.0]
D2_THR = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
KEYS = ("P5", "R5", "F5", "F10", "frame_acc", "n_est", "n_ref")


def load_val() -> list[dict]:
    man = json.load(open(f"{ROOT}/manifests/b1_manifest.json"))["sequences"]
    out = []
    for e in man:
        if e["split"] != SPLIT:
            continue
        z = np.load(f"{ROOT}/data/processed/b1/{SPLIT}/{e['seq_id']}.npz")
        out.append({"id": e["seq_id"], "label": e["label"], "cents": z["cents"].astype(float), "voiced": z["voiced"],
                    "iv": z["intervals"], "lab": z["labels"]})
    return out


def score(seqs: list[dict], posts: list[dict], ragas: list[str], fn) -> dict:
    """Pooled boundary P/R/F at 5 and 10 s and frame accuracy of decoder fn(post_dict) -> (intervals, label ids)."""
    c5, c10, fa, tot = [], [], 0, 0
    for s, p in zip(seqs, posts):
        iv, lab = fn(p)
        names = np.array([ragas[i] for i in lab])
        c5.append(boundary_counts(s["iv"], iv, 5.0))
        c10.append(boundary_counts(s["iv"], iv, 10.0))
        a, b = frame_accuracy(s["iv"], s["lab"], iv, names, skip=())
        fa += a
        tot += b
    p5, r5, f5 = pooled_prf(c5)
    return {"P5": p5, "R5": r5, "F5": f5, "F10": pooled_prf(c10)[2], "frame_acc": fa / tot,
            "n_est": sum(c[1] for c in c5), "n_ref": sum(c[2] for c in c5)}


def best_of(rs: list[dict]) -> dict:
    return max(rs, key=lambda r: (r["F5"], r["frame_acc"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="w1,w2")
    a = ap.parse_args()
    cfg = load_cfg()
    hop, dcfg = cfg["window"]["hop_s"], cfg["decoder"]
    seqs = load_val()
    print(len(seqs), "B1-val sequences", flush=True)
    rows, d2rows, summary = [], [], []
    frozen = {"_header": f"Produced by: python scripts/10_tune_decoders.py | commit {git_hash()} | tuned on B1-val only"}
    for name in a.models.split(","):
        m = load_model(name)
        ragas = m["ragas"]
        t0 = time.time()
        posts = [sequence_posteriors(m, name, s["cents"], s["voiced"]) for s in seqs]
        print(name, "posteriors", f"{time.time() - t0:.0f}s", flush=True)
        base = {"D0": score(seqs, posts, ragas, lambda p: decode_d0(p["post"], p["valid"], p["centres"], hop)),
                "D1": score(seqs, posts, ragas, lambda p: decode_d1(p["post"], p["valid"], p["centres"], hop, dcfg["d1_width"]))}
        d2 = []
        for thr in D2_THR:
            r = score(seqs, posts, ragas, lambda p: decode_d2(p["hists"], p["post"], p["valid"], p["centres"], hop, dcfg["d2_k"], thr))
            d2.append({"model": name, "thr": thr, **r})
        d2rows += d2
        best2 = best_of(d2)
        base["D2"] = best2
        grid = []
        for eps, md, beta in itertools.product(dcfg["d3_eps_grid"], dcfg["d3_min_dur_grid"], BETAS):
            r = score(seqs, posts, ragas, lambda p: decode_d3(p["post"], p["valid"], p["centres"], hop, eps, md, None, beta))
            grid.append({"model": name, "eps": eps, "min_dur_s": md, "beta": beta, **r})
        rows += grid
        best3 = best_of(grid)
        base["D3"] = best3
        for d, r in base.items():
            summary.append({"model": name, "decoder": d, **{k: r[k] for k in KEYS}})
            print(name, d, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)}, flush=True)
        frozen[name] = {"d1_width": dcfg["d1_width"], "d2": {"k": dcfg["d2_k"], "thr": best2["thr"]},
                        "d3": {"eps": best3["eps"], "min_dur_s": best3["min_dur_s"], "beta": best3["beta"], "prior": "uniform"}}
        print(name, "time", f"{time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(f"{ROOT}/results/d3_tuning.csv", index=False)
    pd.DataFrame(d2rows).to_csv(f"{ROOT}/results/d2_tuning.csv", index=False)
    yaml.safe_dump(frozen, open(f"{ROOT}/configs/decoders_frozen.yaml", "w"), sort_keys=False)
    S = pd.DataFrame(summary)
    L = ["# T10: decoder tuning on B1-val", "",
         f"Produced by: `python scripts/10_tune_decoders.py` | commit {git_hash()} | B1-val only ({len(seqs)} sequences)", "",
         "Pooled boundary P/R/F at +-5 s (F10 at +-10 s), frame accuracy on a 1 s grid. D2 and D3 rows are the best of their grids by F5 "
         "(selected on the data they are scored on, so optimistic; D0 and D1 have no tuned parameters).", "",
         "| model | decoder | P@5 | R@5 | F@5 | F@10 | frame acc | est. boundaries | ref. boundaries |", "|---|---|---|---|---|---|---|---|---|"]
    for r in S.itertuples():
        L.append(f"| {r.model.upper()} | {r.decoder} | {r.P5:.3f} | {r.R5:.3f} | {r.F5:.3f} | {r.F10:.3f} | {r.frame_acc:.3f} | {r.n_est} | {r.n_ref} |")
    L += ["", "## Checks", ""]
    for name in a.models.split(","):
        s = S[S.model == name].set_index("decoder")
        ok = s.loc["D3", "F5"] > s.loc["D0", "F5"]
        L.append(f"- {name.upper()}: D3 F@5 {s.loc['D3', 'F5']:.3f} vs D0 {s.loc['D0', 'F5']:.3f}: **{'PASS' if ok else 'FAIL'}**; "
                 f"vs D1 {s.loc['D1', 'F5']:.3f}: {'D3 higher' if s.loc['D3', 'F5'] > s.loc['D1', 'F5'] else 'D3 not higher'}")
    open(f"{ROOT}/results/t10_summary.md", "w", encoding="utf8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
