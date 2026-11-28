"""T12 / E3 (tuning half): re-tune D3 on B1-VAL for each window length L. Never reads test.

Command: python scripts/11_tune_window_length.py --L 10 [--L 20 ...]   then   python scripts/11_tune_window_length.py --freeze
Per L: W2 model models/w2_L{L}.joblib (L=30: models/w2.joblib), posteriors on every B1-val sequence, D3 grid
(eps x min_dur x beta, same grids as T10), best by pooled boundary F@5 (ties: frame accuracy).
Writes results/e3_tuning_L{L}.csv; --freeze combines them into configs/window_length_frozen.yaml (commit it before run_e3_e4.py).
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import os
import sys
import time

import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/src")
from raga_tracking.decode import decode_d1, decode_d3  # noqa: E402
from raga_tracking.features.dataset import load_cfg  # noqa: E402
from raga_tracking.models.linear import git_hash  # noqa: E402
from raga_tracking.models.posterior import load_model, sequence_posteriors  # noqa: E402

spec = importlib.util.spec_from_file_location("t10", f"{ROOT}/scripts/10_tune_decoders.py")
t10 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t10)  # reuses load_val (val only), score, best_of, grids
LENGTHS = (10, 20, 30, 60)


def model_name(L: int) -> str:
    return "w2" if L == 30 else f"w2_L{L}"


def tune(L: int) -> None:
    cfg = load_cfg()
    hop, dcfg = cfg["window"]["hop_s"], cfg["decoder"]
    seqs = t10.load_val()
    name = model_name(L)
    m = load_model(name)
    assert m["L"] == L, (m["L"], L)
    t0 = time.time()
    posts = [sequence_posteriors(m, name, s["cents"], s["voiced"]) for s in seqs]
    print(f"L={L}: {len(seqs)} val sequences, posteriors {time.time() - t0:.0f}s", flush=True)
    d1 = t10.score(seqs, posts, m["ragas"], lambda p: decode_d1(p["post"], p["valid"], p["centres"], hop, dcfg["d1_width"]))
    rows = []
    for eps, md, beta in itertools.product(dcfg["d3_eps_grid"], dcfg["d3_min_dur_grid"], t10.BETAS):
        r = t10.score(seqs, posts, m["ragas"], lambda p: decode_d3(p["post"], p["valid"], p["centres"], hop, eps, md, None, beta))
        rows.append({"L": L, "eps": eps, "min_dur_s": md, "beta": beta, **r})
    df = pd.DataFrame(rows)
    df["d1_F5"] = d1["F5"]
    df.to_csv(f"{ROOT}/results/e3_tuning_L{L}.csv", index=False)
    b = t10.best_of(rows)
    print(f"L={L} best D3 {b['eps']} {b['min_dur_s']} {b['beta']} F5 {b['F5']:.3f} (D1 {d1['F5']:.3f}) {time.time() - t0:.0f}s", flush=True)


def freeze() -> None:
    fz = {"_header": f"Produced by: python scripts/11_tune_window_length.py --freeze | commit {git_hash()} | tuned on B1-val only (W2)"}
    for L in LENGTHS:
        df = pd.read_csv(f"{ROOT}/results/e3_tuning_L{L}.csv")
        b = df.sort_values(["F5", "frame_acc"], ascending=False).iloc[0]
        fz[f"L{L}"] = {"model": model_name(L), "L": L, "d1_width": 5,
                       "d3": {"eps": float(b.eps), "min_dur_s": int(b.min_dur_s), "beta": float(b.beta), "prior": "uniform"},
                       "val_F5": float(b.F5), "val_D1_F5": float(b.d1_F5)}
    yaml.safe_dump(fz, open(f"{ROOT}/configs/window_length_frozen.yaml", "w"), sort_keys=False)
    print(yaml.safe_dump(fz, sort_keys=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, action="append", default=[])
    ap.add_argument("--freeze", action="store_true")
    a = ap.parse_args()
    for L in a.L:
        tune(L)
    if a.freeze:
        freeze()
