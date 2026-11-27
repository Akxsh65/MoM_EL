"""T11 / E2: the ONE frozen test run. Command: python scripts/run_final.py

Refuses to run unless configs/src/scripts/tests are committed (so the config commit predates the results) and
results/e2_main.csv does not exist yet. Runs every classifier (W1, W2) x decoder (D0-D3) on the B1-test (easy/mid/hard)
and B2-test benchmarks with the decoder settings frozen in configs/decoders_frozen.yaml (tuned on B1-val in T10).
Writes results/e2_main.csv|md, results/e2_paired.csv, results/e2_per_sequence.csv, results/e2_predictions.json.gz.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/src")
from raga_tracking.bench import concerts as b2mod  # noqa: E402
from raga_tracking.bench import synthetic as b1mod  # noqa: E402
from raga_tracking.decode import decode_d0, decode_d1, decode_d2, decode_d3  # noqa: E402
from raga_tracking.eval.bootstrap import COLS, METRICS, ci_table, draws, paired_diff  # noqa: E402
from raga_tracking.eval.metrics import boundary_counts, frame_accuracy, segment_accuracy_counts  # noqa: E402
from raga_tracking.features.dataset import load_cfg  # noqa: E402
from raga_tracking.models.posterior import load_model, sequence_posteriors  # noqa: E402

SPLIT = "test"
DECODERS = ("D0", "D1", "D2", "D3")
N_BOOT = 1000


def sh(*a: str) -> str:
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def assert_frozen(allow_rerun: str | None) -> str:
    dirty = sh("status", "--porcelain", "--", "configs", "src", "scripts", "tests", "manifests")
    if dirty:
        sys.exit("Refusing to run: uncommitted changes in configs/src/scripts/tests/manifests:\n" + dirty)
    if os.path.exists(f"{ROOT}/results/e2_main.csv") and not allow_rerun:
        sys.exit("results/e2_main.csv exists: the test run happens once per frozen config (CLAUDE.md rule 2). "
                 "Pass --allow_rerun 'reason' only for a bug fix, and log it in docs/decisions.md.")
    return sh("rev-parse", "--short", "HEAD")


def load_units() -> list[dict]:
    """B1-test sequences (easy/mid/hard) and B2 concerts, hash-checked against their manifests."""
    units = []
    for e in json.load(open(f"{ROOT}/manifests/b1_manifest.json"))["sequences"]:
        if e["split"] != SPLIT:
            continue
        z = np.load(f"{ROOT}/data/processed/b1/{SPLIT}/{e['seq_id']}.npz")
        s = {k: z[k] for k in ("cents", "voiced", "intervals", "labels")}
        assert b1mod.content_hash(s) == e["content_sha256"], e["seq_id"]
        units.append({"bench": f"B1-{e['label']}", "id": e["seq_id"], "cluster": e["concert"], **s})
    for e in json.load(open(f"{ROOT}/manifests/b2_manifest.json"))["concerts"]:
        import re
        z = np.load(f"{ROOT}/data/processed/b2/{re.sub(r'[^A-Za-z0-9]+', '_', e['concert'])}.npz")
        s = {k: z[k] for k in ("cents", "voiced", "intervals", "labels", "item_intervals")}
        assert b2mod.content_hash(s) == e["content_sha256"], e["concert"]
        units.append({"bench": "B2", "id": e["concert"], "cluster": e["concert"], **s})
    return units


def decoder_fns(fz: dict, hop: float) -> dict:
    d2, d3 = fz["d2"], fz["d3"]
    return {"D0": lambda p: decode_d0(p["post"], p["valid"], p["centres"], hop),
            "D1": lambda p: decode_d1(p["post"], p["valid"], p["centres"], hop, fz["d1_width"]),
            "D2": lambda p: decode_d2(p["hists"], p["post"], p["valid"], p["centres"], hop, d2["k"], d2["thr"]),
            "D3": lambda p: decode_d3(p["post"], p["valid"], p["centres"], hop, d3["eps"], d3["min_dur_s"], None, d3["beta"])}


def unit_counts(u: dict, iv: np.ndarray, names: np.ndarray) -> list[float]:
    t5, t10 = boundary_counts(u["intervals"], iv, 5.0), boundary_counts(u["intervals"], iv, 10.0)
    fa = frame_accuracy(u["intervals"], u["labels"], iv, names)
    sa = segment_accuracy_counts(u["intervals"], u["labels"], iv, names)
    return [*t5, *t10, *fa, *sa, len(u["cents"]) / 100.0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow_rerun", default=None)
    a = ap.parse_args()
    commit = assert_frozen(a.allow_rerun)
    cfg = load_cfg()
    hop, seed = cfg["window"]["hop_s"], cfg["seed"]
    fz = yaml.safe_load(open(f"{ROOT}/configs/decoders_frozen.yaml"))
    cfg_hash = hashlib.sha256(open(f"{ROOT}/configs/decoders_frozen.yaml", "rb").read()).hexdigest()[:12]
    units = load_units()
    print(len(units), "test units", flush=True)
    rows, preds = [], {}
    for name in ("w1", "w2"):
        m = load_model(name)
        fns = decoder_fns(fz[name], hop)
        for i, u in enumerate(units):
            p = sequence_posteriors(m, name, u["cents"].astype(float), u["voiced"])
            for d in DECODERS:
                iv, lab = fns[d](p)
                names = np.array([m["ragas"][j] for j in lab])
                rows.append({"model": name, "decoder": d, "bench": u["bench"], "id": u["id"], "cluster": u["cluster"],
                             **dict(zip(COLS, unit_counts(u, iv, names)))})
                preds[f"{name}|{d}|{u['id']}"] = {"intervals": np.round(iv, 3).tolist(), "labels": names.tolist()}
            if i % 100 == 0:
                print(name, i, flush=True)
    per = pd.DataFrame(rows)
    per.to_csv(f"{ROOT}/results/e2_per_sequence.csv", index=False)
    with gzip.open(f"{ROOT}/results/e2_predictions.json.gz", "wt") as f:
        json.dump(preds, f)
    rng = np.random.default_rng(seed + 7)
    out, paired = [], []
    for (name, bench), g in per.groupby(["model", "bench"], sort=False):
        ids = g[g.decoder == "D0"].id.tolist()
        clusters = g[g.decoder == "D0"].cluster.to_numpy()
        mult = draws(len(ids), N_BOOT, rng)
        multc = draws(len(ids), N_BOOT, rng, clusters)
        cnt = {d: g[g.decoder == d].set_index("id").loc[ids, list(COLS)].to_numpy(float) for d in DECODERS}
        for d in DECODERS:
            t, tc = ci_table(cnt[d], mult), ci_table(cnt[d], multc)
            row = {"model": name.upper(), "decoder": d, "bench": bench, "n_units": len(ids), "n_concerts": len(set(clusters))}
            for k in METRICS:
                row[k], row[k + "_lo"], row[k + "_hi"] = t[k]
            row["F5_lo_concert"], row["F5_hi_concert"] = tc["F5"][1], tc["F5"][2]
            out.append(row)
        for a_, b_ in (("D3", "D1"), ("D3", "D0")):
            r = paired_diff(cnt[a_], cnt[b_], mult, "F5")
            rc = paired_diff(cnt[a_], cnt[b_], multc, "F5")
            paired.append({"model": name.upper(), "bench": bench, "A": a_, "B": b_, "dF5": r["diff"], "lo": r["lo"], "hi": r["hi"],
                           "p_A_not_better": r["p_a_not_better"], "lo_concert": rc["lo"], "hi_concert": rc["hi"]})
    hdr = (f"Produced by: python scripts/run_final.py | commit {commit} | decoders_frozen.yaml sha256[:12] {cfg_hash} | "
           f"bootstrap {N_BOOT} resamples over sequences (B2: over concerts), seed {seed + 7}; *_concert = concert-cluster bootstrap")
    df, pf = pd.DataFrame(out), pd.DataFrame(paired)
    for path, frame in ((f"{ROOT}/results/e2_main.csv", df), (f"{ROOT}/results/e2_paired.csv", pf)):
        with open(path, "w", newline="") as fh:
            fh.write("# " + hdr + "\n")  # read with pd.read_csv(path, comment="#")
            frame.to_csv(fh, index=False)
    write_md(df, pf, hdr)


def fmt(r, k) -> str:
    return f"{r[k]:.3f} [{r[k + '_lo']:.3f}, {r[k + '_hi']:.3f}]"


def write_md(df: pd.DataFrame, pf: pd.DataFrame, hdr: str) -> None:
    L = ["# E2: main test evaluation", "", hdr, "",
         "Boundary P/R/F at +-5 s (F@10 at +-10 s), frame accuracy (reference `other` excluded), segment accuracy, false switches "
         "per 10 min (estimated boundaries with no reference boundary within 5 s). Cells: estimate [95% CI]. B1 units are sequences; "
         "B2 units are the 8 concerts. B1 sequences reuse a few source items, so the sequence-bootstrap CIs are too narrow; "
         "see `F5 concert-CI`.", ""]
    for bench in ("B1-easy", "B1-hard", "B1-mid", "B2"):
        L += [f"## {bench}", "", "| model | decoder | units | F@5 | P@5 | R@5 | F@10 | frame acc | segment acc | false sw. /10 min | F@5 concert-CI |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in df[df.bench == bench].to_dict("records"):
            L.append(f"| {r['model']} | {r['decoder']} | {r['n_units']} | {fmt(r, 'F5')} | {fmt(r, 'P5')} | {fmt(r, 'R5')} | {fmt(r, 'F10')} | "
                     f"{fmt(r, 'frame_acc')} | {fmt(r, 'seg_acc')} | {fmt(r, 'false_sw_10min')} | [{r['F5_lo_concert']:.3f}, {r['F5_hi_concert']:.3f}] |")
        L.append("")
    L += ["## Paired bootstrap on F@5 (A minus B)", "", "| model | benchmark | A vs B | diff | 95% CI (sequences) | 95% CI (concert clusters) | resamples with A not better |",
          "|---|---|---|---|---|---|---|"]
    for r in pf.to_dict("records"):
        L.append(f"| {r['model']} | {r['bench']} | {r['A']} vs {r['B']} | {r['dF5']:+.3f} | [{r['lo']:+.3f}, {r['hi']:+.3f}] | "
                 f"[{r['lo_concert']:+.3f}, {r['hi_concert']:+.3f}] | {r['p_A_not_better']:.3f} |")
    open(f"{ROOT}/results/e2_main.md", "w", encoding="utf8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
