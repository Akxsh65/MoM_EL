"""T7: temperature scaling for W1/W2 (fit on val), ECE (15 bins) before/after, median max-probability.

Command: python scripts/07_calibrate.py
Uses ONLY the val split. Besides the plan's in-sample numbers (T fit and scored on val), reports a cross-fitted
check: T fit on half of the val recordings, scored on the other half (and swapped), to expose optimism.
Writes: models/{w1,w2}_temperature.json, results/calibration.md, results/calibration.json
"""
from __future__ import annotations

import json
import os
import sys

import joblib
import numpy as np
from scipy.special import log_softmax

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/src")
from raga_tracking.features.dataset import load_cfg  # noqa: E402
from raga_tracking.features.histogram import window_histograms  # noqa: E402
from raga_tracking.features.tdms import tdms_windows  # noqa: E402
from raga_tracking.models.calibration import apply_temperature, ece, fit_temperature  # noqa: E402
from raga_tracking.models.linear import git_hash, sample_split  # noqa: E402


def val_logits(name: str, cents: np.ndarray) -> np.ndarray:
    m = joblib.load(f"{ROOT}/models/{name}.joblib")
    if name == "w1":
        X = window_histograms(cents, m["n_bins"])
    else:
        X = tdms_windows(cents, m["tau_s"], m["n_bins"], m["alpha"], m["sigma"]) * (m["n_bins"] ** 2)
    return m["clf"].decision_function(X)


def stats(logits: np.ndarray, y: np.ndarray, T: float) -> dict:
    p = apply_temperature(logits, T)
    return {"ece": ece(p, y), "nll": float(-log_softmax(logits / T, axis=1)[np.arange(len(y)), y].mean()),
            "median_max_prob": float(np.median(p.max(1))), "mean_conf": float(p.max(1).mean()), "acc": float((p.argmax(1) == y).mean())}


def main() -> None:
    cfg = load_cfg()
    ragas_n = None
    res, rows = {}, []
    for name in ("w1", "w2"):
        m = joblib.load(f"{ROOT}/models/{name}.joblib")
        cents, y, rid = sample_split("val", m["ragas"], m["L"], m["hop"], cfg["w1"]["max_windows_per_rec"], cfg["seed"] + 1)
        z = val_logits(name, cents)
        T = fit_temperature(z, y)
        before, after = stats(z, y, 1.0), stats(z, y, T)
        # cross-fit: 2 folds over val recordings (seeded)
        recs = np.unique(rid)
        rng = np.random.default_rng(cfg["seed"])
        fold = dict(zip(recs, rng.permutation(len(recs)) % 2))
        f = np.array([fold[r] for r in rid])
        cf_b, cf_a, Ts = [], [], []
        for k in (0, 1):
            Tk = fit_temperature(z[f != k], y[f != k])
            Ts.append(Tk)
            cf_b.append(stats(z[f == k], y[f == k], 1.0)["ece"])
            cf_a.append(stats(z[f == k], y[f == k], Tk)["ece"])
        res[name] = {"T": T, "before": before, "after": after,
                     "crossfit": {"T_folds": Ts, "ece_before": float(np.mean(cf_b)), "ece_after": float(np.mean(cf_a))},
                     "n_val_windows": int(len(y))}
        json.dump({"T": T, "fit_on": "val", "commit": git_hash()}, open(f"{ROOT}/models/{name}_temperature.json", "w"))
        rows.append((name, res[name]))
        print(name, "T=%.3f" % T, "ECE %.4f -> %.4f" % (before["ece"], after["ece"]),
              "median max-p %.3f -> %.3f" % (before["median_max_prob"], after["median_max_prob"]),
              "crossfit ECE %.4f -> %.4f" % (res[name]["crossfit"]["ece_before"], res[name]["crossfit"]["ece_after"]), flush=True)
    json.dump(res, open(f"{ROOT}/results/calibration.json", "w"), indent=2)
    L = ["# Calibration (T7)", "", f"Produced by: `python scripts/07_calibrate.py` | commit {git_hash()} | val split only", "",
         "Temperature scaling, T fit on val windows (NLL). ECE with 15 equal-width bins on the max-probability prediction.", "",
         "| model | T | ECE before | ECE after | NLL before | NLL after | median max-p before | median max-p after | mean conf. before | val acc |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in rows:
        b, a = r["before"], r["after"]
        L.append(f"| {n.upper()} | {r['T']:.3f} | {b['ece']:.4f} | {a['ece']:.4f} | {b['nll']:.3f} | {a['nll']:.3f} | "
                 f"{b['median_max_prob']:.3f} | {a['median_max_prob']:.3f} | {b['mean_conf']:.3f} | {a['acc']:.3f} |")
    L += ["", "Cross-fitted check (T fit on one half of the val recordings, ECE scored on the other half, averaged over both halves):", "",
          "| model | T (fold 0, fold 1) | ECE before | ECE after |", "|---|---|---|---|"]
    for n, r in rows:
        c = r["crossfit"]
        L.append(f"| {n.upper()} | {c['T_folds'][0]:.3f}, {c['T_folds'][1]:.3f} | {c['ece_before']:.4f} | {c['ece_after']:.4f} |")
    L += ["", "## Check results (computed)", ""]
    for n, r in rows:
        ok_ece = r["after"]["ece"] < r["before"]["ece"]
        ok_cf = r["crossfit"]["ece_after"] < r["crossfit"]["ece_before"]
        ok_med = r["after"]["median_max_prob"] < 0.99
        L.append(f"- {n.upper()}: ECE decreases in-sample: **{'PASS' if ok_ece else 'FAIL'}**; cross-fitted: "
                 f"**{'PASS' if ok_cf else 'FAIL'}**; median max-prob < 0.99: **{'PASS' if ok_med else 'FAIL'}**")
    L += ["", "Reading: mean confidence below accuracy means the model is under-confident, so a temperature T > 1 (which flattens)",
          "cannot reduce ECE. If a check fails, the T7 backup applies: no temperature, tune a posterior exponent beta in D3 on B1-val (T10)."]
    open(f"{ROOT}/results/calibration.md", "w", encoding="utf8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
