"""T5: W2 TDMS window classifier (logistic regression on flattened surfaces). Uses ONLY train and val.

Command: python scripts/05_train_w2.py [--L 30] [--tag NAME] [--fast]
Stage 1: grid over tau and n_bins with C fixed (config w2.C_stage1); stage 2: tune C on the best (tau, bins).
Writes: models/w2{tag}.joblib, results/e1_w2{tag}.json, results/figures/e1_w2{tag}_confusion.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import joblib
import matplotlib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/src")
from raga_tracking.features.dataset import load_cfg, raga_list  # noqa: E402
from raga_tracking.features.tdms import tdms_windows  # noqa: E402
from raga_tracking.models.linear import SPLITS_USED, git_hash, majority_by_recording, sample_split  # noqa: E402


def feats(cents: np.ndarray, tau: float, bins: int, cfg: dict) -> np.ndarray:
    """TDMS features rescaled so the mean cell value is 1 (global rescale, no per-feature standardisation)."""
    return tdms_windows(cents, tau, bins, cfg["alpha"], cfg["sigma"]) * (bins * bins)


def fit_eval(Xtr, ytr, Xva, yva, C, cw):
    clf = LogisticRegression(C=C, max_iter=300, class_weight=cw)
    clf.fit(Xtr, ytr)
    p = clf.predict(Xva)
    return clf, p, accuracy_score(yva, p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=float, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--fix", default=None, help="tau,bins: skip stage 1 and reuse results/e1_w2_stage1.json")
    ap.add_argument("--fast", action="store_true", help="smoke test: one (tau, bins) config, one C")
    a = ap.parse_args()
    cfg = load_cfg()
    w2, w1 = cfg["w2"], cfg["w1"]
    L, hop, seed = a.L or cfg["window"]["length_s"], cfg["window"]["hop_s"], cfg["seed"]
    ragas, K, cw = raga_list(), len(raga_list()), w1.get("class_weight")
    Ctr, ytr, _ = sample_split("train", ragas, L, hop, w1["max_windows_per_rec"], seed)
    Cva, yva, rva = sample_split("val", ragas, L, hop, w1["max_windows_per_rec"], seed + 1)
    print(f"train windows {Ctr.shape}, val windows {Cva.shape}, K={K}", flush=True)

    taus = [w2["tau_s"]] if a.fast else w2["tau_grid"]
    binss = [w2["n_bins"]] if a.fast else w2["bins_grid"]
    stage1, best = [], None
    if a.fix:  # stage-1 grid already run (committed in e1_w2_stage1.json); only re-tune C
        taus, binss = [], []
        stage1 = json.load(open(f"{ROOT}/results/e1_w2_stage1.json"))["stage1_grid"]
        best = (0.0, float(a.fix.split(",")[0]), int(a.fix.split(",")[1]))
    for tau in taus:
        for bins in binss:
            t0 = time.time()
            Xtr, Xva = feats(Ctr, tau, bins, w2), feats(Cva, tau, bins, w2)
            _, _, acc = fit_eval(Xtr, ytr, Xva, yva, w2["C_stage1"], cw)
            stage1.append({"tau_s": tau, "n_bins": bins, "C": w2["C_stage1"], "val_window_acc": acc})
            print(f"tau={tau} bins={bins} C={w2['C_stage1']}: val acc {acc:.4f} ({time.time()-t0:.0f}s)", flush=True)
            if best is None or acc > best[0]:
                best = (acc, tau, bins)
    _, tau, bins = best
    Xtr, Xva = feats(Ctr, tau, bins, w2), feats(Cva, tau, bins, w2)
    stage2, top = [], None
    for C in ([w2["C_stage1"]] if a.fast else w2["C_grid"]):
        clf, p, acc = fit_eval(Xtr, ytr, Xva, yva, C, cw)
        stage2.append({"C": C, "val_window_acc": acc})
        print(f"[best tau={tau} bins={bins}] C={C}: val acc {acc:.4f}", flush=True)
        if top is None or acc > top[0]:
            top = (acc, C, clf, p)
    acc, C, clf, pred = top
    cm = confusion_matrix(yva, pred, labels=list(range(K)))
    recall = np.diag(cm) / np.maximum(cm.sum(1), 1)
    out = {
        "_header": f"Produced by: python scripts/05_train_w2.py {' '.join(sys.argv[1:])} | commit {git_hash()} | splits used: {SPLITS_USED}",
        "model": "w2", "tau_s": tau, "n_bins": bins, "alpha": w2["alpha"], "sigma": w2["sigma"], "L_s": L, "hop_s": hop,
        "K": K, "chance": 1 / K, "best_C": C, "class_weight": cw,
        "stage1_grid": stage1, "stage2_C_grid": stage2,
        "val_window_acc": acc,
        "val_macro_f1": f1_score(yva, pred, average="macro", labels=list(range(K)), zero_division=0),
        "val_recording_majority_acc": majority_by_recording(pred, yva, rva),
        "n_train_windows": int(len(ytr)), "n_val_windows": int(len(yva)), "n_val_recordings": int(len(np.unique(rva))),
        "per_raga_recall_val": {r: float(x) for r, x in zip(ragas, recall)}, "min_raga_recall": float(recall.min()),
    }
    joblib.dump({"clf": clf, "ragas": ragas, "tau_s": tau, "n_bins": bins, "alpha": w2["alpha"], "sigma": w2["sigma"],
                 "L": L, "hop": hop, "C": C}, f"{ROOT}/models/w2{a.tag}.joblib")
    json.dump(out, open(f"{ROOT}/results/e1_w2{a.tag}.json", "w"), indent=2)
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.imshow(cm / np.maximum(cm.sum(1, keepdims=True), 1), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels(ragas, rotation=90, fontsize=7); ax.set_yticklabels(ragas, fontsize=7)
    ax.set_xlabel("predicted raga"); ax.set_ylabel("true raga")
    fig.tight_layout(); fig.savefig(f"{ROOT}/results/figures/e1_w2{a.tag}_confusion.png", dpi=130)
    print(json.dumps({k: v for k, v in out.items() if k not in ("per_raga_recall_val", "stage1_grid", "stage2_C_grid", "_header")}, indent=1))


if __name__ == "__main__":
    main()
