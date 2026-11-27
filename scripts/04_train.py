"""T4: W1 histogram window classifier (multinomial logistic regression). Uses ONLY train and val splits.

Command: python scripts/04_train.py --model w1 [--n_bins 24] [--L 30] [--tag NAME]
Writes: models/w1{tag}.joblib, results/e1_w1{tag}.json, results/figures/e1_w1{tag}_confusion.png
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import joblib
import matplotlib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/src")
from raga_tracking.features.dataset import load_cfg, raga_list, recordings, sample_windows  # noqa: E402
from raga_tracking.features.histogram import window_histograms  # noqa: E402

SPLITS_USED = ("train", "val")  # never "test" (CLAUDE.md rule 2)


def build(split: str, ragas: list[str], L: float, hop: float, cap: int, n_bins: int, seed: int):
    """Features X, labels y, recording ids per window for one split."""
    assert split in SPLITS_USED
    rng = np.random.default_rng(seed)
    X, y, rid = [], [], []
    for _, r in recordings(split).iterrows():
        w = sample_windows(r.rec_id, L, hop, cap, rng)
        if len(w["idx"]) == 0:
            continue
        X.append(window_histograms(w["cents"], n_bins))
        y += [ragas.index(r.raga)] * len(w["idx"])
        rid += [r.rec_id] * len(w["idx"])
    return np.vstack(X), np.array(y), np.array(rid)


def majority_by_recording(pred: np.ndarray, y: np.ndarray, rid: np.ndarray) -> float:
    ok = []
    for r in np.unique(rid):
        m = rid == r
        ok.append(np.bincount(pred[m]).argmax() == y[m][0])
    return float(np.mean(ok))


def git_hash() -> str:
    try:
        h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
        return h + ("+uncommitted" if dirty else "")
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="w1", choices=["w1"])
    ap.add_argument("--n_bins", type=int, default=None)
    ap.add_argument("--L", type=float, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--class_weight", default="cfg", choices=["cfg", "none", "balanced"])
    a = ap.parse_args()
    cfg = load_cfg()
    n_bins = a.n_bins or cfg["w1"]["n_bins"]
    L, hop, cap, seed = a.L or cfg["window"]["length_s"], cfg["window"]["hop_s"], cfg["w1"]["max_windows_per_rec"], cfg["seed"]
    cw = cfg["w1"].get("class_weight") if a.class_weight == "cfg" else (None if a.class_weight == "none" else a.class_weight)
    ragas = raga_list()
    K = len(ragas)

    Xtr, ytr, _ = build("train", ragas, L, hop, cap, n_bins, seed)
    Xva, yva, rva = build("val", ragas, L, hop, cap, n_bins, seed + 1)
    print(f"train windows {Xtr.shape}, val windows {Xva.shape}, K={K}", flush=True)

    grid, best = [], None
    for C in cfg["w1"]["C_grid"]:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000, class_weight=cw))
        clf.fit(Xtr, ytr)
        p = clf.predict(Xva)
        acc = accuracy_score(yva, p)
        grid.append({"C": C, "val_window_acc": acc})
        print(f"C={C}: val acc {acc:.4f}", flush=True)
        if best is None or acc > best[0]:
            best = (acc, C, clf, p)
    acc, C, clf, pred = best
    f1 = f1_score(yva, pred, average="macro", labels=list(range(K)), zero_division=0)
    rec_acc = majority_by_recording(pred, yva, rva)
    cm = confusion_matrix(yva, pred, labels=list(range(K)))
    recall = (np.diag(cm) / np.maximum(cm.sum(1), 1))
    out = {
        "_header": f"Produced by: python scripts/04_train.py {' '.join(sys.argv[1:])} | commit {git_hash()} | splits used: {SPLITS_USED}",
        "model": "w1", "n_bins": n_bins, "L_s": L, "hop_s": hop, "K": K, "chance": 1 / K,
        "best_C": C, "class_weight": cw, "C_grid": grid,
        "val_window_acc": acc, "val_macro_f1": f1, "val_recording_majority_acc": rec_acc,
        "n_train_windows": int(len(ytr)), "n_val_windows": int(len(yva)),
        "n_val_recordings": int(len(np.unique(rva))),
        "per_raga_recall_val": {r: float(x) for r, x in zip(ragas, recall)},
        "min_raga_recall": float(recall.min()),
    }
    os.makedirs(f"{ROOT}/models", exist_ok=True)
    joblib.dump({"clf": clf, "ragas": ragas, "n_bins": n_bins, "L": L, "hop": hop, "C": C}, f"{ROOT}/models/w1{a.tag}.joblib")
    json.dump(out, open(f"{ROOT}/results/e1_w1{a.tag}.json", "w"), indent=2)
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.imshow(cm / np.maximum(cm.sum(1, keepdims=True), 1), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels(ragas, rotation=90, fontsize=7); ax.set_yticklabels(ragas, fontsize=7)
    ax.set_xlabel("predicted raga"); ax.set_ylabel("true raga")
    fig.tight_layout(); fig.savefig(f"{ROOT}/results/figures/e1_w1{a.tag}_confusion.png", dpi=130)
    print(json.dumps({k: v for k, v in out.items() if k not in ("per_raga_recall_val", "C_grid", "_header")}, indent=1))


if __name__ == "__main__":
    main()
