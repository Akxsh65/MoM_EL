"""T14 / E6: error analysis of the best configuration (W2 + D3, L=30, the frozen E2 setting) on the TEST benchmarks.

Command: python scripts/12_error_analysis.py
Post-hoc description of saved predictions (results/e2_predictions.json.gz); nothing is tuned and no new test decision is made.
Writes results/e6_ref_boundaries.csv, e6_est_boundaries.csv, e6_by_shared.csv, e6_pairs.csv, e6_b2_kinds.csv, e6_tables.md,
figures e6_error_vs_shared.png, e6_timing_error.png, e6_worst_b2_<rank>_<concert>.png (all 8 B2 concerts, worst first).
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import os
import re
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mir_eval.util import match_events  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/src")
from raga_tracking.bench.synthetic import load_scales  # noqa: E402
from raga_tracking.eval.metrics import label_at  # noqa: E402
from raga_tracking.models.linear import git_hash  # noqa: E402
from raga_tracking.models.posterior import load_model, sequence_posteriors  # noqa: E402

spec = importlib.util.spec_from_file_location("rf", f"{ROOT}/scripts/run_final.py")
rf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rf)
KEY = "w2|D3|"
BUCKETS = ["<=3", "4", "5", "6", ">=7"]
N_BOOT = 1000


def bucket(n: float) -> str:
    return "<=3" if n <= 3 else (">=7" if n >= 7 else str(int(n)))


def match_pairs(ref: np.ndarray, est: np.ndarray, w: float) -> dict[int, int]:
    return {int(r): int(e) for r, e in match_events(ref, est, w)} if len(ref) and len(est) else {}


def boundary_tables(units, preds, scales):
    """One row per reference boundary and one per estimated boundary."""
    R, E = [], []
    for u in units:
        pr = preds[KEY + u["id"]]
        piv, plab = np.array(pr["intervals"]), np.array(pr["labels"])
        iv, lab = u["intervals"], list(u["labels"])
        rb, eb = iv[:-1, 1], piv[:-1, 1]
        m5, m10 = match_pairs(rb, eb, 5.0), match_pairs(rb, eb, 10.0)
        for i, t in enumerate(rb):
            prev, new = lab[i], lab[i + 1]
            kind = "raga-raga" if "other" not in (prev, new) else ("raga-to-other" if new == "other" else "other-to-raga")
            sh = len(scales[prev] & scales[new]) if kind == "raga-raga" else np.nan
            seg_end = iv[i + 1, 1]
            probe = min(t + 30.0, (t + seg_end) / 2 + 15.0)  # a point well inside the new segment
            ok_after = bool(label_at(piv, plab, np.array([probe]))[0] == new) if new != "other" else np.nan
            R.append({"bench": u["bench"], "unit": u["id"], "cluster": u["cluster"], "t_s": t, "prev": prev, "new": new,
                      "kind": kind, "shared": sh, "hit5": i in m5, "hit10": i in m10,
                      "err_s": float(eb[m10[i]] - t) if i in m10 else np.nan, "label_ok_30s_after": ok_after})
        inv5, inv10 = set(m5.values()), set(m10.values())
        for j, t in enumerate(eb):
            a, b = plab[j], plab[j + 1]
            near = float(np.min(np.abs(rb - t))) if len(rb) else np.inf
            where = label_at(iv, lab, np.array([t]))[0]
            E.append({"bench": u["bench"], "unit": u["id"], "cluster": u["cluster"], "t_s": t, "pred_prev": a, "pred_next": b,
                      "shared": len(scales[a] & scales[b]), "tp5": j in inv5, "tp10": j in inv10, "dist_to_ref_s": near,
                      "ref_label_here": where})
    return pd.DataFrame(R), pd.DataFrame(E)


def cluster_boot(df: pd.DataFrame, col: str, key: str, levels: list[str], rng) -> pd.DataFrame:
    """Rate of df[col] per level of df[key] with concert-cluster bootstrap CI."""
    cl = df.cluster.unique()
    g = {c: df[df.cluster == c] for c in cl}
    out = []
    boots = []
    for _ in range(N_BOOT):
        s = pd.concat([g[c] for c in rng.choice(cl, len(cl))])
        boots.append(s.groupby(key)[col].mean().reindex(levels))
    B = pd.concat(boots, axis=1)
    for lv in levels:
        x = df[df[key] == lv][col]
        out.append({"level": lv, "n": len(x), "rate": float(x.mean()) if len(x) else np.nan,
                    "lo": float(np.nanpercentile(B.loc[lv], 2.5)) if len(x) else np.nan,
                    "hi": float(np.nanpercentile(B.loc[lv], 97.5)) if len(x) else np.nan})
    return pd.DataFrame(out)


def fig_shared(R, E, rng, tabs) -> None:
    r = R[(R.bench.str.startswith("B1")) & (R.kind == "raga-raga")].copy()
    r["bucket"] = r.shared.map(bucket)
    e = E[E.bench.str.startswith("B1")].copy()
    e["bucket"] = e.shared.map(bucket)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for col, lab, c in (("hit5", "within +-5 s", "#1b9e77"), ("hit10", "within +-10 s", "#7570b3")):
        t = cluster_boot(r, col, "bucket", BUCKETS, rng)
        t["metric"] = f"B1 reference boundaries found, {lab}"
        tabs.append(t)
        x = np.arange(len(BUCKETS))
        axs[0].errorbar(x, t.rate, yerr=[t.rate - t.lo, t.hi - t.rate], fmt="-o", color=c, capsize=3, label=lab)
    axs[0].set_xticks(range(len(BUCKETS))); axs[0].set_xticklabels([f"{b}\n(n={n})" for b, n in zip(BUCKETS, t.n)])
    axs[0].set_xlabel("svaras shared by the true neighbouring ragas"); axs[0].set_ylabel("reference boundaries found (recall)")
    axs[0].set_ylim(0, 1); axs[0].legend(frameon=False)
    t = cluster_boot(e, "tp5", "bucket", BUCKETS, rng)
    t["metric"] = "B1 estimated boundaries that are correct (+-5 s)"
    tabs.append(t)
    axs[1].errorbar(np.arange(len(BUCKETS)), t.rate, yerr=[t.rate - t.lo, t.hi - t.rate], fmt="-o", color="#d95f02", capsize=3)
    axs[1].set_xticks(range(len(BUCKETS))); axs[1].set_xticklabels([f"{b}\n(n={n})" for b, n in zip(BUCKETS, t.n)])
    axs[1].set_xlabel("svaras shared by the PREDICTED neighbouring ragas"); axs[1].set_ylabel("estimated boundaries that are correct (precision)")
    axs[1].set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(f"{ROOT}/results/figures/e6_error_vs_shared.png", dpi=150); plt.close(fig)


def fig_timing(R) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for bench, c in (("B1", "#1b9e77"), ("B2", "#7570b3")):
        x = R[R.bench.str.startswith(bench)].err_s.dropna()
        ax.hist(x, bins=np.arange(-10, 10.01, 1), alpha=0.55, color=c, density=True, label=f"{bench} (n={len(x)})")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("estimated minus reference boundary time (s), matched within +-10 s"); ax.set_ylabel("density")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(f"{ROOT}/results/figures/e6_timing_error.png", dpi=150); plt.close(fig)


def fig_worst(units, preds, R, E) -> list[dict]:
    """All 8 B2 concerts (fewer than the 10 the plan asks for), worst boundary F@5 first."""
    b2 = [u for u in units if u["bench"] == "B2"]
    rows = []
    for u in b2:
        r, e = R[R.unit == u["id"]], E[E.unit == u["id"]]
        tp, ne, nr = int(r.hit5.sum()), len(e), len(r)
        p, rc = (tp / ne if ne else 0), (tp / nr if nr else 0)
        rows.append({"concert": u["id"], "F5": 2 * p * rc / (p + rc) if p + rc else 0.0, "tp": tp, "n_est": ne, "n_ref": nr})
    rows.sort(key=lambda d: d["F5"])
    m = load_model("w2")
    cmap = plt.get_cmap("tab20")
    for rank, d in enumerate(rows, 1):
        u = next(x for x in b2 if x["id"] == d["concert"])
        pr = preds[KEY + u["id"]]
        p = sequence_posteriors(m, "w2", u["cents"].astype(float), u["voiced"])
        top = np.argsort(-p["post"].mean(0))[:3]
        names = list(dict.fromkeys(list(u["labels"]) + pr["labels"]))
        col = {n: cmap(i % 20) for i, n in enumerate(names)}
        col["other"] = (0.8, 0.8, 0.8, 1)
        T = len(u["cents"]) / 100.0
        fig, axs = plt.subplots(4, 1, figsize=(13, 7.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.6, 0.6, 0.6]})
        c = u["cents"].astype(float)
        pc = np.mod(c, 1200)
        t = np.arange(len(c)) / 100.0
        ok = ~np.isnan(pc) & u["voiced"].astype(bool)
        axs[0].hist2d(t[ok], pc[ok], bins=[int(T // 30), 24], range=[[0, T], [0, 1200]], cmap="Greys", cmin=1)
        axs[0].set_ylabel("pitch class (cents above Sa)")
        for k in top:
            axs[1].plot(p["centres"][p["valid"]], p["post"][p["valid"], k], lw=0.8, label=m["ragas"][k])
        axs[1].set_ylabel("window posterior"); axs[1].legend(frameon=False, ncol=3, fontsize=8, title="3 ragas with highest mean posterior", title_fontsize=8)
        for ax, iv, lab, name in ((axs[2], u["intervals"], u["labels"], "reference"), (axs[3], np.array(pr["intervals"]), pr["labels"], "D3")):
            for (a, b), l in zip(iv, lab):
                ax.barh(0, b - a, left=a, color=col[l], edgecolor="white", height=1)
                if b - a > T / 40:
                    ax.text((a + b) / 2, 0, l, ha="center", va="center", fontsize=7)
            ax.set_yticks([]); ax.set_ylabel(name, rotation=0, ha="right", va="center")
        for b in u["intervals"][:-1, 1]:
            for ax in axs:
                ax.axvline(b, color="k", lw=0.5, ls=":")
        axs[3].set_xlabel("time in concatenated concert (s)"); axs[3].set_xlim(0, T)
        fig.tight_layout()
        slug = re.sub(r"[^A-Za-z0-9]+", "_", d["concert"])
        fig.savefig(f"{ROOT}/results/figures/e6_worst_b2_{rank}_{slug}.png", dpi=110); plt.close(fig)
        d["rank"] = rank
    return rows


def md_table(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    L = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for r in df.itertuples(index=False):
        L.append("| " + " | ".join(fmt.get(c, lambda v: f"{v:.3f}" if isinstance(v, float) else str(v))(v) for c, v in zip(df.columns, r)) + " |")
    return "\n".join(L)


def main() -> None:
    cfg_rng = np.random.default_rng(14)
    scales = load_scales()
    units = rf.load_units()
    preds = json.load(gzip.open(f"{ROOT}/results/e2_predictions.json.gz", "rt"))
    R, E = boundary_tables(units, preds, scales)
    R.to_csv(f"{ROOT}/results/e6_ref_boundaries.csv", index=False)
    E.to_csv(f"{ROOT}/results/e6_est_boundaries.csv", index=False)
    tabs: list[pd.DataFrame] = []
    fig_shared(R, E, cfg_rng, tabs)
    fig_timing(R)
    by = pd.concat(tabs)
    by.to_csv(f"{ROOT}/results/e6_by_shared.csv", index=False)
    # rāga pairs (B1 + B2 raga-raga), unordered
    rr = R[R.kind == "raga-raga"].copy()
    rr["pair"] = [" / ".join(sorted([a, b])) for a, b in zip(rr.prev, rr.new)]
    pairs = rr.groupby("pair").agg(n=("hit5", "size"), shared=("shared", "first"), recall5=("hit5", "mean"), recall10=("hit10", "mean"),
                                   label_ok=("label_ok_30s_after", "mean"), concerts=("cluster", "nunique")).reset_index()
    pairs.sort_values(["recall5", "n"], ascending=[True, False]).to_csv(f"{ROOT}/results/e6_pairs.csv", index=False)
    # B2 boundary kinds and where the false boundaries fall
    b2r = R[R.bench == "B2"]
    kinds = b2r.groupby("kind").agg(n=("hit5", "size"), recall5=("hit5", "mean"), recall10=("hit10", "mean")).reset_index()
    kinds.to_csv(f"{ROOT}/results/e6_b2_kinds.csv", index=False)
    b2e = E[E.bench == "B2"].copy()
    b2e["where"] = np.where(b2e.tp5, "correct (+-5 s)", np.where(b2e.tp10, "near miss (5-10 s)",
                                                np.where(b2e.ref_label_here == "other", "false, inside an `other` item", "false, inside an in-set item")))
    # B2 merges neighbouring items with the same label, so an `other` block can hold several items: how many of the
    # estimated boundaries inside `other` blocks sit on a real (unscored) item change?
    item_b = {u["id"]: u["item_intervals"][:-1, 1] for u in units if u["bench"] == "B2"}
    b2e["near_item_change"] = [bool(np.min(np.abs(item_b[a] - t)) <= 5.0) if len(item_b[a]) else False for a, t in zip(b2e.unit, b2e.t_s)]
    blocks = []
    for u in units:
        if u["bench"] != "B2":
            continue
        for (a, b), l in zip(u["intervals"], u["labels"]):
            if l == "other":
                n_items = int(((u["item_intervals"][:, 0] >= a - 1e-6) & (u["item_intervals"][:, 1] <= b + 1e-6)).sum())
                blocks.append({"concert": u["id"], "start_s": a, "dur_s": b - a, "items_in_block": n_items})
    blocks = pd.DataFrame(blocks)
    in_other = b2e[(~b2e.tp5) & (~b2e.tp10) & (b2e.ref_label_here == "other")]
    other_note = pd.DataFrame([{"other blocks": len(blocks), "blocks with >=2 items": int((blocks.items_in_block >= 2).sum()),
                                "false boundaries inside other": len(in_other),
                                "...of which within 5 s of a real item change": int(in_other.near_item_change.sum())}])
    where = b2e.groupby("where").size().rename("n").reset_index()
    where["share"] = where.n / where.n.sum()
    b1e = E[E.bench.str.startswith("B1")]
    b1where = pd.DataFrame({"where": ["correct (+-5 s)", "near miss (5-10 s)", "false (>10 s from any reference boundary)"],
                            "n": [int(b1e.tp5.sum()), int((b1e.tp10 & ~b1e.tp5).sum()), int((~b1e.tp10).sum())]})
    b1where["share"] = b1where.n / b1where.n.sum()
    rows = fig_worst(units, preds, R, E)
    # summary numbers
    b1 = R[R.bench.str.startswith("B1")]
    lines = ["# E6 auto-generated tables (numbers behind results/e6_errors.md)", "",
             f"Produced by: `python scripts/12_error_analysis.py` | commit {git_hash()} | W2 + D3, L=30 (frozen E2 config), test split, saved predictions", "",
             "## Recall of reference boundaries (B1 + B2)", "",
             md_table(pd.DataFrame([{"set": s, "n": len(g), "found@5": g.hit5.mean(), "found@10": g.hit10.mean()}
                                    for s, g in (("B1 all", b1), ("B1 raga-raga, shared<=4 (easy-like)", b1[b1.shared <= 4]),
                                                 ("B1 raga-raga, shared>=6 (hard-like)", b1[b1.shared >= 6]), ("B2 all", b2r))])), "",
             "## By shared svaras (concert-cluster bootstrap 95% CI)", "", md_table(by.rename(columns={"level": "shared svaras"})[["metric", "shared svaras", "n", "rate", "lo", "hi"]]), "",
             "## B2 reference boundaries by kind", "", md_table(kinds), "",
             "## B2 estimated boundaries by where they fall", "", md_table(where), "",
             "## B2 `other` blocks (neighbouring same-label items are merged in the reference)", "", md_table(other_note), "",
             "## B1 estimated boundaries by distance to the nearest reference boundary", "", md_table(b1where), "",
             "## Timing error of matched boundaries (est - ref, s; matched within +-10 s)", "",
             md_table(pd.DataFrame([{"set": s, "n": int(g.err_s.notna().sum()), "median": g.err_s.median(), "mean": g.err_s.mean(),
                                     "share_within_5s": float((g.err_s.abs() <= 5).sum() / max(g.err_s.notna().sum(), 1))}
                                    for s, g in (("B1", b1), ("B2", b2r))])), "",
             "## Label correctness 30 s after a found boundary (B1, B2 in-set targets)", "",
             md_table(pd.DataFrame([{"set": s, "boundary found@5": g[g.hit5].label_ok_30s_after.astype(float).mean(),
                                     "boundary missed@5": g[~g.hit5].label_ok_30s_after.astype(float).mean()}
                                    for s, g in (("B1", b1), ("B2", b2r[b2r.new != "other"]))])), "",
             "## Weakest rāga pairs (B1+B2 raga-raga boundaries, n>=6, lowest recall@5 first)", "",
             md_table(pairs[pairs.n >= 6].sort_values(["recall5", "n"], ascending=[True, False]).head(12)), "",
             "## Strongest rāga pairs (n>=6)", "", md_table(pairs[pairs.n >= 6].sort_values(["recall5", "n"], ascending=[False, False]).head(8)), "",
             "## B2 concerts, worst first (boundary F@5)", "", md_table(pd.DataFrame(rows)[["rank", "concert", "F5", "tp", "n_est", "n_ref"]]), ""]
    open(f"{ROOT}/results/e6_tables.md", "w", encoding="utf8").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
