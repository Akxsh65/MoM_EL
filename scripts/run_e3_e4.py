"""T12 / E3 + E4: window length and online latency on the TEST benchmarks. ONE run per frozen config.

Command: python scripts/run_e3_e4.py
Refuses to run with uncommitted configs/src/scripts/tests/manifests or if results/e3_window_length.csv exists.
Uses configs/window_length_frozen.yaml (D3 tuned per L on B1-val by scripts/11_tune_window_length.py) with the W2 model
trained at each L. Per L: offline D0/D1/D3 on B1-easy/mid/hard and B2 (boundary/frame metrics with bootstrap CIs) and
online latency (D3 forward filtering, and D0 which is causal as it is) at every reference boundary.
Latency uses window END time as the moment a prediction is available (centre + L/2); see src/raga_tracking/eval/latency.py.
Writes results/e3_window_length.csv|md, results/e4_latency.csv|md, results/e4_latency_per_boundary.csv,
results/figures/e3_f_vs_L.png, results/figures/e4_latency_cdf_by_L.png, results/figures/e4_latency_cdf_L30.png
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import sys

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/src")
from raga_tracking.decode import decode_d0, decode_d1, decode_d3  # noqa: E402
from raga_tracking.decode.hmm import emissions, forward_filter  # noqa: E402
from raga_tracking.eval.bootstrap import COLS, METRICS, ci_table, draws  # noqa: E402
from raga_tracking.eval.latency import detection_latencies, summarize  # noqa: E402
from raga_tracking.features.dataset import load_cfg  # noqa: E402
from raga_tracking.models.posterior import load_model, sequence_posteriors  # noqa: E402

spec = importlib.util.spec_from_file_location("rf", f"{ROOT}/scripts/run_final.py")
rf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rf)  # load_units (hash-checked test data), unit_counts, assert_frozen
LENGTHS = (10, 20, 30, 60)
BENCHES = ("B1-easy", "B1-hard", "B1-mid", "B2")
OUT = f"{ROOT}/results"  # --smoke redirects this to a scratch dir
N_BOOT, GRID = 1000, np.arange(0, 61, 1.0)


def load_val_units() -> list[dict]:
    """Smoke test only: B1-val sequences (hash-checked), same dict layout as run_final.load_units."""
    import json
    from raga_tracking.bench import synthetic as b1mod
    units = []
    for e in json.load(open(f"{ROOT}/manifests/b1_manifest.json"))["sequences"]:
        if e["split"] != "val":
            continue
        z = np.load(f"{ROOT}/data/processed/b1/val/{e['seq_id']}.npz")
        s_ = {k: z[k] for k in ("cents", "voiced", "intervals", "labels")}
        assert b1mod.content_hash(s_) == e["content_sha256"], e["seq_id"]
        units.append({"bench": f"B1-{e['label']}", "id": e["seq_id"], "cluster": e["concert"], **s_})
    return units


def online_labels(post, valid, m_hops, eps, beta, names):
    em = emissions(post, valid, None, beta)
    return np.asarray(names)[forward_filter(em, m_hops, eps).argmax(1)]


def cdf(lat: np.ndarray) -> np.ndarray:
    """Share of ALL boundaries detected within x s (misses never count, so the curve plateaus at 1 - miss rate)."""
    ok = lat[~np.isnan(lat)]
    return (ok[None, :] <= GRID[:, None]).sum(1) / max(len(lat), 1)


def boot_latency(lat, unit_idx, n_units, mult):
    """Bootstrap median / miss rate / CDF over units (mult = (n_boot, n_units) multiplicities)."""
    med, miss, cd = [], [], []
    for w in mult[:, unit_idx]:
        sel = np.repeat(np.arange(len(lat)), w)
        s = summarize(lat[sel]) if len(sel) else {"median_s": np.nan, "miss_rate": np.nan}
        med.append(s["median_s"]); miss.append(s["miss_rate"]); cd.append(cdf(lat[sel]) if len(sel) else np.full(len(GRID), np.nan))
    pc = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]  # noqa: E731
    return pc(med), pc(miss), np.nanpercentile(np.array(cd), [2.5, 97.5], axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow_rerun", default=None)
    ap.add_argument("--smoke", default=None, help="DIR: run on B1-VAL (never test) and write there; skips the frozen-config guards")
    a = ap.parse_args()
    global OUT
    if a.smoke:
        OUT = a.smoke
        os.makedirs(OUT, exist_ok=True)
    if not a.smoke and os.path.exists(f"{ROOT}/results/e3_window_length.csv") and not a.allow_rerun:
        sys.exit("results/e3_window_length.csv exists: one test run per frozen config (CLAUDE.md rule 2).")
    dirty = "" if a.smoke else rf.sh("status", "--porcelain", "--", "configs", "src", "scripts", "tests", "manifests")
    if dirty:
        sys.exit("Refusing to run: uncommitted changes:\n" + dirty)
    commit = rf.sh("rev-parse", "--short", "HEAD")
    cfg = load_cfg()
    hop, seed = cfg["window"]["hop_s"], cfg["seed"]
    fzpath = f"{ROOT}/configs/window_length_frozen.yaml"
    fz = yaml.safe_load(open(fzpath))
    fz_hash = hashlib.sha256(open(fzpath, "rb").read()).hexdigest()[:12]
    units = load_val_units() if a.smoke else rf.load_units()
    uid = {u["id"]: i for i, u in enumerate(units)}
    off_rows, lat_rows = [], []
    for L in LENGTHS:
        f = fz[f"L{L}"]
        d3 = f["d3"]
        m_hops = max(1, int(round(d3["min_dur_s"] / hop)))
        name = f["model"]
        m = load_model(name)
        assert m["L"] == L
        names_all = np.array(m["ragas"])
        for i, u in enumerate(units):
            p = sequence_posteriors(m, name, u["cents"].astype(float), u["voiced"])
            dec = {"D0": decode_d0(p["post"], p["valid"], p["centres"], hop),
                   "D1": decode_d1(p["post"], p["valid"], p["centres"], hop, f["d1_width"]),
                   "D3": decode_d3(p["post"], p["valid"], p["centres"], hop, d3["eps"], d3["min_dur_s"], None, d3["beta"])}
            for d, (iv, lab) in dec.items():
                off_rows.append({"L": L, "decoder": d, "bench": u["bench"], "id": u["id"], "cluster": u["cluster"],
                                 **dict(zip(COLS, rf.unit_counts(u, iv, names_all[lab])))})
            avail = p["centres"] + L / 2.0
            online = {"D3-online": online_labels(p["post"], p["valid"], m_hops, d3["eps"], d3["beta"], names_all),
                      "D0-online": names_all[p["post"].argmax(1)]}
            for d, pred in online.items():
                for j, lt in enumerate(detection_latencies(u["intervals"], u["labels"], pred, avail)):
                    lat_rows.append({"L": L, "decoder": d, "bench": u["bench"], "id": u["id"], "cluster": u["cluster"],
                                     "boundary_idx": j, "latency_s": lt})
            if i % 100 == 0:
                print(f"L={L} unit {i}/{len(units)}", flush=True)
    per = pd.DataFrame(off_rows)
    lat = pd.DataFrame(lat_rows)
    lat.to_csv(f"{OUT}/e4_latency_per_boundary.csv", index=False)
    # consistency with E2 (not applicable in smoke mode) (W2, D0/D1/D3 at L=30 must reproduce e2_per_sequence exactly)
    e2 = pd.read_csv(f"{ROOT}/results/e2_per_sequence.csv")
    e2 = e2[e2.model.str.lower() == "w2"]
    ok_e2 = True
    for d in (() if a.smoke else ("D0", "D1", "D3")):
        x = per[(per.L == 30) & (per.decoder == d)].set_index("id")[list(COLS)].sort_index()
        y = e2[e2.decoder == d].set_index("id")[list(COLS)].sort_index()
        ok_e2 &= bool(np.allclose(x.to_numpy(float), y.to_numpy(float)))
    print("consistent with E2 at L=30:", ok_e2, flush=True)

    rng = np.random.default_rng(seed + 11)
    hdr = (f"Produced by: python scripts/run_e3_e4.py | commit {commit} | window_length_frozen.yaml sha256[:12] {fz_hash} | "
           f"W2 per L; bootstrap {N_BOOT} resamples (units = sequences, B2 = concerts; *_concert = concert-cluster) seed {seed + 11} | "
           f"offline L=30 rows reproduce E2 exactly: {'n/a (smoke, val)' if a.smoke else ok_e2}")
    out_off, out_lat, cdfs = [], [], {}
    for bench in [b for b in BENCHES if b in set(per.bench)]:
        ids = per[(per.bench == bench) & (per.L == 10) & (per.decoder == "D3")].id.tolist()
        clusters = per[(per.bench == bench) & (per.L == 10) & (per.decoder == "D3")].cluster.to_numpy()
        mult, multc = draws(len(ids), N_BOOT, rng), draws(len(ids), N_BOOT, rng, clusters)
        for L in LENGTHS:
            for d in ("D0", "D1", "D3"):
                g = per[(per.bench == bench) & (per.L == L) & (per.decoder == d)].set_index("id").loc[ids, list(COLS)].to_numpy(float)
                t, tc = ci_table(g, mult), ci_table(g, multc)
                row = {"L": L, "decoder": d, "bench": bench, "n_units": len(ids)}
                for k in METRICS:
                    row[k], row[k + "_lo"], row[k + "_hi"] = t[k]
                row["F5_lo_concert"], row["F5_hi_concert"] = tc["F5"][1], tc["F5"][2]
                out_off.append(row)
            for d in ("D3-online", "D0-online"):
                g = lat[(lat.bench == bench) & (lat.L == L) & (lat.decoder == d)]
                arr, ui = g.latency_s.to_numpy(float), g.id.map({k: j for j, k in enumerate(ids)}).to_numpy(int)
                s = summarize(arr)
                (m_lo, m_hi), (x_lo, x_hi), band = boot_latency(arr, ui, len(ids), mult)
                (mc_lo, mc_hi), (xc_lo, xc_hi), bandc = boot_latency(arr, ui, len(ids), multc)
                out_lat.append({"L": L, "decoder": d, "bench": bench, "n_boundaries": s["n"], "miss_rate": s["miss_rate"],
                                "miss_lo": x_lo, "miss_hi": x_hi, "miss_lo_concert": xc_lo, "miss_hi_concert": xc_hi,
                                "median_s": s["median_s"], "median_lo": m_lo, "median_hi": m_hi,
                                "median_lo_concert": mc_lo, "median_hi_concert": mc_hi, "q25_s": s["q25_s"], "q75_s": s["q75_s"]})
                cdfs[(bench, L, d)] = (cdf(arr), bandc)
    off, lt = pd.DataFrame(out_off), pd.DataFrame(out_lat)
    for path, frame in ((f"{OUT}/e3_window_length.csv", off), (f"{OUT}/e4_latency.csv", lt)):
        with open(path, "w", newline="") as fh:
            fh.write("# " + hdr + "\n")
            frame.to_csv(fh, index=False)
    write_md(off, lt, hdr, fz)
    plot(off, cdfs)


def fmt(r, k):
    return f"{r[k]:.3f} [{r[k + '_lo']:.3f}, {r[k + '_hi']:.3f}]"


def write_md(off, lt, hdr, fz) -> None:
    L = ["# E3: window length (offline) and E4: online latency", "", hdr, "",
         "W2 classifier retrained at each L (tau, bins from L=30; C re-tuned on val); D3 re-tuned per L on B1-val. D0/D1 have no tuned "
         "parameters. Cells: estimate [95% CI]. Latency = time from the reference boundary until the online prediction equals the new raga "
         "for 2 consecutive hops, where a hop is available at its window END; boundaries into `other` are not scored; no match within 60 s "
         "(or before the next reference boundary) is a miss. Median/IQR are over detected boundaries; misses are in the miss rate. "
         "CIs are sequence bootstrap (B2: concerts); B1 `concert` CIs resample whole concerts and are the honest ones.", "",
         "Tuned D3 per L (B1-val):", "", "| L (s) | eps | min_dur (s) | beta | B1-val D3 F@5 | B1-val D1 F@5 |", "|---|---|---|---|---|---|"]
    for Lh in LENGTHS:
        f = fz[f"L{Lh}"]
        L.append(f"| {Lh} | {f['d3']['eps']} | {f['d3']['min_dur_s']} | {f['d3']['beta']} | {f['val_F5']:.3f} | {f['val_D1_F5']:.3f} |")
    L.append("")
    for bench in [b for b in BENCHES if b in set(off.bench)]:
        L += [f"## {bench} offline", "", "| L (s) | decoder | F@5 | P@5 | R@5 | F@10 | frame acc | false sw./10 min | F@5 concert-CI |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in off[off.bench == bench].to_dict("records"):
            L.append(f"| {r['L']} | {r['decoder']} | {fmt(r, 'F5')} | {fmt(r, 'P5')} | {fmt(r, 'R5')} | {fmt(r, 'F10')} | "
                     f"{fmt(r, 'frame_acc')} | {fmt(r, 'false_sw_10min')} | [{r['F5_lo_concert']:.3f}, {r['F5_hi_concert']:.3f}] |")
        L += ["", f"## {bench} online latency", "",
              "| L (s) | decoder | boundaries | median (s) [concert CI] | IQR (s) | miss rate [concert CI] |", "|---|---|---|---|---|---|"]
        for r in lt[lt.bench == bench].to_dict("records"):
            L.append(f"| {r['L']} | {r['decoder']} | {r['n_boundaries']} | {r['median_s']:.1f} [{r['median_lo_concert']:.1f}, {r['median_hi_concert']:.1f}] | "
                     f"{r['q25_s']:.1f}-{r['q75_s']:.1f} | {r['miss_rate']:.3f} [{r['miss_lo_concert']:.3f}, {r['miss_hi_concert']:.3f}] |")
        L.append("")
    open(f"{OUT}/e3_e4_summary.md", "w", encoding="utf8").write("\n".join(L) + "\n")


def plot(off, cdfs) -> None:
    cols = {"B1-easy": "#1b9e77", "B1-hard": "#d95f02", "B2": "#7570b3"}
    fig, ax = plt.subplots(figsize=(6, 4))
    for bench, c in [(b, c) for b, c in cols.items() if b in set(off.bench)]:
        g = off[(off.bench == bench) & (off.decoder == "D3")].sort_values("L")
        ax.plot(g.L, g.F5, "-o", color=c, label=f"{bench} (D3)")
        ax.fill_between(g.L, g.F5_lo_concert, g.F5_hi_concert, color=c, alpha=0.15)
    ax.set_xlabel("window length L (s)"); ax.set_ylabel("boundary F-measure at +-5 s")
    ax.set_xticks(LENGTHS); ax.legend(frameon=False); ax.set_ylim(0, 0.8)
    fig.tight_layout(); fig.savefig(f"{OUT}/e3_f_vs_L.png", dpi=150); plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, bench in zip(axs, ("B1-easy", "B1-hard")):
        for L, c in zip(LENGTHS, ("#1b9e77", "#d95f02", "#7570b3", "#e7298a")):
            y, band = cdfs[(bench, L, "D3-online")]
            ax.plot(GRID, y, color=c, label=f"L = {L} s")
            ax.fill_between(GRID, band[0], band[1], color=c, alpha=0.12)
        ax.set_xlabel("time after reference boundary (s)"); ax.set_title(bench, fontsize=10)
    axs[0].set_ylabel("share of boundaries detected"); axs[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(f"{OUT}/e4_latency_cdf_by_L.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for bench, c in [(b, c) for b, c in cols.items() if b in set(off.bench)]:
        y, band = cdfs[(bench, 30, "D3-online")]
        ax.plot(GRID, y, color=c, label=bench)
        ax.fill_between(GRID, band[0], band[1], color=c, alpha=0.15)
    ax.set_xlabel("time after reference boundary (s)"); ax.set_ylabel("share of boundaries detected"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(f"{OUT}/e4_latency_cdf_L30.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
