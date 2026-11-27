"""T3: cache cents + voiced mask at 100 Hz for every usable recording, then Sa/Pa sanity checks.

Command: python scripts/03_cache_pitch.py
Writes: data/interim/pitch/{safe_rec_id}.npz (cents float32 NaN=unvoiced, voiced bool, t0, fs),
        results/t3_pitchclass_check.csv, results/figures/t3_sanity.png
"""
from __future__ import annotations

import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/src")
from raga_tracking.io import read_pitch_file, read_tonic_file, recording_paths, safe_name  # noqa: E402
from raga_tracking.pitch import FS, hz_to_cents, pitch_class, resample_to_grid  # noqa: E402

CACHE = f"{ROOT}/data/interim/pitch"
SA_MIN = 0.07  # ~ flat-distribution level (81 cents / 1200)


def cache_one(rec_id: str, pitch_path: str, tonic_path: str) -> str:
    t, f = read_pitch_file(pitch_path)
    cents, voiced = hz_to_cents(f, read_tonic_file(tonic_path))
    r = resample_to_grid(t, cents, voiced, FS)
    dest = f"{CACHE}/{safe_name(rec_id)}.npz"
    np.savez_compressed(dest, cents=r["cents"].astype(np.float32), voiced=r["voiced"], t0=r["t0"], fs=r["fs"])
    return dest


def pc_stats(rec_id: str) -> dict:
    """Mass of the pitch-class histogram within +-40 cents of Sa and Pa, and the argmax pitch class."""
    z = np.load(f"{CACHE}/{safe_name(rec_id)}.npz")
    pc = pitch_class(z["cents"][z["voiced"]])
    h, _ = np.histogram(pc, bins=120, range=(0, 1200))
    h = h / h.sum()
    near = lambda c: h[[int(((c + k) % 1200) // 10) for k in range(-40, 41, 10)]].sum()  # noqa: E731
    centre = 10 * np.argmax(h) + 5
    return dict(rec_id=rec_id, sa_mass=near(0), pa_mass=near(702), argmax_pc=centre,
                voiced_frac=float(z["voiced"].mean()), dur_s=len(z["voiced"]) / FS)


def main() -> None:
    os.makedirs(CACHE, exist_ok=True)
    sp = pd.read_csv(f"{ROOT}/data/processed/splits.csv", keep_default_na=False)
    need = set(sp[(sp.raga != "") & ~sp.dup].rec_id)
    paths = recording_paths()
    missing = sorted(need - set(paths))
    print(f"{len(need)} usable recordings, {len(missing)} without paths")
    for i, rid in enumerate(sorted(need & set(paths))):
        cache_one(rid, *paths[rid])
        if i % 100 == 0:
            print(i, flush=True)
    rows = [pc_stats(r) for r in sorted(need & set(paths))]
    df = pd.DataFrame(rows)
    df.to_csv(f"{ROOT}/results/t3_pitchclass_check.csv", index=False)
    bad = df[df.sa_mass < SA_MIN].rec_id.tolist()
    import yaml
    yaml.safe_dump({"rule": f"sa_mass(+-40c) < {SA_MIN} after tonicFine (no evidence of a tonic)", "excluded": bad},
                   open(f"{ROOT}/configs/excluded_recordings.yaml", "w"))
    print(len(bad), "recordings excluded (Sa mass <", SA_MIN, ")")
    print(df[["sa_mass", "pa_mass", "voiced_frac", "dur_s"]].describe().round(3).to_string())
    plot(df)


def plot(df: pd.DataFrame) -> None:
    rng = np.random.default_rng(0)
    pick = rng.choice(df.rec_id.to_numpy(), 3, replace=False)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
    for a, rid in zip(ax, pick):
        z = np.load(f"{CACHE}/{safe_name(rid)}.npz")
        pc = pitch_class(z["cents"][z["voiced"]])
        a.hist(pc, bins=120, range=(0, 1200), density=True, color="#4C72B0")
        for c, n in ((0, "Sa"), (702, "Pa")):
            a.axvline(c, color="#C44E52", ls="--", lw=1)
            a.text(c + 12, a.get_ylim()[1] * 0.92 if False else 0.0, n, color="#C44E52")
        a.set_title(rid.split(":")[0] + ":" + rid.split(":")[-1][:24], fontsize=9)
        a.set_xlabel("pitch class (cents above tonic)")
    ax[0].set_ylabel("density (per cent)")
    fig.tight_layout()
    os.makedirs(f"{ROOT}/results/figures", exist_ok=True)
    fig.savefig(f"{ROOT}/results/figures/t3_sanity.png", dpi=130)
    print("plotted", list(pick))


if __name__ == "__main__":
    main()
