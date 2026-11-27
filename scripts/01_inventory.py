"""T1: build data/processed/inventory.csv and results/inventory_summary.md.

Command: python scripts/01_inventory.py
Sources: CMD Carnatic features (data/raw/cmd) and Saraga Carnatic 1.5 metadata (data/raw/saraga).
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
import unicodedata

import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CMD = os.path.join(ROOT, "data/raw/cmd/RagaDataset/Carnatic")
SAR = os.path.join(ROOT, "data/raw/saraga/saraga1.5_carnatic")
ALIASES = os.path.join(ROOT, "configs/raga_aliases.yaml")
FAILS: list[str] = []


def norm(s: str) -> str:
    """Lowercase ASCII key: strip diacritics and every non-alphanumeric character."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", s.lower())


def load_aliases() -> dict[str, str]:
    """alias key -> canonical key, from configs/raga_aliases.yaml (canonical: [alias, ...])."""
    if not os.path.exists(ALIASES):
        return {}
    d = yaml.safe_load(open(ALIASES, encoding="utf8")) or {}
    out = {}
    for canon, al in d.items():
        out[norm(canon)] = norm(canon)
        for a in al or []:
            out[norm(a)] = norm(canon)
    return out


def read_pitch(path: str) -> tuple[float, float] | None:
    """Return (duration_s, hop_s) or None if the file does not parse."""
    try:
        a = pd.read_csv(path, sep=r"\s+", header=None, dtype=float).to_numpy()
        if a.shape[1] != 2 or len(a) < 2:
            raise ValueError(f"bad shape {a.shape}")
        return float(a[-1, 0]), float(np.median(np.diff(a[:, 0])))
    except Exception as e:  # logged, never silent
        FAILS.append(f"{path}: {e}")
        return None


def cmd_rows() -> list[dict]:
    names = json.load(open(f"{CMD}/_info_/ragaId_to_ragaName_mapping.json", encoding="utf8"))
    mbids = {}
    for v in json.load(open(f"{CMD}/_info_/path_mbid_ragaid.json")).values():
        pp = v["path"].split("/")
        mbids[(pp[3], pp[4], pp[5], pp[6])] = v["mbid"]
    rows = []
    for p in sorted(glob.glob(f"{CMD}/features/*/*/*/*/*.pitch")):
        rel = os.path.relpath(p, f"{CMD}/features").replace("\\", "/").split("/")
        ragaid, artist, concert, item = rel[:4]
        base = p[: -len(".pitch")]
        pr = read_pitch(p)
        rows.append(dict(
            rec_id=f"cmd:{ragaid[:8]}:{artist}:{concert}:{item}", source="cmd", artist=artist.replace("_", " ").strip(),
            concert=concert, item_order=np.nan, raga_raw=names.get(ragaid, ragaid), n_ragas=1,
            duration_s=pr[0] if pr else np.nan, hop_s=pr[1] if pr else np.nan, has_pitch=pr is not None,
            has_tonic=os.path.exists(base + ".tonic"), has_sections=False, ragamalika_flag=False,
            mbid=mbids.get((ragaid, artist, concert, item), "")))
    return rows


def mb_order() -> dict[str, int]:
    """recording mbid -> 1-based track position in its MusicBrainz release (cached by 01b_fetch_item_order.py)."""
    out = {}
    for f in glob.glob(f"{ROOT}/data/raw/musicbrainz/*.json"):
        r = json.load(open(f, encoding="utf8"))
        n = 0
        for m in r.get("media", []):
            for t in m.get("tracks", []):
                n += 1
                out[t["recording"]["id"]] = n
    return out


def saraga_rows() -> list[dict]:
    order = mb_order()
    rows = []
    for j in sorted(glob.glob(f"{SAR}/*/*/*.json")):
        d = json.load(open(j, encoding="utf8"))
        base = j[: -len(".json")]
        artists = [a["artist"]["name"] for a in d.get("artists", []) if a.get("lead")] or \
                  [a["artist"]["name"] for a in d.get("album_artists", [])]
        ragas = [r.get("common_name") or r["name"] for r in d.get("raaga", [])]
        forms = [f.get("name", "") for f in d.get("form", [])]
        pr = read_pitch(base + ".pitch.txt") if os.path.exists(base + ".pitch.txt") else None
        secs = glob.glob(base + ".sections-manual*.txt")
        rows.append(dict(
            rec_id=f"sar:{d['mbid']}", source="saraga", artist=artists[0] if artists else "",
            concert=(d.get("concert") or [{}])[0].get("title", os.path.basename(os.path.dirname(os.path.dirname(j)))),
            item_order=order.get(d["mbid"], np.nan), raga_raw="|".join(ragas), n_ragas=len(ragas),
            duration_s=pr[0] if pr else d.get("length", 0) / 1000.0, hop_s=pr[1] if pr else np.nan,
            has_pitch=pr is not None, has_tonic=os.path.exists(base + ".ctonic.txt"),
            has_sections=bool(secs), mbid=d["mbid"],
            ragamalika_flag=any("malika" in norm(x) for x in forms + ragas) or len(ragas) > 1))
    return rows


def main() -> None:
    al = load_aliases()
    df = pd.DataFrame(cmd_rows() + saraga_rows())
    df["raga"] = [al.get(norm(r.split("|")[0]), norm(r.split("|")[0])) if r else "" for r in df.raga_raw]
    df.loc[df.raga == "ragamalika", "raga"] = ""  # ragamalika is not a single raga
    df["dup_in_other_source"] = df.mbid.duplicated(keep=False) & (df.mbid != "")
    df["artist_key"] = df.artist.map(norm)
    cols = ["rec_id", "source", "artist", "concert", "item_order", "raga_raw", "raga", "n_ragas", "duration_s",
            "hop_s", "has_pitch", "has_tonic", "has_sections", "ragamalika_flag", "mbid", "dup_in_other_source", "artist_key"]
    os.makedirs(f"{ROOT}/data/processed", exist_ok=True)
    df[cols].to_csv(f"{ROOT}/data/processed/inventory.csv", index=False)
    write_summary(df)
    print(f"{len(df)} recordings; {len(FAILS)} parse failures")
    for f in FAILS:
        print("FAIL", f)


def write_summary(df: pd.DataFrame) -> None:
    L = ["# Inventory summary", "", "Produced by: `python scripts/01_inventory.py`", ""]
    for s, g in df.groupby("source"):
        hops = g.hop_s.dropna()
        L.append(f"- **{s}**: {len(g)} recordings, pitch ok {int(g.has_pitch.sum())}, tonic {int(g.has_tonic.sum())} "
                 f"({g.has_tonic.mean():.1%}), sections {int(g.has_sections.sum())}, "
                 f"hop min/max {hops.min():.6f}/{hops.max():.6f} s, artists {g.artist_key.nunique()}, concerts {g.concert.nunique()}")
    L += ["", f"Parse failures: {len(FAILS)}", ""]
    L += ["## Recordings and artists per raga (pooled, normalised key)", "", "| raga | recordings | artists | cmd | saraga |", "|---|---|---|---|---|"]
    r = df[df.raga != ""].groupby("raga").agg(n=("rec_id", "size"), a=("artist_key", "nunique"),
                                             c=("source", lambda x: (x == "cmd").sum()), s=("source", lambda x: (x == "saraga").sum()))
    for k, v in r.sort_values("n", ascending=False).iterrows():
        L.append(f"| {k} | {v.n} | {v.a} | {v.c} | {v.s} |")
    cand = df[df.ragamalika_flag]
    L += ["", f"## Ragamalika candidates: {len(cand)}", ""]
    for _, c in cand.iterrows():
        L.append(f"- {c.rec_id} | {c.artist} | {c.concert} | ragas: {c.raga_raw} | sections: {c.has_sections}")
    open(f"{ROOT}/results/inventory_summary.md", "w", encoding="utf8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
