"""Artist-disjointness and config sanity for T2. Skips if the data-derived files do not exist yet."""
import os

import pandas as pd
import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = os.path.join(ROOT, "data/processed/splits.csv")
needs_data = pytest.mark.skipif(not os.path.exists(SPLITS), reason="run scripts/02_splits.py first")


@needs_data
def test_artist_disjoint():
    d = pd.read_csv(SPLITS, keep_default_na=False)
    per = d.groupby("group").split.nunique()
    assert (per == 1).all(), f"artists in several splits: {list(per[per > 1].index)}"


@needs_data
def test_every_raga_in_all_splits_and_gate_g1():
    d = pd.read_csv(SPLITS, keep_default_na=False)
    rs = yaml.safe_load(open(os.path.join(ROOT, "configs/raga_set.yaml")))
    assert rs["K"] >= 15 and rs["K"] == len(rs["ragas"])
    u = d[d.in_set & ~d.dup]
    for s in ("train", "val", "test"):
        assert set(u[u.split == s].raga) == set(rs["ragas"]), s


@needs_data
def test_test_concerts():
    d = pd.read_csv(SPLITS, keep_default_na=False)
    t = d[(d.split == "test") & d.in_set & (d.source == "saraga")].groupby("concert").size()
    assert (t >= 3).sum() >= 8


def test_scales_cover_raga_set():
    p = os.path.join(ROOT, "configs/raga_set.yaml")
    if not os.path.exists(p):
        pytest.skip("no raga_set.yaml")
    rs = yaml.safe_load(open(p))["ragas"]
    sc = yaml.safe_load(open(os.path.join(ROOT, "configs/scales.yaml")))
    assert set(rs) <= set(sc), set(rs) - set(sc)
    for r in rs:
        s = sc[r]["svaras"]
        assert 0 in s
        assert all(0 <= x < 12 for x in s) and s == sorted(set(s))
