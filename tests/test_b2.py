import json
import os

import numpy as np
import pandas as pd
import pytest

from raga_tracking.bench.concerts import content_hash, render, trim_unvoiced
from raga_tracking.io import ROOT

MAN = f"{ROOT}/manifests/b2_manifest.json"


def test_trim_unvoiced():
    v = np.zeros(10000, bool)
    v[2000:8000] = True
    (s, e), = trim_unvoiced(v)
    assert 15 <= s <= 25 and 75 <= e <= 85
    assert trim_unvoiced(np.zeros(10000, bool)) == [(0.0, 100.0)]  # nothing voiced -> keep whole (caller filters)


@pytest.mark.skipif(not os.path.exists(MAN), reason="B2 not built")
def test_manifest_order_split_and_rebuild():
    m = json.load(open(MAN))["concerts"]
    inv = pd.read_csv(f"{ROOT}/data/processed/inventory.csv", keep_default_na=False).set_index("rec_id")
    sp = pd.read_csv(f"{ROOT}/data/processed/splits.csv", keep_default_na=False).set_index("rec_id")
    for e in m:
        orders = [i["order"] for i in e["items"]]
        assert orders == sorted(orders) and len(set(orders)) == len(orders)           # performance order
        for i in e["items"]:
            assert int(float(inv.loc[i["rec_id"], "item_order"])) == i["order"]       # matches the metadata
            assert sp.loc[i["rec_id"], "split"] == "test" and inv.loc[i["rec_id"], "concert"] == e["concert"]
        ref = e["reference"]
        assert all(r["end"] > r["start"] for r in ref)                                # no zero-length segments
        assert all(a["label"] != b["label"] for a, b in zip(ref[:-1], ref[1:]))       # merged neighbours differ
        assert content_hash(render(e)) == e["content_sha256"]
    assert len(m) >= 8 and sum(e["n_boundaries"] for e in m) >= 40
