import json
import os

import numpy as np
import pandas as pd
import pytest

from raga_tracking.bench.synthetic import (content_hash, label_sequence, melodic_intervals, pick_excerpt, render)
from raga_tracking.io import ROOT

MAN = f"{ROOT}/data/processed/b1/b1_manifest.json"
SC = {"a": {0, 2, 4, 5, 7, 9, 11}, "b": {0, 2, 4, 5, 7, 9, 10}, "c": {0, 1, 3, 5, 7, 8, 10}, "d": {0, 2, 4, 7, 9}}


def test_labels():
    assert label_sequence(["a", "b"], SC)[0] == "hard"     # shares 6
    assert label_sequence(["a", "c"], SC)[0] == "easy"     # shares 3
    assert label_sequence(["a", "d"], SC)[0] == "mid"      # shares 5
    assert label_sequence(["a", "d"], SC, hard_min=5)[0] == "hard"
    assert label_sequence(["c", "d", "a"], SC)[1] == [2, 5]


def test_pick_excerpt_respects_voicing_and_bounds():
    rng = np.random.default_rng(0)
    v = np.ones(30000, bool)
    v[10000:] = False
    s, e = pick_excerpt(rng, np.zeros(30000), v, [(0.0, 300.0)], 60, 240, 0.6) or (None, None)
    assert s is not None and 60 <= e - s <= 240 and v[int(s * 100):int(e * 100)].mean() >= 0.6
    assert pick_excerpt(rng, np.zeros(30000), np.zeros(30000, bool), [(0.0, 300.0)], 60, 240, 0.6) is None
    assert pick_excerpt(rng, np.zeros(3000), np.ones(3000, bool), [(0.0, 30.0)], 60, 240, 0.6) is None


@pytest.mark.skipif(not os.path.exists(MAN), reason="B1 not built (data/ is not in git)")
def test_manifest_properties_and_rebuild():
    m = json.load(open(MAN))["sequences"]
    sp = pd.read_csv(f"{ROOT}/data/processed/splits.csv", keep_default_na=False).set_index("rec_id")
    train_groups = set(sp[sp.split == "train"].group)
    for e in m:
        assert e["split"] in ("val", "test")
        ragas = [x["raga"] for x in e["excerpts"]]
        assert 2 <= len(ragas) <= 5 and len(set(ragas)) == len(ragas)           # neighbouring (indeed all) ragas differ
        for x in e["excerpts"]:
            assert x["end_s"] - x["start_s"] >= 60 - 1e-6                       # all segments >= 60 s
            assert sp.loc[x["rec_id"], "split"] == e["split"]                   # sources come from the sequence's own split
            assert sp.loc[x["rec_id"], "group"] not in train_groups             # no source artist in train
            assert sp.loc[x["rec_id"], "concert"] == e["concert"]               # same concert
    for e in m[:: max(1, len(m) // 25)]:                                        # rebuild a spread of sequences
        assert content_hash(render(e)) == e["content_sha256"]


@pytest.mark.skipif(not os.path.exists(MAN), reason="B1 not built")
def test_enough_easy_and_hard_test_sequences():
    m = json.load(open(MAN))["sequences"]
    t = [e["label"] for e in m if e["split"] == "test"]
    assert t.count("easy") >= 100 and t.count("hard") >= 100
