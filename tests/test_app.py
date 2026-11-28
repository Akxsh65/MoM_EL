"""Tests for the interactive explorer's backend (no trained models or data needed)."""
import importlib.util
import json
import os
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("serve", f"{ROOT}/app/serve.py")
serve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(serve)


def _fake(key="t"):
    """3 ragas, 120 windows (hop 5 s, centres 2.5+5i): raga 0 until 300 s, then raga 1; one 1-window flip at 100 s."""
    c = 2.5 + 5.0 * np.arange(120)
    post = np.full((120, 3), 0.05)
    lab = (c > 300).astype(int)
    lab[(c > 100) & (c < 105)] = 2
    post[np.arange(120), lab] = 0.9
    post /= post.sum(1, keepdims=True)
    serve.ITEMS[key] = {"name": key, "cents": np.zeros(60000), "voiced": np.ones(60000, bool), "ref_iv": np.array([[0.0, 300.0], [300.0, 600.0]]),
                        "ref_lab": ["a", "b"], "note": "x"}
    serve.ANALYSES[(key, "m")] = {"post": post, "valid": np.ones(120, bool), "centres": c, "hists": np.full((120, 12), 1 / 12), "ragas": ["a", "b", "c"],
                                  "L": 30.0, "hop": 5.0, "heat": np.ones((6, 24), int), "tb": 100.0}


PRM = {"eps": 0.01, "min_dur_s": 30, "beta": 1.0, "d1_width": 5, "d2_thr": 0.2}


def test_d3_removes_flip_and_finds_the_true_boundary():
    _fake()
    r = serve.decode("t", "m", "D3", PRM)
    assert r["metrics"]["D3"]["n_pred"] == 1 and r["metrics"]["D3"]["F5"] == 1.0
    assert [s["raga"] for s in r["segments"]] == ["a", "b"]
    assert r["metrics"]["D0"]["n_pred"] > 1          # argmax keeps the flip (extra boundaries)
    assert r["metrics"]["D0"]["F5"] < 1.0


def test_metrics_follow_the_sliders():
    _fake()
    strict = serve.decode("t", "m", "D3", {**PRM, "min_dur_s": 120})["metrics"]["D3"]
    loose = serve.decode("t", "m", "D3", {**PRM, "min_dur_s": 10})["metrics"]["D3"]
    assert strict["n_pred"] <= loose["n_pred"]


def test_online_is_only_computed_when_selected():
    _fake()
    assert "D3-online" not in serve.decode("t", "m", "D3", PRM)["metrics"]
    assert "D3-online" in serve.decode("t", "m", "D3-online", PRM)["metrics"]


def test_upload_parsing_and_validation():
    t = np.arange(0, 60, 0.01)
    txt = "\n".join(f"{a:.3f} 220.0" for a in t)
    it = serve.parse_upload({"pitch_text": txt, "tonic_hz": 220})
    assert it["ref_iv"] is None and abs(np.nanmean(it["cents"])) < 1e-6 and it["voiced"].all()
    with pytest.raises(ValueError):
        serve.parse_upload({"pitch_text": txt, "tonic_hz": 5})
    with pytest.raises(ValueError):
        serve.parse_upload({"pitch_text": "0 220\n0.01 220", "tonic_hz": 220})


def test_http_endpoints_and_page():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), serve.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "Raga Tracking Explorer" in page and "cdn" not in page.lower()
        cfg = json.loads(urllib.request.urlopen(base + "/api/config").read())
        assert "models" in cfg
        _fake("http")
        req = urllib.request.Request(base + "/api/decode", data=json.dumps({"key": "http", "model": "m", "decoder": "D3", "params": PRM}).encode(), headers={"Content-Type": "application/json"})
        assert json.loads(urllib.request.urlopen(req).read())["metrics"]["D3"]["F5"] == 1.0
        bad = urllib.request.Request(base + "/api/decode", data=b"{}", headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(bad)
    finally:
        srv.shutdown()
