"""Interactive raga-tracking explorer (local web app, standard library server, real pipeline behind it).

Command: python app/serve.py [--port 8765] [--open]
Opens http://127.0.0.1:8765 . Pick a benchmark sequence, a cached recording or upload a pitch file, choose the classifier,
then move the decoder sliders: the repo's own decoders (src/raga_tracking/decode) re-run on the cached window posteriors and
the metrics (src/raga_tracking/eval/metrics.py) are recomputed. Nothing is written to results/ and no data leaves the machine.

Needs the trained models in models/ and the cached pitch / benchmark arrays in data/ (see README). Binds to 127.0.0.1 only.
Exploring settings on TEST sequences here is for understanding, not for choosing settings (CLAUDE.md rule 2).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{ROOT}/src")
from raga_tracking.decode import decode_d0, decode_d1, decode_d2, decode_d3  # noqa: E402
from raga_tracking.eval.metrics import boundary_counts, frame_accuracy  # noqa: E402
from raga_tracking.io import safe_name  # noqa: E402
from raga_tracking.models.posterior import load_model, sequence_posteriors  # noqa: E402
from raga_tracking.pitch import hz_to_cents, resample_to_grid  # noqa: E402

MODELS = [("w1", "W1 histogram, 30 s"), ("w2_L10", "W2 melody surface, 10 s"), ("w2_L20", "W2 melody surface, 20 s"),
          ("w2", "W2 melody surface, 30 s"), ("w2_L60", "W2 melody surface, 60 s")]
LOCK = threading.Lock()
ITEMS: dict[str, dict] = {}      # source key -> {name, cents, voiced, ref_iv, ref_lab, ref_note}
ANALYSES: dict[tuple, dict] = {}  # (source key, model) -> window posteriors etc.
MAX_UPLOAD = 60 * 2**20


def read_yaml(rel: str) -> dict:
    return yaml.safe_load(open(f"{ROOT}/{rel}", encoding="utf8"))


def frozen_for(model: str) -> dict:
    """Frozen D3/D2/D1 settings for a model id (from the committed configs)."""
    if model == "w1":
        z = read_yaml("configs/decoders_frozen.yaml")["w1"]
        return {"eps": z["d3"]["eps"], "min_dur_s": z["d3"]["min_dur_s"], "beta": z["d3"]["beta"], "d1_width": z["d1_width"], "d2_thr": z["d2"]["thr"]}
    wl = read_yaml("configs/window_length_frozen.yaml")
    for k, v in wl.items():
        if isinstance(v, dict) and v.get("model") == model:
            d2 = read_yaml("configs/decoders_frozen.yaml")["w2"]["d2"]["thr"]
            return {"eps": v["d3"]["eps"], "min_dur_s": v["d3"]["min_dur_s"], "beta": v["d3"]["beta"], "d1_width": v["d1_width"], "d2_thr": d2}
    return {"eps": 0.01, "min_dur_s": 60, "beta": 0.25, "d1_width": 5, "d2_thr": 0.2}


def config() -> dict:
    models = [{"id": m, "label": lab, "frozen": frozen_for(m)} for m, lab in MODELS if os.path.exists(f"{ROOT}/models/{m}.joblib")]
    return {"models": models, "hop_s": 5.0}


def sources() -> dict:
    out = {"b1": [], "b2": [], "rec": []}
    p = f"{ROOT}/manifests/b1_manifest.json"
    if os.path.exists(p):
        for e in json.load(open(p))["sequences"]:
            f = f"{ROOT}/data/processed/b1/{e['split']}/{e['seq_id']}.npz"
            if os.path.exists(f):
                out["b1"].append({"id": "b1:" + e["split"] + ":" + e["seq_id"], "split": e["split"], "label": e["label"],
                                  "name": f"{e['seq_id']} ({e['label']}, {len(e['excerpts'])} excerpts)"})
    p = f"{ROOT}/manifests/b2_manifest.json"
    if os.path.exists(p):
        for e in json.load(open(p))["concerts"]:
            f = f"{ROOT}/data/processed/b2/{re.sub(r'[^A-Za-z0-9]+', '_', e['concert'])}.npz"
            if os.path.exists(f):
                out["b2"].append({"id": "b2:" + e["concert"], "name": e["concert"], "split": "test"})
    p = f"{ROOT}/data/processed/splits.csv"
    if os.path.exists(p):
        import pandas as pd
        s = pd.read_csv(p)
        for r in s[(s.in_set) & (s.usable)].itertuples():
            if os.path.exists(f"{ROOT}/data/interim/pitch/{safe_name(r.rec_id)}.npz"):
                out["rec"].append({"id": "rec:" + r.rec_id, "split": r.split, "raga": r.raga, "name": f"{r.artist}, {r.concert.replace('_', ' ')}, {r.raga} ({r.split})"})
    return out


def load_item(key: str, upload: dict | None = None) -> dict:
    """Load cents/voiced and the reference for a source key (cached)."""
    if key in ITEMS:
        return ITEMS[key]
    kind, _, rest = key.partition(":")
    if kind == "b1":
        split, seq = rest.split(":", 1)
        z = np.load(f"{ROOT}/data/processed/b1/{split}/{seq}.npz")
        it = {"name": seq, "cents": z["cents"].astype(float), "voiced": z["voiced"], "ref_iv": z["intervals"], "ref_lab": [str(x) for x in z["labels"]],
              "note": f"B1 {split} splice (reference = true excerpt order)"}
    elif kind == "b2":
        z = np.load(f"{ROOT}/data/processed/b2/{re.sub(r'[^A-Za-z0-9]+', '_', rest)}.npz")
        it = {"name": rest, "cents": z["cents"].astype(float), "voiced": z["voiced"], "ref_iv": z["intervals"], "ref_lab": [str(x) for x in z["labels"]],
              "note": "B2 test concert (reference = item order; `other` = outside the 40 ragas or a ragamalika)"}
    elif kind == "rec":
        import pandas as pd
        s = pd.read_csv(f"{ROOT}/data/processed/splits.csv").set_index("rec_id")
        z = np.load(f"{ROOT}/data/interim/pitch/{safe_name(rest)}.npz")
        c = z["cents"].astype(float)
        it = {"name": rest, "cents": c, "voiced": z["voiced"], "ref_iv": np.array([[0.0, len(c) / 100.0]]), "ref_lab": [str(s.loc[rest, "raga"])],
              "note": "single-raga recording (reference = its one raga, so there are no true boundaries)"}
    elif kind == "upload" and upload is not None:
        it = parse_upload(upload)
    else:
        raise KeyError(key)
    ITEMS[key] = it
    return it


def parse_upload(u: dict) -> dict:
    """Two-column text (time_s, f_hz; 0 = unvoiced) + tonic in Hz, as in the CMD .pitch files."""
    tonic = float(u["tonic_hz"])
    if not (50 <= tonic <= 600):
        raise ValueError("tonic should be a frequency in Hz between 50 and 600")
    arr = np.loadtxt(io.StringIO(u["pitch_text"]), ndmin=2)
    if arr.shape[1] < 2 or arr.shape[0] < 3000:
        raise ValueError("expected a two-column pitch file (time in s, frequency in Hz) with at least 30 s of frames")
    cents, voiced = hz_to_cents(arr[:, 1], tonic)
    g = resample_to_grid(arr[:, 0], cents, voiced)
    return {"name": u.get("name") or "uploaded pitch file", "cents": g["cents"].astype(float), "voiced": g["voiced"], "ref_iv": None, "ref_lab": None,
            "note": "uploaded file (no reference, so no boundary scores)"}


def analyze(key: str, model: str, upload: dict | None = None) -> dict:
    with LOCK:
        it = load_item(key, upload)
        k = (key, model)
        if k not in ANALYSES:
            m = load_model(model)
            p = sequence_posteriors(m, "w1" if model.startswith("w1") else "w2", it["cents"], it["voiced"])
            c = np.where(it["voiced"], np.mod(it["cents"], 1200.0), np.nan)
            tb = max(5.0, round(len(c) / 100.0 / 600.0))  # at most ~600 time columns
            ti = (np.arange(len(c)) / 100.0 // tb).astype(int)
            ok = np.isfinite(c)
            pb = np.minimum((c[ok] / 50.0).astype(int), 23)
            nt = int(ti.max()) + 1
            heat = np.bincount(ti[ok] * 24 + pb, minlength=nt * 24).reshape(nt, 24)
            ANALYSES[k] = {**p, "ragas": m["ragas"], "L": float(m["L"]), "hop": float(m["hop"]), "heat": heat, "tb": tb}
        return ANALYSES[k]


def labels_names(an: dict, lab) -> np.ndarray:
    return np.array([an["ragas"][j] for j in lab])


def metrics(it: dict, iv: np.ndarray, names: np.ndarray) -> dict:
    out = {"n_pred": int(max(len(iv) - 1, 0))}
    if it["ref_iv"] is None:
        return out
    t5, t10 = boundary_counts(it["ref_iv"], iv, 5.0), boundary_counts(it["ref_iv"], iv, 10.0)
    nr = t5[2]
    out["n_ref"] = int(nr)
    fa = frame_accuracy(it["ref_iv"], it["ref_lab"], iv, names)
    out["frame_acc"] = fa[0] / fa[1] if fa[1] else None
    if nr:
        for tag, t in (("5", t5), ("10", t10)):
            p, r = (t[0] / t[1] if t[1] else 0.0), t[0] / t[2]
            out["P" + tag], out["R" + tag] = p, r
            out["F" + tag] = 2 * p * r / (p + r) if p + r else 0.0
    return out


def decode(key: str, model: str, selected: str, prm: dict) -> dict:
    an = analyze(key, model)
    it = ITEMS[key]
    hop = an["hop"]
    post, valid, cen, h = an["post"], an["valid"], an["centres"], an["hists"]
    fns = {"D0": lambda: decode_d0(post, valid, cen, hop),
           "D1": lambda: decode_d1(post, valid, cen, hop, int(prm["d1_width"])),
           "D2": lambda: decode_d2(h, post, valid, cen, hop, 3, float(prm["d2_thr"])),
           "D3": lambda: decode_d3(post, valid, cen, hop, float(prm["eps"]), float(prm["min_dur_s"]), None, float(prm["beta"])),
           "D3-online": lambda: decode_d3(post, valid, cen, hop, float(prm["eps"]), float(prm["min_dur_s"]), None, float(prm["beta"]), online=True)}
    res, seg = {}, None
    for d, fn in fns.items():
        if d == "D3-online" and selected != "D3-online":
            continue
        iv, lab = fn()
        names = labels_names(an, lab)
        res[d] = metrics(it, iv, names)
        if d == selected:
            segs = []
            for (a, b), j in zip(iv, lab):
                m = (cen >= a) & (cen < b) & valid
                segs.append({"start": float(a), "end": float(b), "raga": an["ragas"][j], "p": float(post[m, j].mean()) if m.any() else None})
            seg = segs
    return {"metrics": res, "segments": seg, "selected": selected}


def analysis_payload(key: str, model: str, upload: dict | None = None) -> dict:
    an = analyze(key, model, upload)
    it = ITEMS[key]
    post, valid = an["post"], an["valid"]
    mean = post[valid].mean(0) if valid.any() else post.mean(0)
    cols = list(np.argsort(-mean)[:8])
    for l in (it["ref_lab"] or []):
        if l in an["ragas"] and an["ragas"].index(l) not in cols:
            cols.append(an["ragas"].index(l))
    top = np.argsort(-post, axis=1)[:, :5]
    return {"key": key, "name": it["name"], "note": it["note"], "duration_s": len(it["cents"]) / 100.0, "hop": an["hop"], "L": an["L"],
            "centres": np.round(an["centres"], 2).tolist(), "valid": valid.astype(int).tolist(), "ragas": an["ragas"],
            "cols": [{"raga": an["ragas"][j], "p": np.round(post[:, j], 3).tolist()} for j in cols],
            "top_idx": top.tolist(), "top_p": np.round(np.take_along_axis(post, top, axis=1), 3).tolist(),
            "heat": {"tb": an["tb"], "bins": 24, "data": an["heat"].tolist()},
            "ref": None if it["ref_iv"] is None else {"intervals": np.round(it["ref_iv"], 2).tolist(), "labels": it["ref_lab"]}}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def json(self, obj, code: int = 200) -> None:
        self.send(code, json.dumps(obj).encode("utf8"))

    def do_GET(self) -> None:
        try:
            if self.path in ("/", "/index.html"):
                self.send(200, open(f"{ROOT}/app/index.html", "rb").read(), "text/html; charset=utf-8")
            elif self.path == "/api/config":
                self.json(config())
            elif self.path == "/api/sources":
                self.json(sources())
            else:
                self.json({"error": "not found"}, 404)
        except Exception as e:  # noqa: BLE001
            self.json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_POST(self) -> None:
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n > MAX_UPLOAD:
                return self.json({"error": "request too large (60 MB limit)"}, 413)
            body = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/analyze":
                key = body["source"]
                if key not in ITEMS and key.startswith("upload:"):
                    key = "upload:" + str(abs(hash(body["upload"]["pitch_text"][:2000] + str(body["upload"]["tonic_hz"]))))
                self.json(analysis_payload(key, body["model"], body.get("upload")))
            elif self.path == "/api/decode":
                self.json(decode(body["key"], body["model"], body["decoder"], body["params"]))
            else:
                self.json({"error": "not found"}, 404)
        except Exception as e:  # noqa: BLE001
            self.json({"error": f"{type(e).__name__}: {e}"}, 400)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true")
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://127.0.0.1:{a.port}"
    print(f"Raga tracking explorer on {url} (Ctrl+C to stop)", flush=True)
    if a.open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
