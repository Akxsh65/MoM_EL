"""Independent recomputation of boundary F@5 for 5 test sequences (no raga_tracking.eval import). Usage: python scripts/check_e2_independent.py"""
import gzip, json, os
import numpy as np, pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
per = pd.read_csv(f"{R}/results/e2_per_sequence.csv")
pred = json.load(gzip.open(f"{R}/results/e2_predictions.json.gz", "rt"))
man = {e["seq_id"]: e for e in json.load(open(f"{R}/manifests/b1_manifest.json"))["sequences"]}
rows = per[(per.model == "W2") | (per.model == "w2")]
rows = rows[rows.bench.str.startswith("B1") & (rows.decoder == "D3")].head(5)
for r in rows.itertuples():
    seg = [0.0]; ref_b = []                       # reference boundaries straight from the manifest excerpt lengths
    for e in man[r.id]["excerpts"]:
        seg.append(seg[-1] + e["end_s"] - e["start_s"])
    ref_b = np.array(seg[1:-1])
    est_b = np.array([iv[1] for iv in pred[f"w2|D3|{r.id}"]["intervals"]][:-1])
    A = csr_matrix((np.abs(ref_b[:, None] - est_b[None, :]) <= 5.0).astype(int))
    tp = int((maximum_bipartite_matching(A, perm_type="column") >= 0).sum()) if len(ref_b) and len(est_b) else 0
    p, rc = tp / max(len(est_b), 1), tp / len(ref_b)
    f = 2 * p * rc / (p + rc) if p + rc else 0.0
    f_run = 2 * (r.tp5 / max(r.ne5, 1)) * (r.tp5 / r.nr5) / ((r.tp5 / max(r.ne5, 1)) + (r.tp5 / r.nr5)) if r.tp5 else 0.0
    print(r.id, "independent F5 %.4f  run_final F5 %.4f" % (f, f_run), "OK" if abs(f - f_run) < 1e-9 and tp == r.tp5 else "MISMATCH")
