import numpy as np

from raga_tracking.eval.bootstrap import COLS, ci_table, draws, metrics_from_sums, paired_diff
from raga_tracking.eval.metrics import segment_accuracy_counts


def _counts(n, rng, tp_rate=0.5):
    c = np.zeros((n, len(COLS)))
    c[:, 0] = rng.binomial(4, tp_rate, n); c[:, 1] = 4; c[:, 2] = 4
    c[:, 3], c[:, 4], c[:, 5] = c[:, 0], 4, 4
    c[:, 6], c[:, 7], c[:, 8], c[:, 9], c[:, 10] = 30, 60, 2, 3, 600
    return c


def test_point_estimate_matches_direct_sum():
    c = _counts(50, np.random.default_rng(0))
    m = metrics_from_sums(c.sum(0))
    assert np.isclose(m["P5"], c[:, 0].sum() / 200) and np.isclose(m["frame_acc"], 0.5) and np.isclose(m["false_sw_10min"], (200 - c[:, 0].sum()) / 50)


def test_ci_contains_point_and_draw_shapes():
    rng = np.random.default_rng(1)
    c = _counts(80, rng)
    mult = draws(80, 500, rng)
    assert mult.shape == (500, 80) and (mult.sum(1) == 80).all()
    t = ci_table(c, mult)
    for m, (p, lo, hi) in t.items():
        assert lo <= p <= hi or np.isnan(p)


def test_cluster_draws_share_multiplicity():
    rng = np.random.default_rng(2)
    cl = np.array(["a"] * 3 + ["b"] * 2 + ["c"] * 4)
    mult = draws(9, 50, rng, cl)
    assert (mult[:, 0] == mult[:, 2]).all() and (mult[:, 3] == mult[:, 4]).all()


def test_paired_diff_detects_better_system():
    rng = np.random.default_rng(3)
    a, b = _counts(100, rng, 0.9), _counts(100, rng, 0.3)
    r = paired_diff(a, b, draws(100, 500, rng))
    assert r["diff"] > 0.3 and r["lo"] > 0 and r["p_a_not_better"] == 0.0


def test_segment_accuracy_skips_other_and_uses_overlap():
    ref = np.array([[0, 100], [100, 200], [200, 300]])
    lab = ["a", "other", "b"]
    est = np.array([[0, 90], [90, 300]])
    assert segment_accuracy_counts(ref, lab, est, ["a", "b"]) == (2, 2)
    assert segment_accuracy_counts(ref, lab, est, ["b", "a"]) == (0, 2)
