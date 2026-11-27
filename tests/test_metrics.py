import mir_eval
import numpy as np

from raga_tracking.eval.metrics import boundary_counts, frame_accuracy, pooled_prf


def test_matches_mir_eval_detection():
    rng = np.random.default_rng(1)

    def mk(n):
        b = np.sort(rng.uniform(10, 500, n))
        e = np.concatenate([[0], b, [520]])
        return np.stack([e[:-1], e[1:]], 1)

    for _ in range(20):
        ref, est = mk(rng.integers(1, 5)), mk(rng.integers(1, 6))
        tp, ne, nr = boundary_counts(ref, est, 5.0)
        p, r, f = mir_eval.segment.detection(ref, est, window=5.0, trim=True)
        pp, rr, ff = pooled_prf([(tp, ne, nr)])
        assert np.allclose([p, r, f], [pp, rr, ff])


def test_frame_accuracy_skips_other():
    ref = np.array([[0, 10], [10, 20]])
    lab = np.array(["a", "other"])
    est = np.array([[0, 5], [5, 20]])
    el = np.array(["a", "b"])
    assert frame_accuracy(ref, lab, est, el) == (5, 10)
