import numpy as np

from raga_tracking.eval.latency import detection_latencies, summarize

REF = np.array([[0, 100], [100, 200], [200, 300]], float)
LAB = ["a", "b", "c"]


def _avail():
    return np.arange(5.0, 300.0, 5.0)  # window-end times, hop 5


def test_latency_is_first_of_two_consecutive_hops_minus_boundary():
    av = _avail()
    pred = np.array(["a"] * len(av), dtype=object)
    pred[(av >= 120) & (av < 300)] = "b"      # switches to b at 120 (20 s late), then stays b
    pred[av >= 230] = "c"                     # c from 230 (30 s late)
    lat = detection_latencies(REF, LAB, pred, av)
    assert np.allclose(lat, [20.0, 30.0])


def test_single_hop_is_not_enough_and_miss_is_nan_not_dropped():
    av = _avail()
    pred = np.array(["a"] * len(av), dtype=object)
    pred[av == 110] = "b"                     # isolated hop: not 2 consecutive
    lat = detection_latencies(REF, LAB, pred, av)
    assert len(lat) == 2 and np.isnan(lat[0]) and np.isnan(lat[1])
    assert summarize(lat)["miss_rate"] == 1.0


def test_no_match_within_60s_is_a_miss():
    av = _avail()
    pred = np.array(["a"] * len(av), dtype=object)
    pred[av >= 170] = "b"                     # 70 s after the boundary at 100
    assert np.isnan(detection_latencies(REF, LAB, pred, av)[0])


def test_other_boundaries_skipped_and_causal_in_time():
    av = _avail()
    pred = np.array(["a"] * len(av), dtype=object)
    lat = detection_latencies(REF, ["a", "other", "c"], pred, av)
    assert len(lat) == 1                      # boundary into `other` is not scored
    pred[av >= 205] = "c"
    pred_early = pred.copy(); pred_early[av < 90] = "c"   # a hit before the boundary must not count
    assert np.allclose(detection_latencies(REF, ["a", "other", "c"], pred_early, av), [5.0])
