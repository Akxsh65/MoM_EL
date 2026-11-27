import numpy as np
from raga_tracking.pitch import hz_to_cents, pitch_class, resample_to_grid
from raga_tracking.windows import make_windows, n_windows


def test_cents_values():
    c, v = hz_to_cents(np.array([220.0, 440.0]), 220.0)
    assert np.allclose(c, [0, 1200]) and v.all()
    assert pitch_class(np.array([1200.0]))[0] == 0
    assert pitch_class(np.array([-100.0]))[0] == 1100


def test_unvoiced_masked_not_zero():
    c, v = hz_to_cents(np.array([0.0, -1.0, np.nan, 220.0]), 220.0)
    assert list(v) == [False, False, False, True]
    assert np.isnan(c[:3]).all()


def test_window_count():
    for dur, L, hop in [(100, 30, 5), (30, 30, 5), (29, 30, 5), (97.5, 30, 5)]:
        n = int(dur * 100)
        assert n_windows(n, 100, L, hop) == (0 if dur < L else int((dur - L) // hop) + 1)
        w = make_windows(np.zeros(n), np.ones(n, bool), 100, L, hop)
        assert len(w["centres"]) == n_windows(n, 100, L, hop)


def test_masked_frames_never_in_features():
    n = 6000
    v = np.ones(n, bool); v[1000:2000] = False
    w = make_windows(np.zeros(n), v, 100, 30, 5)
    assert np.isnan(w["cents"][~w["voiced"]]).all()
    assert not np.isnan(w["cents"][w["voiced"]]).any()


def test_centre_and_validity():
    n = 6000
    v = np.zeros(n, bool); v[:2000] = True
    w = make_windows(np.zeros(n), v, 100, 30, 5)
    assert w["centres"][0] == 15.0
    assert w["valid"][0] and not w["valid"][-1]


def test_resample_keeps_gaps_unvoiced():
    t = np.concatenate([np.arange(0, 1, 0.01), np.arange(2, 3, 0.01)])
    c = np.zeros(len(t)); v = np.ones(len(t), bool)
    r = resample_to_grid(t, c, v, fs=100)
    assert not r["voiced"][150] and r["voiced"][50]
