import numpy as np

from raga_tracking.decode import decode_d0, decode_d1, decode_d2, decode_d3
from raga_tracking.decode.hmm import emissions, forward_filter

HOP = 5.0


def _post(labels, K=4, conf=0.9):
    p = np.full((len(labels), K), (1 - conf) / (K - 1))
    p[np.arange(len(labels)), labels] = conf
    return p


def _centres(T):
    return 15.0 + HOP * np.arange(T)


def test_one_switch_one_boundary_at_right_window():
    labels = np.array([0] * 20 + [2] * 20)
    p, c = _post(labels), _centres(40)
    for dec in (decode_d0, decode_d1, lambda *a: decode_d3(*a, eps=1e-2, min_dur=30)):
        iv, lab = dec(p, np.ones(40, bool), c, HOP)
        assert list(lab) == [0, 2] and len(iv) == 2
        assert np.isclose(iv[0, 1], c[20] - HOP / 2)  # boundary between window 19 and 20


def test_d3_removes_isolated_flips_and_d0_does_not():
    labels = np.array([0] * 30)
    labels[[7, 15, 22]] = 3
    p, c, v = _post(labels, conf=0.7), _centres(30), np.ones(30, bool)
    assert len(decode_d0(p, v, c, HOP)[1]) > 1
    iv, lab = decode_d3(p, v, c, HOP, eps=1e-3, min_dur=30)
    assert list(lab) == [0]


def test_online_filter_is_causal():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(4), size=60)
    v = np.ones(60, bool)
    em1 = emissions(p, v)
    p2 = p.copy()
    p2[40:] = rng.dirichlet(np.ones(4), size=20)
    em2 = emissions(p2, v)
    f1, f2 = forward_filter(em1, 6, 1e-2), forward_filter(em2, 6, 1e-2)
    assert np.allclose(f1[:40], f2[:40])
    assert not np.allclose(f1[40:], f2[40:])


def test_d3_min_duration_respected_except_edges():
    rng = np.random.default_rng(1)
    for _ in range(5):
        p = rng.dirichlet(np.ones(5) * 0.3, size=120)
        iv, lab = decode_d3(p, np.ones(120, bool), _centres(120), HOP, eps=0.05, min_dur=45)
        durs = iv[:, 1] - iv[:, 0]
        assert (durs[:-1] >= 45 - 1e-9).all()  # all but the last (edge) segment
        assert durs[0] >= 45 - 1e-9


def test_silent_windows_carry_state():
    labels = np.array([1] * 20)
    p, c = _post(labels), _centres(20)
    v = np.ones(20, bool)
    v[8:12] = False
    p[8:12] = 0.25
    for dec in (decode_d0, decode_d1, lambda *a: decode_d3(*a, min_dur=30)):
        assert list(dec(p, v, c, HOP)[1]) == [1]


def test_d2_finds_change_in_histograms():
    h = np.ones((30, 24)); h[:15, :6] = 20; h[15:, 12:18] = 20
    h = h / h.sum(1, keepdims=True)
    labels = np.array([0] * 15 + [1] * 15)
    iv, lab = decode_d2(h, _post(labels, K=3), np.ones(30, bool), _centres(30), HOP, k=3, thr=0.5)
    assert list(lab) == [0, 1]
