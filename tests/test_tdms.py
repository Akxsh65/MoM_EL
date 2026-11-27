import numpy as np

from raga_tracking.features.tdms import tdms_windows

KW = dict(tau_s=0.3, n_bins=24, alpha=1.0, sigma=0.0)


def test_constant_pitch_single_diagonal_cell():
    c = np.full((1, 1000), 702.0)
    S = tdms_windows(c, **KW).reshape(24, 24)
    i = np.argwhere(S > 0)
    assert len(i) == 1 and i[0][0] == i[0][1] == 14  # Pa -> bin 14, on the diagonal
    assert np.isclose(S.sum(), 1)


def test_shift_by_one_bin_rolls_both_axes():
    rng = np.random.default_rng(0)
    c = np.cumsum(rng.normal(0, 40, (1, 3000)), axis=1) + 300
    S0 = tdms_windows(c, **KW).reshape(24, 24)
    S1 = tdms_windows(c + 50.0, **KW).reshape(24, 24)  # exactly one 50-cent bin
    assert np.allclose(np.roll(S0, (1, 1), axis=(0, 1)), S1, atol=1e-6)
    # also with smoothing + compression (circular smoothing commutes with rolling)
    kw = dict(tau_s=0.3, n_bins=24, alpha=0.75, sigma=1.0)
    A = tdms_windows(c, **kw).reshape(24, 24)
    B = tdms_windows(c + 50.0, **kw).reshape(24, 24)
    assert np.allclose(np.roll(A, (1, 1), axis=(0, 1)), B, atol=1e-6)


def test_masked_pairs_excluded():
    c = np.full((1, 1000), 702.0)
    c[0, 500:] = np.nan  # second half unvoiced
    c[0, 100] = 0.0       # one voiced frame at Sa: its pairs must appear
    S = tdms_windows(c, **KW).reshape(24, 24)
    assert S[14, 14] > 0.9
    assert S[0, :].sum() + S[:, 0].sum() > 0  # pairs involving the Sa frame at t=100 (and t-30)
    # NaN frames never land in bin 0 through the np.where(...,0.0) fill
    d = np.full((1, 1000), np.nan)
    d[0, ::2] = 702.0
    T = tdms_windows(d, tau_s=0.01, n_bins=24, alpha=1.0, sigma=0.0)  # tau = 1 frame: every pair has a NaN
    assert np.allclose(T, 1 / 576)  # no voiced pairs -> uniform fallback, not mass at (0, 0)


def test_tau_is_applied_in_frames():
    c = np.zeros((1, 200))
    c[0, :100] = 0.0
    c[0, 100:] = 600.0
    S = tdms_windows(c, tau_s=0.3, n_bins=24, alpha=1.0, sigma=0.0).reshape(24, 24)
    assert S[0, 12] > 0 and S[12, 0] == 0  # (pc(t), pc(t+tau)): from Sa to the tritone, never the reverse
