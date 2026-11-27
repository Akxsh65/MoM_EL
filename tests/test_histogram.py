import numpy as np

from raga_tracking.features.histogram import window_histograms


def test_sums_to_one_and_smoothed():
    c = np.random.default_rng(0).uniform(-3000, 3000, (5, 500))
    h = window_histograms(c, 24)
    assert np.allclose(h.sum(1), 1) and (h > 0).all()


def test_sa_in_bin_zero_and_octave_folding():
    for cents in (0.0, 1200.0, -1200.0, 10.0):
        h = window_histograms(np.full((1, 100), cents), 24)
        assert h.argmax() == 0
    assert window_histograms(np.full((1, 100), 702.0), 24).argmax() == 14  # Pa ~ 702/50 = 14.04
    assert window_histograms(np.full((1, 100), -100.0), 24).argmax() == 22  # pc 1100


def test_masked_frames_not_counted():
    c = np.full((1, 100), 0.0)
    c[0, :50] = np.nan
    c2 = np.full((1, 100), 0.0)
    c2[0, :50] = np.nan
    c2[0, 50:] = 0.0
    assert np.allclose(window_histograms(c), window_histograms(c2))
    # masking does not shift mass to bin 0: a window of NaN + Pa is pure Pa
    d = np.full((1, 100), np.nan)
    d[0, 50:] = 702.0
    assert window_histograms(d).argmax() == 14


def test_shift_by_one_bin_rolls():
    rng = np.random.default_rng(1)
    c = rng.uniform(0, 1200, (1, 4000))
    h, h2 = window_histograms(c, 24), window_histograms(c + 50.0, 24)
    assert np.allclose(np.roll(h, 1), h2, atol=2e-3)
