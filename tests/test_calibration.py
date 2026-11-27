import numpy as np

from raga_tracking.models.calibration import apply_temperature, ece, fit_temperature


def _synthetic(scale, n=4000, K=5, seed=0):
    rng = np.random.default_rng(seed)
    true = rng.normal(size=(n, K))
    p = np.exp(true) / np.exp(true).sum(1, keepdims=True)
    y = np.array([rng.choice(K, p=pi) for pi in p])
    return true * scale, y  # logits scaled by `scale`: scale=1 is calibrated


def test_overconfident_model_gets_T_above_one_and_lower_ece():
    logits, y = _synthetic(3.0)
    T = fit_temperature(logits, y)
    assert 2.0 < T < 4.5
    assert ece(apply_temperature(logits, T), y) < ece(apply_temperature(logits, 1.0), y)


def test_calibrated_model_keeps_T_near_one():
    logits, y = _synthetic(1.0)
    assert 0.8 < fit_temperature(logits, y) < 1.25


def test_ece_bounds():
    y = np.array([0, 1, 2, 0])
    perfect = np.eye(3)[y]
    assert ece(perfect, y) == 0.0
    wrong = np.eye(3)[(y + 1) % 3]
    assert abs(ece(wrong, y) - 1.0) < 1e-9
