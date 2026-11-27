# Calibration (T7)

Produced by: `python scripts/07_calibrate.py` | commit e1547b5+uncommitted | val split only

Temperature scaling, T fit on val windows (NLL). ECE with 15 equal-width bins on the max-probability prediction.

| model | T | ECE before | ECE after | NLL before | NLL after | median max-p before | median max-p after | mean conf. before | val acc |
|---|---|---|---|---|---|---|---|---|---|
| W1 | 1.200 | 0.0421 | 0.0951 | 1.942 | 1.919 | 0.407 | 0.340 | 0.469 | 0.504 |
| W2 | 1.077 | 0.0472 | 0.0744 | 1.370 | 1.366 | 0.607 | 0.569 | 0.618 | 0.665 |

Cross-fitted check (T fit on one half of the val recordings, ECE scored on the other half, averaged over both halves):

| model | T (fold 0, fold 1) | ECE before | ECE after |
|---|---|---|---|
| W1 | 1.199, 1.201 | 0.0426 | 0.0949 |
| W2 | 1.014, 1.139 | 0.0479 | 0.0743 |

## Check results (computed)

- W1: ECE decreases in-sample: **FAIL**; cross-fitted: **FAIL**; median max-prob < 0.99: **PASS**
- W2: ECE decreases in-sample: **FAIL**; cross-fitted: **FAIL**; median max-prob < 0.99: **PASS**

Reading: mean confidence below accuracy means the model is under-confident, so a temperature T > 1 (which flattens)
cannot reduce ECE. If a check fails, the T7 backup applies: no temperature, tune a posterior exponent beta in D3 on B1-val (T10).
