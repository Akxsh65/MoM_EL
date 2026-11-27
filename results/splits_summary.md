# Splits summary

Produced by: `python scripts/02_splits.py`

K = 40 (strict thresholds). Artist groups: 74.

| split | artists | in-set recs | frac | ragas covered | saraga recs | test-style concerts (>=3 in-set) |
|---|---|---|---|---|---|---|
| train | 53 | 328 | 62.0% | 40/40 | 22 | 4 |
| val | 7 | 94 | 17.8% | 40/40 | 20 | 4 |
| test | 7 | 107 | 20.2% | 40/40 | 47 | 8 |

Ragas missing from a split (cannot be covered given artist counts):
- val: []
- test: []
- train: []
