# Splits summary

Produced by: `python scripts/02_splits.py`

K = 40 (strict thresholds). Artist groups: 74.

| split | artists | in-set recs | frac | ragas covered | saraga recs | test-style concerts (>=3 in-set) |
|---|---|---|---|---|---|---|
| train | 53 | 336 | 62.9% | 40/40 | 21 | 4 |
| val | 7 | 87 | 16.3% | 40/40 | 19 | 4 |
| test | 7 | 111 | 20.8% | 40/40 | 51 | 8 |

Ragas missing from a split (cannot be covered given artist counts):
- val: []
- test: []
- train: []
