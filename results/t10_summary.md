# T10: decoder tuning on B1-val

Produced by: `python scripts/10_tune_decoders.py` | commit 073467e+uncommitted | B1-val only (200 sequences)

Pooled boundary P/R/F at +-5 s (F10 at +-10 s), frame accuracy on a 1 s grid. D2 and D3 rows are the best of their grids by F5 (selected on the data they are scored on, so optimistic; D0 and D1 have no tuned parameters).

| model | decoder | P@5 | R@5 | F@5 | F@10 | frame acc | est. boundaries | ref. boundaries |
|---|---|---|---|---|---|---|---|---|
| W1 | D0 | 0.063 | 0.684 | 0.115 | 0.155 | 0.342 | 4693 | 431 |
| W1 | D1 | 0.089 | 0.510 | 0.152 | 0.257 | 0.352 | 2473 | 431 |
| W1 | D2 | 0.153 | 0.309 | 0.204 | 0.376 | 0.375 | 871 | 431 |
| W1 | D3 | 0.415 | 0.401 | 0.408 | 0.696 | 0.436 | 417 | 431 |
| W2 | D0 | 0.077 | 0.717 | 0.138 | 0.176 | 0.491 | 4036 | 431 |
| W2 | D1 | 0.117 | 0.587 | 0.195 | 0.281 | 0.511 | 2170 | 431 |
| W2 | D2 | 0.153 | 0.309 | 0.204 | 0.376 | 0.536 | 871 | 431 |
| W2 | D3 | 0.486 | 0.452 | 0.469 | 0.712 | 0.584 | 401 | 431 |

## Checks

- W1: D3 F@5 0.408 vs D0 0.115: **PASS**; vs D1 0.152: D3 higher
- W2: D3 F@5 0.469 vs D0 0.138: **PASS**; vs D1 0.195: D3 higher
