# T10: decoder tuning on B1-val

Produced by: `python scripts/10_tune_decoders.py` | commit e882b28+uncommitted | B1-val only (200 sequences)

Pooled boundary P/R/F at +-5 s (F10 at +-10 s), frame accuracy on a 1 s grid. D2 and D3 rows are the best of their grids by F5 (selected on the data they are scored on, so optimistic; D0 and D1 have no tuned parameters).

| model | decoder | P@5 | R@5 | F@5 | F@10 | frame acc | est. boundaries | ref. boundaries |
|---|---|---|---|---|---|---|---|---|
| W1 | D0 | 0.064 | 0.675 | 0.117 | 0.160 | 0.362 | 4461 | 424 |
| W1 | D1 | 0.090 | 0.505 | 0.153 | 0.261 | 0.373 | 2381 | 424 |
| W1 | D2 | 0.143 | 0.297 | 0.193 | 0.363 | 0.395 | 882 | 424 |
| W1 | D3 | 0.400 | 0.403 | 0.402 | 0.696 | 0.460 | 427 | 424 |
| W2 | D0 | 0.079 | 0.715 | 0.143 | 0.182 | 0.513 | 3823 | 424 |
| W2 | D1 | 0.121 | 0.587 | 0.201 | 0.293 | 0.535 | 2056 | 424 |
| W2 | D2 | 0.143 | 0.297 | 0.193 | 0.363 | 0.561 | 882 | 424 |
| W2 | D3 | 0.491 | 0.455 | 0.472 | 0.720 | 0.616 | 393 | 424 |

## Checks

- W1: D3 F@5 0.402 vs D0 0.117: **PASS**; vs D1 0.153: D3 higher
- W2: D3 F@5 0.472 vs D0 0.143: **PASS**; vs D1 0.201: D3 higher
