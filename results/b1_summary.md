# B1 summary (T8)

Produced by: `python scripts/08_build_b1.py --hard_min 6` | commit 3953350+uncommitted

| split | label | sequences | mean excerpts | mean duration (min) | concerts |
|---|---|---|---|---|---|
| val | easy | 80 | 2.56 | 6.2 | 3 |
| val | mid | 40 | 2.83 | 6.0 | 3 |
| val | hard | 80 | 3.91 | 8.8 | 4 |
| test | easy | 200 | 2.54 | 5.7 | 8 |
| test | mid | 100 | 3.10 | 6.9 | 8 |
| test | hard | 200 | 3.62 | 7.7 | 6 |

easy = all neighbouring pairs share <= 4 svaras; hard = at least one pair shares >= 6; mid = neither.
Labels rest on configs/scales.yaml, which still has unverified entries; they are re-derivable from the manifest's raga list.
