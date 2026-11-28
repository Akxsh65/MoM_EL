# E6: error analysis of the best configuration

Configuration: W2 + D3, L = 30 s (the frozen E2 setting), test split, saved predictions from `results/e2_predictions.json.gz`.
All numbers come from `python scripts/12_error_analysis.py` (commit in `results/e6_tables.md`); `T` = table in `results/e6_tables.md`,
`F` = figure in `results/figures/`. This is a post-hoc description of one frozen run. Nothing here was used to tune or choose anything.
B1 = 1,038 reference boundaries from 500 sequences (8 concerts); B2 = 53 boundaries from 8 concerts. Confidence intervals resample whole concerts.

## 1. Where boundaries are found or missed

1. **Most "errors" at +-5 s are near misses, not missing boundaries.** On B1, 47.7% of estimated boundaries are correct at +-5 s, a further 33.0% fall 5-10 s from a reference boundary, and only 19.3% are more than 10 s from any reference boundary (T: "B1 estimated boundaries by distance"). Reference recall rises from 0.428 at +-5 s to 0.724 at +-10 s (T: "Recall of reference boundaries"). The matched errors are spread over the whole +-10 s range with a median of +1.1 s, so there is no systematic delay on B1 (F: `e6_timing_error.png`; T: "Timing error"). B2 matches are few (36) and lean early (median -1.6 s); too few to interpret.
2. **Boundary recall falls as the two true neighbouring rāgas share more svaras.** B1 recall at +-5 s is 0.467 (<=3 shared), 0.464 (4), 0.412 (5), 0.376 (6) and 0.314 (>=7); at +-10 s it is 0.80 down to 0.59 (F: `e6_error_vs_shared.png`, left; T: "By shared svaras"). Easy-like pairs (<=4 shared) are found at 0.465 and hard-like pairs (>=6) at 0.364 (T: "Recall of reference boundaries"). The trend is monotone but gentle: neighbouring buckets' concert CIs overlap, and only the two ends (<=3 vs >=7) are separated, barely (0.378 vs 0.371 at the CI edges). This agrees with the small easy/hard gap in E2.
3. **False boundaries are not mainly flips between svara-similar rāgas.** The share of estimated boundaries that are correct barely depends on how many svaras the two *predicted* rāgas share (0.50, 0.50, 0.50, 0.42, 0.44 for <=3 ... >=7; overlapping CIs; F: `e6_error_vs_shared.png`, right; T: "By shared svaras").
4. **Rāga labels are often right even when the boundary is missed.** Thirty seconds into the new B1 segment the predicted label is correct for 0.766 of found boundaries and 0.680 of missed ones (T: "Label correctness 30 s after a found boundary"). The weakness is where D3 places and commits to a switch more than which rāga it names.

## 2. B2 (real concerts)

5. **Only a quarter of D3's boundaries on B2 are correct.** Of the estimated boundaries, 25.3% are correct at +-5 s, 11.1% are near misses, 32.3% fall inside an `other` item and 31.3% inside an in-set item (T: "B2 estimated boundaries by where they fall").
6. **D3 splits `other` blocks into several invented rāgas.** A closed-set classifier must name one of the 40 rāgas, and D3 then switches between them (F: `e6_worst_b2_1_Live_at_Vani_Mahal.png`, 660-2000 s; `e6_worst_b2_2_Cherthala_Ranganatha_Sharma_at_Arkay.png`, 3300-7400 s, an `other` block of about 4,100 s that D3 cuts into shanmukhapriya, varali, purvikalyani and others). **Caveat on the reference:** B2 merges neighbouring items that share a label, so an `other` block can contain several real items. 9 of the 18 `other` blocks hold two or more items, and 6 of the 32 false boundaries inside `other` lie within 5 s of a real item change that the reference does not score (T: "B2 `other` blocks"). So the B2 precision is somewhat understated, but the other 26 are not near any item change.
7. **By kind, raga-to-raga boundaries are found least often** (0.417 at +-5 s, n = 24, versus 0.50 raga-to-`other`, n = 18, and 0.545 `other`-to-raga, n = 11; T: "B2 reference boundaries by kind"). The groups are small and the differences are within noise.
8. **The worst concerts are those with long `other` blocks or an item where the reference rāga's posterior collapses.** Concerts ranked by boundary F@5 (T: "B2 concerts, worst first"): Live at Vani Mahal 0.207 (3 of 21 estimated boundaries correct), Cherthala Ranganatha Sharma 0.250, Sanjay at Sastri Hall 0.276, Kuldeep Pai 0.286, V Shankarnarayanan 0.300, Live at Kamarajar Hall 0.435, Srividya Janakiraman 0.545, Chaitra Sairam 0.600. The plan asks for the 10 worst sequences; B2 has only 8 concerts, so all 8 are plotted (F: `e6_worst_b2_<rank>_<concert>.png`). In Live at Vani Mahal the last shanmukhapriya item (about 5400-9300 s) keeps its reference label while the shanmukhapriya posterior drops to near zero for roughly its final 1,000 s, and D3 breaks that stretch into several segments (F: `e6_worst_b2_1_...`). I have not investigated why (possibly unmarked non-melodic material). The best concert, Chaitra Sairam, shows the contrast: the in-set items are cut correctly and the errors are in the `other` item and at the end (F: `e6_worst_b2_8_Chaitra_Sairam_at_Arkay.png`).

## 3. Rāga pairs

9. **Some pairs are almost never found at +-5 s.** Weakest pairs with n >= 6 (T: "Weakest rāga pairs", `results/e6_pairs.csv`): harikambhoji / mohanam 0.06 (n = 16), kamas / kapi 0.07 (15), sriranjani / todi 0.10 (21), madhyamavati / sriranjani 0.15 (20), behag / kedaragaula 0.18 (17), kambhoji / mohanam 0.25 (20). Mohanam's five svaras all occur in harikambhoji and kambhoji, which would make a pitch-class or time-delayed histogram hard to separate. This is a plausible explanation that I have not tested. Sriranjani / todi share only 4 svaras yet are also weak, so the shared-svara count alone does not explain pair difficulty.
10. **Do not generalise from the pair table.** Almost every pair comes from one concert (column `concerts`), so a pair's recall is confounded with that concert's artist, recording and the other pairs in it.

## Limitations of this analysis

- One configuration, one frozen run; shared-svara counts use `configs/scales.yaml`, which still has 11 entries flagged `verify` (sencurutti unresolved).
- B2 has 8 concerts and 53 boundaries, so every B2 statement above is descriptive.
- Pair-level numbers are small-sample and concert-confounded (item 10).
- The "invented rāgas in `other` blocks" finding depends on the closed-set design; an explicit `other` class was not part of the plan.
