# E6 auto-generated tables (numbers behind results/e6_errors.md)

Produced by: `python scripts/12_error_analysis.py` | commit 4335744+uncommitted | W2 + D3, L=30 (frozen E2 config), test split, saved predictions

## Recall of reference boundaries (B1 + B2)

| set | n | found@5 | found@10 |
|---|---|---|---|
| B1 all | 1038 | 0.428 | 0.724 |
| B1 raga-raga, shared<=4 (easy-like) | 553 | 0.465 | 0.776 |
| B1 raga-raga, shared>=6 (hard-like) | 269 | 0.364 | 0.636 |
| B2 all | 53 | 0.472 | 0.679 |

## By shared svaras (concert-cluster bootstrap 95% CI)

| metric | shared svaras | n | rate | lo | hi |
|---|---|---|---|---|---|
| B1 reference boundaries found, within +-5 s | <=3 | 105 | 0.467 | 0.378 | 0.556 |
| B1 reference boundaries found, within +-5 s | 4 | 448 | 0.464 | 0.429 | 0.516 |
| B1 reference boundaries found, within +-5 s | 5 | 216 | 0.412 | 0.340 | 0.490 |
| B1 reference boundaries found, within +-5 s | 6 | 218 | 0.376 | 0.298 | 0.449 |
| B1 reference boundaries found, within +-5 s | >=7 | 51 | 0.314 | 0.220 | 0.371 |
| B1 reference boundaries found, within +-10 s | <=3 | 105 | 0.800 | 0.654 | 0.906 |
| B1 reference boundaries found, within +-10 s | 4 | 448 | 0.770 | 0.663 | 0.872 |
| B1 reference boundaries found, within +-10 s | 5 | 216 | 0.699 | 0.579 | 0.813 |
| B1 reference boundaries found, within +-10 s | 6 | 218 | 0.647 | 0.556 | 0.753 |
| B1 reference boundaries found, within +-10 s | >=7 | 51 | 0.588 | 0.417 | 0.787 |
| B1 estimated boundaries that are correct (+-5 s) | <=3 | 130 | 0.500 | 0.443 | 0.566 |
| B1 estimated boundaries that are correct (+-5 s) | 4 | 315 | 0.495 | 0.450 | 0.538 |
| B1 estimated boundaries that are correct (+-5 s) | 5 | 217 | 0.498 | 0.412 | 0.611 |
| B1 estimated boundaries that are correct (+-5 s) | 6 | 206 | 0.422 | 0.360 | 0.473 |
| B1 estimated boundaries that are correct (+-5 s) | >=7 | 63 | 0.444 | 0.189 | 0.655 |

## B2 reference boundaries by kind

| kind | n | recall5 | recall10 |
|---|---|---|---|
| other-to-raga | 11 | 0.545 | 0.727 |
| raga-raga | 24 | 0.417 | 0.708 |
| raga-to-other | 18 | 0.500 | 0.611 |

## B2 estimated boundaries by where they fall

| where | n | share |
|---|---|---|
| correct (+-5 s) | 25 | 0.253 |
| false, inside an `other` item | 32 | 0.323 |
| false, inside an in-set item | 31 | 0.313 |
| near miss (5-10 s) | 11 | 0.111 |

## B2 `other` blocks (neighbouring same-label items are merged in the reference)

| other blocks | blocks with >=2 items | false boundaries inside other | ...of which within 5 s of a real item change |
|---|---|---|---|
| 18 | 9 | 32 | 6 |

## B1 estimated boundaries by distance to the nearest reference boundary

| where | n | share |
|---|---|---|
| correct (+-5 s) | 444 | 0.477 |
| near miss (5-10 s) | 307 | 0.330 |
| false (>10 s from any reference boundary) | 180 | 0.193 |

## Timing error of matched boundaries (est - ref, s; matched within +-10 s)

| set | n | median | mean | share_within_5s |
|---|---|---|---|---|
| B1 | 751 | 1.120 | 0.853 | 0.591 |
| B2 | 36 | -1.620 | -1.210 | 0.694 |

## Label correctness 30 s after a found boundary (B1, B2 in-set targets)

| set | boundary found@5 | boundary missed@5 |
|---|---|---|
| B1 | 0.766 | 0.680 |
| B2 | 0.875 | 0.895 |

## Weakest rāga pairs (B1+B2 raga-raga boundaries, n>=6, lowest recall@5 first)

| pair | n | shared | recall5 | recall10 | label_ok | concerts |
|---|---|---|---|---|---|---|
| harikambhoji / mohanam | 16 | 5.000 | 0.062 | 0.188 | 0.562 | 1 |
| kamas / kapi | 15 | 6.000 | 0.067 | 0.333 | 0.733 | 2 |
| sriranjani / todi | 21 | 4.000 | 0.095 | 0.143 | 0.524 | 1 |
| bhairavi / kalyani | 7 | 4.000 | 0.143 | 0.714 | 0.714 | 1 |
| madhyamavati / sriranjani | 20 | 4.000 | 0.150 | 0.150 | 0.000 | 1 |
| harikambhoji / sankarabharanam | 6 | 6.000 | 0.167 | 0.167 | 0.833 | 1 |
| behag / kedaragaula | 17 | 6.000 | 0.176 | 0.353 | 0.353 | 1 |
| purvikalyani / sindhubhairavi | 9 | 3.000 | 0.222 | 0.778 | 1.000 | 1 |
| kambhoji / mohanam | 20 | 5.000 | 0.250 | 0.300 | 0.700 | 1 |
| bhairavi / nata | 8 | 5.000 | 0.250 | 0.625 | 1.000 | 1 |
| kambhoji / sankarabharanam | 8 | 6.000 | 0.250 | 0.250 | 0.625 | 1 |
| kamas / sankarabharanam | 11 | 6.000 | 0.273 | 0.273 | 0.636 | 2 |

## Strongest rāga pairs (n>=6)

| pair | n | shared | recall5 | recall10 | label_ok | concerts |
|---|---|---|---|---|---|---|
| kamas / sindhubhairavi | 9 | 4.000 | 0.778 | 0.889 | 0.444 | 1 |
| kamas / shanmukhapriya | 7 | 4.000 | 0.714 | 1.000 | 0.857 | 1 |
| purvikalyani / sankarabharanam | 7 | 5.000 | 0.714 | 0.857 | 1.000 | 1 |
| bhairavi / sama | 10 | 5.000 | 0.700 | 0.900 | 0.400 | 1 |
| kalyani / ritigaula | 9 | 4.000 | 0.667 | 0.889 | 0.444 | 1 |
| sankarabharanam / todi | 9 | 3.000 | 0.667 | 0.889 | 1.000 | 1 |
| bhairavi / purvikalyani | 6 | 3.000 | 0.667 | 0.833 | 1.000 | 1 |
| kanada / ritigaula | 6 | 7.000 | 0.667 | 0.667 | 0.833 | 1 |

## B2 concerts, worst first (boundary F@5)

| rank | concert | F5 | tp | n_est | n_ref |
|---|---|---|---|---|---|
| 1 | Live at Vani Mahal | 0.207 | 3 | 21 | 8 |
| 2 | Cherthala Ranganatha Sharma at Arkay | 0.250 | 2 | 10 | 6 |
| 3 | Sanjay at Sastri Hall | 0.276 | 4 | 18 | 11 |
| 4 | Kuldeep Pai at Arkay | 0.286 | 2 | 9 | 5 |
| 5 | V Shankarnarayanan at Arkay | 0.300 | 3 | 14 | 6 |
| 6 | Live at Kamarajar Hall | 0.435 | 5 | 15 | 8 |
| 7 | Srividya Janakiraman at Arkay | 0.545 | 3 | 6 | 5 |
| 8 | Chaitra Sairam at Arkay | 0.600 | 3 | 6 | 4 |
