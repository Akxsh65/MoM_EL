# Tracking rāga over time in Carnatic recordings: window classifiers, temporal decoders and benchmarks

*Draft for the team. Authors, affiliations and venue: to be confirmed. Every number below comes from a file in `results/` (named in brackets);
commands and commit hashes are in those files' headers. The related-work claims rest on abstracts and metadata only (see Section 2).*

## Abstract

Rāga recognition is usually evaluated on whole recordings, but a performance can change rāga within a recording (a rāgamālika, or successive items of a concert).
We study *time-resolved* recognition: which of 40 Carnatic rāgas is sung at each moment, and where the rāga changes. From open pitch tracks and tonics only, we classify
sliding windows (a pitch-class histogram, W1, or a time-delayed melody surface, W2) and decode the window posteriors with four temporal decoders (argmax, majority filter,
novelty-based change points, and an HMM with a minimum segment duration). We introduce two benchmarks: B1, 500 synthetic splices of same-concert excerpts, labelled easy or hard by the
number of svaras neighbouring rāgas share, and B2, eight real concerts in performance order. On a frozen test run, the HMM decoder (D3) improves boundary F-measure at ±5 s over argmax
and majority filtering by +0.21 to +0.34 on every benchmark (all paired bootstrap differences are positive in all 1,000 resamples). Absolute accuracy is modest: with 30 s windows, F@5 is 0.43 to 0.48 on B1 and 0.33 on B2.
Shorter windows localise changes better (B1 F@5 up to 0.77 with 10 s windows) at lower window accuracy, and online detection of a change takes a median of 26 to 40 s with 10 to 30 s windows, with 27 to 41% of changes not detected within 60 s.
Errors grow with the number of shared svaras but only gently, and on real concerts the closed rāga set makes the decoder invent rāga changes inside items outside the set.
A planned rāgamālika benchmark with human annotators (B3) was not done.

## 1. Introduction

A Carnatic rāga is a melodic framework, and a single recording often contains more than one: concerts are sequences of items in different rāgas, and a rāgamālika
moves through several rāgas within one composition. Standard rāga recognition assigns one label per recording (or per clip), so it cannot say *when* the rāga changes. Koduri et al. (2012)
name rāgamālika as an open problem for pitch-distribution methods. We ask two questions: (i) can a window classifier plus a temporal decoder recover the rāga at every moment,
and (ii) how precisely and how quickly can it locate a change? We use only pitch tracks and tonics from open datasets, and evaluate boundaries explicitly.

Contributions: (1) a pipeline with four decoders compared under one protocol; (2) two benchmarks, B1 (synthetic same-concert splices, easy/hard) and B2 (real concerts), released as manifests (IDs, cut times, seeds);
(3) results with confidence intervals, a window-length and online-latency study, and an error analysis; (4) a record of every deviation and negative result (`docs/decisions.md`).

## 2. Related work

*Whole-recording recognition.* Chordia and Rae (ISMIR 2007) and Koduri, Gulati, Rao and Serra (JNMR 2012) classify by pitch-class distributions; Gulati et al. (ISMIR 2016) introduce the time-delayed melody surface (TDMS) and report 87% on the 480-recording, 40-rāga CMD Carnatic set and 98% on a 300-recording, 30-rāga Hindustani set, which we reuse as W2 on windows.
Sequential pitch distributions (Narasinh and Senthil Raja, AIMC 2023) keep note-transition direction and report 88.13% on Carnatic; deep sequence models (DeepSRGM, arXiv 2402.10168; a TDNN+LSTM model, arXiv 2405.16000; a preprocessing/architecture comparison, arXiv 2212.05335) likewise classify whole recordings or clips.
*Real-time systems.* Chordia and Rae (NIME 2008) describe real-time raag recognition that shows how likely each raag is as time passes, and Rāgawise (Gulati, Ganguli, Gupta and Srinivasamurthy, ISMIR 2015 hackathon) plots the evolving likelihood of each rāga from live vocals (Hindustani, demo no longer maintained).
From their abstracts, neither scores where a rāga *changes*.
*Segmentation.* Sarala and Murthy (ISMIR 2013) and Padi and Murthy (Sādhanā 2018) segment concerts into items using applause and merge segments by rāga classification, assuming one rāga per item; structural segmentation of Dhrupad alap (Rao, Vinutha and Rohit, TISMIR 2020) and of Carnatic music (Kulkarni 2014) does not identify rāgas.
In a literature pass of 16 papers (`docs/related_work.md`; web search of abstracts and metadata, no full texts read) we found none that performs windowed rāga recognition with temporal decoding and evaluates rāga-change boundaries within recordings.
This is an absence in a limited search, not proof, and should be re-checked with full-text Scholar and ISMIR-archive searches before submission.

## 3. Data and benchmarks

**Data.** CMD Carnatic features (480 recordings, 63 artists) and Saraga Carnatic 1.5 (249 recordings; 197 with usable pitch and tonic, 24 artists, 26 concerts) [`results/inventory_summary.md`].
Both provide pitch tracks (hop 4.4 ms, unvoiced = 0 Hz) and a tonic; we resample cents-above-tonic to 100 Hz with an explicit voiced mask. Rāga names are normalised across sources (`configs/raga_aliases.yaml`).
The rāga set has K = 40 rāgas with at least 8 recordings and 3 artists. Artists are split into train, val and test with no artist in more than one (tested by `tests/test_splits.py`):
train 53 artists / 328 recordings, val 7 / 94, test 7 / 107 [`results/splits_summary.md`].

**B1: synthetic same-concert splices.** From val and test Saraga concerts, each sequence concatenates 2 to 5 excerpts (60 to 240 s each) from distinct in-set rāgas of the same concert, skipping non-melodic sections.
A sequence is *easy* if all neighbouring rāga pairs share at most 4 svaras, *hard* if at least one pair shares 6 or more, otherwise *mid*, using `configs/scales.yaml` (svaras per rāga from external references, with sources per entry).
Test: 200 easy, 100 mid, 200 hard sequences from 8 concerts (hard from 6) [`results/b1_summary.md`]. Sequences are rebuilt byte-identically from the manifest.
Because sequences reuse a limited set of source items, we report concert-level bootstrap intervals alongside sequence-level ones, and use the former when they differ.

**B2: real concerts.** The eight test concerts, items concatenated in performance order (MusicBrainz track order), non-melodic sections trimmed; 53 reference boundaries in total.
Items outside the 40-rāga set, or rāgamālikas, are labelled `other`; neighbouring items with the same label are merged, and `other` frames are excluded from accuracy [`results/b2_summary.md`].

**B3 (rāgamālika case study)** was planned but not carried out (Section 8).

## 4. Method

**Windows.** Cents above tonic, windows of length L (default 30 s) with hop 5 s; a window is valid if at least 40% of its frames are voiced; invalid windows get an uninformative emission in the decoder.

**Classifiers.** *W1:* 12-bin pitch-class histogram (L1-normalised after adding 1e-6 to every bin) with class-weighted multinomial logistic regression (C chosen on val). *W2:* TDMS per window with τ = 0.3 s, 24 bins, power compression 0.75 and Gaussian smoothing σ = 1 bin (Gulati et al. used 120 bins; our smaller grid was selected on val, see `docs/decisions.md`), flattened, logistic regression with C = 0.01.
Hyperparameters were selected on val only. We did not train the optional CNN (W3).

**Calibration.** Temperature scaling on val *increased* expected calibration error for both models (W1 0.042 to 0.095, W2 0.047 to 0.074), because both are under-confident (T > 1 cannot reduce it) [`results/calibration.md`].
Per the plan's backup we use the raw softmax posteriors and a tuned posterior exponent β inside D3.

**Decoders.** D0 argmax per window; D1 majority filter (width 5); D2 novelty: symmetric KL between the mean W1 histograms of the 3 windows before and after each point, peak-picked, segments labelled by mean posterior;
D3: an HMM over (rāga, sub-state) chains so that every segment lasts at least `min_dur` seconds, emission = β·log p(r|x) − log prior (uniform), switching probability ε; Viterbi offline and forward filtering online.
D3's ε, `min_dur` and β, and D2's threshold, were tuned on B1-val by boundary F@5 and then frozen and committed before any test run (`configs/decoders_frozen.yaml`: W2 uses ε = 0.01, `min_dur` = 60 s, β = 0.25; W1 ε = 0.1, 60 s, 0.25).

**Metrics.** Boundary precision, recall and F at ±5 s and ±10 s (one-to-one matching as in `mir_eval`), pooled over sequences; frame accuracy on a 1 s grid excluding `other`; segment accuracy (label of the most-overlapping estimated segment);
false switches per 10 minutes. Confidence intervals: 1,000 bootstrap resamples over sequences (B2: over concerts), plus a concert-cluster bootstrap on B1; paired bootstrap for decoder differences.
Latency (online): for each reference boundary into an in-set rāga, the time from the boundary until the online prediction equals the new rāga for two consecutive hops, where a hop's prediction is available when its window *ends*;
no match within 60 s is a miss (reported, not dropped).

## 5. Results

**E1: window classifiers (val).** With L = 30 s and chance 1/40 = 2.5%, W1 reaches 50.4% window accuracy (macro-F1 0.446; recording-level majority vote 68.1%) and W2 66.5% (macro-F1 0.614; majority vote 85.1%) [`results/e1_w1.json`, `results/e1_w2.json`].
W2 is clearly better, and the plan's gate (best window accuracy at least 60%) is met by W2 only. W1's 24-bin variant failed the recording-level check and its 12-bin variant is used.

**E2: main test evaluation (frozen configs, run once).** Boundary F@5 (95% CI) [`results/e2_main.md`]:

| Benchmark | W2 + D0 | W2 + D1 | W2 + D2 | **W2 + D3** | W1 + D3 |
|---|---|---|---|---|---|
| B1-easy (200) | 0.131 | 0.181 | 0.143 | **0.474** [0.419, 0.529] | 0.476 [0.422, 0.528] |
| B1-hard (200) | 0.159 | 0.212 | 0.161 | **0.427** [0.383, 0.472] | 0.396 [0.349, 0.441] |
| B1-mid (100) | 0.144 | 0.191 | 0.179 | **0.475** [0.402, 0.545] | 0.407 [0.336, 0.475] |
| B2 (8 concerts) | 0.025 | 0.036 | 0.050 | **0.329** [0.266, 0.420] | 0.256 [0.209, 0.325] |

D3 beats D0 and D1 on every benchmark for both classifiers: paired differences in F@5 are +0.21 to +0.34 versus D1 or D0, and none of the 1,000 resamples has D3 not better (concert-cluster intervals stay above zero as well).
D0 and D1 have high recall but very low precision (D0 makes about 25 false switches per 10 minutes against about 0.7 to 2.0 for D3). With W2 + D3, frame accuracy is 0.738 (easy), 0.748 (hard) and 0.903 (B2); F@10 is 0.82, 0.74 and 0.47.
Hard sequences are only slightly harder than easy ones for W2 + D3 (F@5 0.427 versus 0.474, with overlapping concert-level intervals); W2 is higher than W1 on B2 (0.329 versus 0.256) and on hard sequences (0.427 versus 0.396), but those intervals overlap; the two tie on easy sequences (0.474 versus 0.476).
D2 is weak (F@5 about 0.14 to 0.18 on B1). An independent 20-line script recomputes boundary F for five sequences and matches [`scripts/check_e2_independent.py`].

**E3: window length (W2, D3 re-tuned per L on B1-val, test run once).** [`results/e3_e4_summary.md`, `results/figures/e3_f_vs_L.png`]

| L (s) | val window acc. | B1-easy F@5 | B1-hard F@5 | B2 F@5 | B1-easy F@10 |
|---|---|---|---|---|---|
| 10 | 0.482 | **0.766** | **0.712** | 0.311 [0.186, 0.457] | 0.869 |
| 20 | 0.614 | 0.641 | 0.583 | **0.365** [0.298, 0.467] | 0.879 |
| 30 | 0.665 | 0.474 | 0.427 | 0.329 [0.259, 0.429] | 0.819 |
| 60 | 0.736 | 0.261 | 0.263 | 0.160 [0.110, 0.227] | 0.456 |

Longer windows classify better, but boundary F@5 falls steadily as L grows: shorter windows localise changes better. B1 gains from L = 30 to 10 are well outside the intervals; on B2 the intervals for L = 10 to 30 overlap, and only L = 60 is clearly worse.
The default L = 30 s used in E2 is therefore not the best for boundaries on B1; we did not re-tune it on test.

**E4: online latency (D3 online).** Median seconds [IQR] and miss rate (misses = no detection within 60 s); concert-level CIs are in `results/e4_latency.csv`.

| L (s) | B1-easy median | B1-easy miss | B1-hard median | B1-hard miss | B2 median (35 boundaries) | B2 miss |
|---|---|---|---|---|---|---|
| 10 | 26.2 [20.0, 37.6] | 0.27 | 27.0 [20.1, 38.2] | 0.34 | 24.9 | 0.23 |
| 20 | 33.9 [28.1, 40.5] | 0.37 | 34.9 [27.7, 44.6] | 0.38 | 31.4 | 0.26 |
| 30 | 40.3 [34.9, 46.5] | 0.41 | 40.1 [33.4, 48.7] | 0.41 | 37.4 | 0.26 |
| 60 | 51.8 [45.4, 56.1] | 0.66 | 49.9 [43.3, 55.5] | 0.64 | 52.3 | 0.43 |

Latency grows with window length (a window's prediction exists only after it ends) and many changes are not confirmed within a minute. Plain online argmax (D0) is faster (median 16 to 26 s for L = 10 to 30 on B1-easy)
but produces many false switches, so D3's slower, steadier output is a trade-off rather than a free improvement (`results/e4_latency.csv`; CDFs in `results/figures/e4_latency_cdf_*.png`).

## 6. Analysis (E6)

On the frozen W2 + D3 (L = 30) predictions [`results/e6_errors.md`, `results/e6_tables.md`]:
(i) most errors at ±5 s are near misses: on B1, 47.7% of estimated boundaries are correct at ±5 s, 33.0% fall 5 to 10 s from a true boundary and 19.3% are further than 10 s, with no systematic delay (median +1.1 s).
(ii) Boundary recall falls gently as neighbouring rāgas share more svaras (0.467 for at most 3 shared, to 0.314 for 7 or more, at ±5 s; `results/figures/e6_error_vs_shared.png`), while precision does not depend on the shared svaras of the two predicted rāgas.
(iii) The predicted label 30 s into a new B1 segment is correct for 0.77 of found and 0.68 of missed boundaries: placing the switch is harder than naming the rāga.
(iv) On B2 only a quarter of estimated boundaries are correct; about a third fall inside `other` blocks, where the closed-set classifier must name one of 40 rāgas and D3 switches between invented ones
(`results/figures/e6_worst_b2_*.png`). Because B2 merges neighbouring `other` items, 6 of those 32 false boundaries lie on real, unscored item changes, so precision is somewhat understated; the other 26 are not near any item change.
(v) Some rāga pairs are almost never found at ±5 s (e.g. harikambhoji/mohanam 0.06, n = 16), a plausible subset effect that we did not test; each pair mostly comes from one concert, so pair-level numbers are confounded with the concert.

## 7. Limitations

- **B3 missing.** No human-annotated rāgamālika evaluation exists (no annotators; Section 8). The case-study claim of the project is therefore untested; the only rāgamālika items are two `other` items in B2.
- **Synthetic sharpness.** B1 splices excerpts from different items, so transitions are abrupt compared with within-item rāga changes in a rāgamālika, and B1's sequences reuse a few source items (hence the concert-level intervals).
- **Closed set.** The classifier cannot say "none of the 40". B2 `other` items and any rāga outside the set produce spurious switches; an explicit `other` class was not part of the plan. B2 itself is small: 8 concerts, 53 boundaries, so its intervals are wide.
- **Provided tonic.** Tonics come from the datasets; a wrong tonic would shift every pitch class. We checked Sa/Pa peaks per recording and excluded recordings with a bad tonic (`configs/excluded_recordings.yaml`), but we did not study tonic-estimation errors.
- **Scale labels.** The easy/hard labels use `configs/scales.yaml`: 17 of 40 entries from memory were corrected against external sources, 11 still carry a `verify` flag and one (sencurutti) is unresolved.
- **Window-length and latency conventions.** Latency uses the window-end convention and a 60 s cutoff; D3 online is forward filtering without back-tracking, so it does not enforce the minimum duration.
- **Absolute accuracy** is modest, and the results rely on a single seed/split; test was used once per frozen configuration (E2, E3/E4), and the error analysis is descriptive.
- **Related work** is a limited web search of abstracts and metadata (16 papers, no full texts read); the claim that no prior work evaluates rāga-change boundaries is an absence of evidence in that search.

## 8. Conclusion and status

A windowed classifier with an HMM decoder that enforces a minimum segment duration recovers rāga changes far better than argmax or majority filtering (F@5 +0.21 to +0.34), but boundary accuracy is still moderate
(0.43 to 0.48 at 30 s windows on B1, up to 0.77 at 10 s; 0.33 on B2) and online detection takes tens of seconds. Window length trades classification accuracy against localisation and latency.
Open work: a human-annotated rāgamālika benchmark (B3, dropped here per the plan's backup because no annotators were available), an explicit `other`/open-set class, and a broader related-work check.

## Reproducibility

Code, configs, manifests (IDs, cut times, seeds), results and docs are in the repository; raw data is not and must be fetched from Zenodo (CMD record 7278506, Saraga Carnatic 1.5 record 4301737). See `README.md` (`make reproduce-eval`, `make reproduce`).
