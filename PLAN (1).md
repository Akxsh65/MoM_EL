# PLAN.md — Time-Resolved Rāga Recognition and Rāgamālika Segmentation

**Team:** Akash Sivaprasath (1RV23CS024) + teammate(s), CSE, RVCE
**Base paper:** Koduri, Gulati, Rao & Serra, *Rāga Recognition based on Pitch Distribution Methods*, JNMR (preprint 2012). §6.2 names rāgamālika as an open problem.
**Duration:** 8 weeks
**Constraint:** open data only. No audio-access requests, no external annotators.

---

## 1. Goal

Given a Carnatic recording's pitch track and tonic, output **which rāga is sung at every moment and where it changes**:

```
[(start_s, end_s, raga), (start_s, end_s, raga), ...]
```

### Research questions

- **RQ1.** How accurately can a rāga be recognised from short windows (10–60 s) instead of full recordings?
- **RQ2.** How much does temporal decoding (an HMM with a switch penalty and a minimum segment length) improve boundary accuracy over per-window predictions?
- **RQ3.** How quickly does an online system detect a rāga change (detection latency), and how does that depend on window length and on how similar the neighbouring rāgas are?
- **RQ4.** How large is the gap between synthetic splices, real concert sequences, and real rāgamālikas?

### Contributions

1. The first benchmarks for time-resolved rāga recognition, built only from open data:
   - **B1:** synthetic same-concert splices, easy and hard
   - **B2:** real concert sequences, with no annotation needed
   - **B3:** a small rāgamālika case study, notation-guided and annotated by the team
2. An evaluation protocol with boundary F-measure, frame and segment accuracy, false switches, and detection latency.
3. A comparison of three window classifiers × four decoders.
4. Released code, benchmark-building scripts and annotations.

### Non-goals

- No new pitch or tonic extraction. Use the provided features.
- No Hindustānī, no open-set or unseen-rāga detection, and no grahabhēdam study.
- No raw audio is redistributed.

---

## 2. Data (all open)

| ID | Dataset | Content used | Licence | Access |
|---|---|---|---|---|
| D-CMD | Indian Art Music Rāga Recognition Dataset, Carnatic part (CMD) | Pitch tracks + tonic + rāga label per recording; 480 recordings, 40 rāgas | Check the Zenodo record and cite it | Zenodo (features); mirdata `compmusic_raga` |
| D-SAR | Saraga Carnatic 1.5 | Pitch tracks, tonic, section annotations, metadata (rāga, artist, concert) | CC BY-NC-SA 4.0 | Zenodo 4301737; mirdata `saraga_carnatic` |

**Rules:**
- Splits are **artist-disjoint**: an artist appearing in any test set never appears in training.
- Tune hyperparameters on **validation only**. Each test set is touched once per final configuration.
- Never commit raw data. `data/` is gitignored.

---

## 3. Definitions

| Term | Definition |
|---|---|
| Cents | `1200 * log2(f / f_tonic)`; drop frames with f ≤ 0 (unvoiced) |
| Pitch class | `cents mod 1200` |
| Window | 30 s by default, hop 5 s; timestamp = window centre |
| Valid window | At least 40% voiced frames, else marked `silent` and skipped by the classifier (decoder carries the state through) |
| Boundary | A time where the reference rāga label changes |
| Hit | Estimated boundary within ±5 s (main) or ±10 s (secondary) of a reference boundary, one-to-one matching |
| Detection latency | Online mode: seconds from a reference boundary until the prediction equals the new rāga and **holds for 10 s (2 hops)** |
| Easy / hard pair | Neighbouring rāgas share ≤ 4 / ≥ 6 svarasthānas (from a hand-written scale table) |

---

## 4. System

```
pitch track (Hz) + tonic
  → cents above Sa, unvoiced dropped
  → windows (L s, hop 5 s)
  → window features  [W1 histogram | W2 TDMS | W3 raw sequence]
  → window classifier → p(rāga | window), temperature-calibrated
  → decoder [D0 | D1 | D2 | D3]
  → segments + boundaries
```

### Window classifiers

| ID | Features | Model | Notes |
|---|---|---|---|
| W1 | 24-bin octave-folded pitch histogram, L1-normalised, +1e-6 | Multinomial logistic regression (C tuned) | The paper's P_continuous, applied per window |
| W2 | Time-delayed melody surface: 2D histogram of (pc(t), pc(t+τ)) | Logistic regression on flattened surface | Start: τ = 0.3 s, 40 bins/octave, power compression 0.75, Gaussian smoothing; tune on val. **Verify defaults against Gulati et al., ISMIR 2016** |
| W3 | Cents sequence resampled to 50 Hz (L·50 frames), unvoiced → mask | 1D-CNN (4 conv blocks + masked mean pooling + linear) | Optional. First thing to cut |

### Decoders

| ID | Decoder | Details |
|---|---|---|
| D0 | Argmax per window | Lower bound |
| D1 | Argmax + majority filter | Filter width 5 windows |
| D2 | Novelty change-points (label-free) | Symmetric KL between W1 histograms of windows t−k and t+k; peak-pick; label each segment by mean posterior |
| **D3** | **HMM / Viterbi (proposed)** | K rāgas × m chained sub-states (minimum duration m·hop, default 45 s → m = 9); within-chain transitions 1.0; chain end: stay 1−ε, switch ε/(K−1); emission ∝ posterior / class prior; log-space. Offline = Viterbi, online = forward filtering |

---

## 5. Benchmarks

### B1 — Synthetic same-concert splices

- Source: Saraga concerts in the **test and val artist splits** only.
- For each sequence: pick 2–5 items in distinct in-set rāgas from one concert. Cut a contiguous excerpt of 60–240 s from each item's main body, skipping applause, silence and percussion solos using section annotations where available. Concatenate the pitch tracks with the timestamps re-based.
- Subsets: **easy** (all neighbouring pairs easy), **hard** (≥ 1 hard pair).
- Size: target 500 test + 200 val sequences. A fixed seed is saved with the manifest.

### B2 — Real concert sequences

- For each test-split Saraga concert: concatenate its items in performance order, after trimming non-melodic parts.
- Items whose rāga is out-of-set or rāgamālika are labelled `other`. They count for boundaries and are excluded from rāga accuracy.
- Ground truth = item boundaries + metadata rāga. **No annotation needed.**

### B3 — Rāgamālika case study (self-annotated)

- Candidates: Saraga Carnatic items whose rāga metadata is "Rāgamālika" or lists several rāgas.
- Rāga order: from published lyrics and notation (e.g. karnatik.com section-wise rāga lists).
- Boundaries: both teammates mark them independently in Sonic Visualiser from lyrics and pitch view. **Model output is not shown while annotating.**
- Agreement: boundary F at ±3 s between annotators. Disagreements are settled jointly, and disagreement > 10 s is flagged in the CSV.
- Size: 5–10 recordings. Reported as **"non-expert, notation-guided"**.

---

## 6. Evaluation

| Metric | Computation | Benchmarks |
|---|---|---|
| Window accuracy, macro-F1 | Per valid window | Classifier val/test |
| Boundary P / R / F @5 s, @10 s | `mir_eval.segment.detection(ref_intervals, est_intervals, window=w)` | B1, B2, B3 |
| Frame accuracy | Per-second label agreement, `other` excluded | B1, B2, B3 |
| Segment accuracy | % reference segments whose majority predicted label is correct | B1, B2, B3 |
| False switches / 10 min | Estimated boundaries not matched at ±10 s, per 10 min | B1, B2, B3 |
| Detection latency (median, IQR, miss rate at 60 s) | Online forward filtering | B1, B2 |

**Statistics:** 95% bootstrap confidence intervals over sequences (1,000 resamples). Paired bootstrap for D3 vs D1.

---

## 7. Experiments

| Exp | Content | RQ |
|---|---|---|
| E1 | Window classification: W1/W2/W3 at L = 30 s | RQ1 |
| E2 | Main table: {W1, W2, W3} × {D0, D1, D2, D3} on B1-easy, B1-hard, B2 | RQ2 |
| E3 | Window length L ∈ {10, 20, 30, 60} s with best W + D3 | RQ1, RQ3 |
| E4 | Online latency curves, easy vs hard pairs | RQ3 |
| E5 | B3 case study with best configuration | RQ4 |
| E6 | Error analysis: missed or false boundaries vs shared svaras; confusion of neighbouring rāgas | RQ2, RQ4 |

---

## 8. Tasks

Each task lists objective, steps, outputs, **checks** (must pass before moving on) and a **backup plan**.

### T0 — Setup and literature check (Week 1, day 1–2)

- **Objective:** working repo; confirm novelty.
- **Steps:**
  1. Create the repo with the layout in CLAUDE.md, the environment, pre-commit and pytest.
  2. Google Scholar and arXiv: "ragamalika segmentation", "raga change detection", "raga boundary", "continuous raga recognition", "raga tracking". Read every hit from 2015 onward.
- **Outputs:** `docs/related_work.md` (each hit: citation, what it does, overlap yes/no).
- **Checks:** `pytest` runs; no paper does windowed rāga recognition + temporal decoding + boundary evaluation on Carnatic.
- **Backup:**
  - If a close paper exists, reposition as a reproduction plus extension: add the latency study and B2/B3, and compare against their method.
  - If it fully overlaps, switch to the grahabhēdam stress test (see the earlier plan).

### T1 — Data inventory (Week 1)

- **Objective:** know exactly what data exists.
- **Steps:** download D-CMD features and D-SAR via mirdata. Write `scripts/01_inventory.py` producing per-recording rows: id, source, artist, concert, rāga(s), duration, has_pitch, has_tonic, has_sections, pitch hop size (inferred from timestamps).
- **Outputs:** `data/processed/inventory.csv`; `results/inventory_summary.md` (counts per rāga and per artist; number of rāgamālika candidates).
- **Checks:**
  - Every pitch file parses.
  - Hop size is consistent within each source.
  - Tonic is present for ≥ 95% of recordings.
  - Rāga names are normalised across sources (mapping table `configs/raga_aliases.yaml`).
- **Backup:**
  - mirdata download fails → download the Zenodo archives manually and write a minimal loader.
  - CMD features unavailable → Saraga only, with K reduced accordingly.
  - Rāga names unmatched → keep sources separate for training and use only Saraga for B1/B2.

### T2 — Rāga set and splits (Week 1)

- **Objective:** fix K and artist-disjoint train/val/test.
- **Steps:**
  1. Candidate rāgas = in-set across CMD+Saraga with ≥ 8 recordings and ≥ 3 artists.
  2. Split artists about 70/15/15 by greedy assignment so each rāga keeps ≥ 1 artist per split where possible; Saraga concerts with many items preferred for val/test.
  3. Write `configs/scales.yaml`: svarasthāna set per rāga, for easy/hard labelling.
- **Outputs:** `configs/raga_set.yaml`, `data/processed/splits.csv`.
- **Checks:**
  - **Gate G1: K ≥ 15.**
  - No artist in two splits (unit test).
  - Each test concert has ≥ 3 items with in-set rāgas.
- **Backup:**
  - K < 15 → relax to ≥ 6 recordings / ≥ 2 artists.
  - Still < 15 → proceed with K ≥ 10 and state it.
  - Too few test concerts → recording-level artist-disjoint split, and B1 built across concerts of the same artist (flag the tonic and timbre caveat).

### T3 — Pitch preprocessing and windowing (Week 2)

- **Objective:** one cached, uniform representation.
- **Steps:**
  1. Hz → cents above tonic; drop f ≤ 0.
  2. Resample to a common 100 Hz grid with a voiced mask.
  3. Window function with (L, hop) → arrays + centre times + voiced fraction.
  4. Cache per recording as `.npz`.
- **Outputs:** `src/raga_tracking/pitch.py`, `windows.py`; `data/interim/pitch/*.npz`.
- **Checks:** unit tests:
  - A synthetic 220 Hz tone at tonic 220 Hz → 0 cents.
  - 440 Hz → 1200 → pitch class 0.
  - Window count = floor((dur − L)/hop) + 1.
  - Plot 3 random recordings: pitch-class histograms show clear Sa (0) and Pa (~702) peaks.
- **Backup:**
  - No Sa/Pa peaks → the tonic is wrong or in other units: check tonic units (Hz vs cents vs MIDI) and octave.
  - Tonic unreliable for a recording → drop it and log it.

### T4 — W1 histogram classifier (Week 2)

- **Steps:** 24-bin histogram per valid window; logistic regression; tune C on val; train windows sampled up to 200 per recording to balance recordings.
- **Outputs:** `models/w1.joblib`; `results/e1_w1.json`.
- **Checks:**
  - Val window accuracy far above chance (1/K).
  - Recording-level majority vote ≥ 60% on val (sanity check that windows aggregate).
- **Backup:**
  - Low accuracy → try 12 and 48 bins, L = 60 s, k-NN with symmetric KL (the paper's exact method), class weighting.

### T5 — W2 TDMS classifier (Week 2–3)

- **Steps:** implement TDMS (2D pitch-class histogram at delay τ); compression + smoothing; logistic regression; tune τ ∈ {0.2, 0.3, 0.5} s, bins ∈ {24, 40, 60}.
- **Outputs:** `models/w2.joblib`; `results/e1_w2.json`.
- **Checks:**
  - **Gate G2: best of W1/W2 ≥ 60% window accuracy on val at L = 30 s.**
  - W2 ≥ W1, or the result is explained.
  - Unit test: a constant pitch gives a single diagonal cell.
- **Backup (if G2 fails):**
  - Reduce K to the most frequent rāgas until ≥ 60%.
  - Or set default L = 60 s.
  - Or both; record which.

### T6 — W3 CNN (Week 3, optional)

- **Steps:** masked 1D-CNN on 50 Hz cents sequences; augmentation: random crop within window, ±20-cent global shift (not full rotation); AdamW, early stopping on val macro-F1; Colab T4.
- **Outputs:** `models/w3.pt`; `results/e1_w3.json`.
- **Checks:** beats W2 on val or is dropped from the main table. Training under 2 GPU-hours.
- **Backup:** cut entirely if behind schedule at the end of week 3. The paper stands on W1/W2.

### T7 — Calibration (Week 3)

- **Steps:** fit a temperature per classifier on val; report ECE before and after.
- **Outputs:** `models/*_temperature.json`; `results/calibration.md`.
- **Checks:** ECE decreases; posteriors aren't saturated (median max-probability < 0.99).
- **Backup:** if calibration doesn't help, use posterior^β with β tuned in D3 instead.

### T8 — B1 synthetic benchmark (Week 4)

- **Steps:** `bench/synthetic.py` per §5; manifest JSON with source items, cut times, seeds, easy/hard flag.
- **Outputs:** `data/processed/b1/{val,test}/*.npz`, `b1_manifest.json`.
- **Checks:**
  - No train-split artist appears.
  - Every segment ≥ 60 s.
  - Neighbouring rāgas are distinct.
  - Rebuilding from the manifest is byte-identical (determinism test).
  - Easy/hard counts are both ≥ 100 test sequences.
- **Backup:**
  - Too few hard pairs → define hard as ≥ 5 shared svaras.
  - Too few same-concert items → allow same-artist, different-concert, and report that separately.

### T9 — B2 concert benchmark (Week 4)

- **Steps:** `bench/concerts.py`; order items by Saraga concert track order; trim non-melodic sections; label out-of-set/rāgamālika as `other`.
- **Outputs:** `data/processed/b2/*.npz`, `b2_manifest.json`.
- **Checks:**
  - Item order matches the metadata.
  - ≥ 8 test concerts and ≥ 40 boundaries in total.
  - A plot of the reference timeline for 2 concerts looks sane.
- **Backup:**
  - Too few test concerts → move B2 to cross-validation over concerts (artist-grouped folds, classifier retrained per fold).
  - No section annotations → trim only leading and trailing silence via the voiced mask.

### T10 — Decoders D0–D3 (Week 4–5)

- **Steps:** implement in `decode/`; D3 with chained sub-states for minimum duration; tune ε ∈ {1e-4 … 1e-1} and minimum duration ∈ {30, 45, 60, 90} s on B1-val.
- **Outputs:** `src/raga_tracking/decode/*.py`; `results/d3_tuning.csv`.
- **Checks:**
  - Unit tests:
    - A noiseless posterior sequence with one switch returns exactly one boundary at the right window.
    - A posterior with single-window flips is smoothed by D3.
    - The online filter is causal: changing the future doesn't change past outputs.
  - D3 beats D0 on B1-val boundary F@5.
- **Backup:**
  - D3 over-smooths (misses short segments) → lower the minimum duration and raise ε.
  - Viterbi too slow → K·m states is small; vectorise with numpy and check the complexity is O(T·(K·m)).

### T11 — Main evaluation E2 (Week 5)

- **Steps:** run all W × D on B1-easy, B1-hard and B2 test once; bootstrap confidence intervals; paired test D3 vs D1.
- **Outputs:** `results/e2_main.csv`, `results/e2_main.md` (formatted table).
- **Checks:**
  - Metrics recomputed by an independent small script on 5 sequences match.
  - No test-set tuning: the git log shows configs frozen before the run.
- **Backup:** if D3 doesn't beat D1, still report it. Analyse why (calibration, minimum duration) in E6. A negative result with a good benchmark is still a contribution.

### T12 — Latency and window length E3/E4 (Week 6)

- **Steps:** E3: retrain best W at L ∈ {10, 20, 30, 60}; E4: online D3 latency on B1/B2, split easy/hard.
- **Outputs:** `results/e3_window.csv`, `results/e4_latency.csv`, figures.
- **Checks:** latency computed only at matched boundaries; miss rate reported; curves have confidence bands.
- **Backup:** if behind schedule, run only L ∈ {20, 30, 60} and B1 latency.

### T13 — B3 rāgamālika case study (Week 6–7)

- **Steps:** list candidates from the inventory; fetch the section-wise rāga order from notation; both annotators mark boundaries independently (CSV: `start_s,end_s,raga,confidence,notes`); compute agreement; settle disagreements; evaluate the best configuration.
- **Outputs:** `annotations/ragamalika/*.csv`, `annotations/README.md` (protocol + agreement), `results/e5_ragamalika.md`.
- **Checks:**
  - Agreement F@3 s ≥ 0.7, else the boundaries are flagged as uncertain.
  - Rāga order matches the published notation for every item.
- **Backup:**
  - < 5 usable rāgamālikas → drop B3 and say so in the limitations.
  - Agreement low → evaluate at ±10 s only.

### T14 — Error analysis E6 (Week 7)

- **Steps:** bucket errors by shared-svara count and by rāga pair; inspect the 10 worst sequences with pitch plots + posteriors.
- **Outputs:** `results/e6_errors.md`, figures.
- **Checks:** each claim in the write-up has a figure or table behind it.
- **Backup:** none needed. Scale it down if time is short.

### T15 — Write-up and release (Week 8)

- **Steps:** paper (4–6 pages): intro, related work, benchmarks, method, results, limitations. Release code, manifests and annotations. Do **not** release Saraga audio or pitch; release scripts that rebuild from the original sources.
- **Outputs:** `paper/`, tagged release, `README.md` with a reproduce command.
- **Checks:** fresh clone + `make reproduce` regenerates the main table within tolerance.
- **Backup:** if no venue fits the timeline, submit as a course report + arXiv preprint.

---

## 9. Timeline

| Week | Tasks | Gate |
|---|---|---|
| 1 | T0, T1, T2 | G1: K ≥ 15 (or ≥ 10 with note) |
| 2 | T3, T4, T5 start | — |
| 3 | T5, T6 (optional), T7 | G2: window accuracy ≥ 60% at L = 30 s |
| 4 | T8, T9, T10 start | ≥ 100 easy + 100 hard B1 sequences; ≥ 8 B2 concerts |
| 5 | T10, T11 | D3 tuned on val only; main table frozen |
| 6 | T12, T13 start | ≥ 5 rāgamālikas or drop B3 |
| 7 | T13, T14 | — |
| 8 | T15 | `make reproduce` passes |

**Cut order if behind:** W3 → E3 (window study) → B3 → B2 latency. **Never cut:** W1/W2 + D0/D1/D3 on B1 and B2.

## 10. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Prior work found in T0 | Low | Reposition (T0 backup) |
| Too few rāgas with enough artists | Medium | Relax thresholds; K ≥ 10 |
| Window accuracy too low at 30 s | Medium | Longer windows, smaller K |
| Splices easier than real music | High | B2 + B3 measure the gap; report it |
| Saraga has few rāgamālikas | Medium | B3 optional; B2 carries the real-data claim |
| Non-expert annotation errors | Medium | Notation-guided, two annotators, agreement reported |
| Data leakage via artist | Medium | Unit test on splits; manifests checked |

## 11. Paper outline

1. Introduction — rāga changes over time; rāgamālika; the base paper's §6.2
2. Related work — distribution-based (Koduri 2012), TDMS / phrase-based (Gulati 2016), deep sequence (DeepSRGM), real-time demos (Chordia & Rae 2008, Rāgawise); none evaluate boundaries
3. Benchmarks B1–B3
4. Method — windows, W1–W3, calibration, D0–D3
5. Results — E1–E5
6. Analysis — E6
7. Limitations — non-expert B3, synthetic sharpness, closed set, provided tonic
8. Conclusion
