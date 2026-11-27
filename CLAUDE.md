# CLAUDE.md — Rāga Tracking Project

Read this file fully before any work in this repo. `PLAN.md` is the source of truth for scope, tasks, gates and backups. This file tells you **how** to work here: the rules, layout, conventions, task prompts, checks and fallbacks.

---

## 1. Project in one paragraph

We recognise **which Carnatic rāga is being sung at every moment** of a recording and **where it changes**. We use only open pitch tracks and tonics (CMD features + Saraga Carnatic). Pipeline: tonic-normalised pitch → sliding windows (default 30 s, hop 5 s) → window classifier (W1 histogram, W2 time-delayed melody surface, W3 optional CNN) → calibrated posteriors → temporal decoder (D0 argmax, D1 majority filter, D2 novelty change-points, D3 HMM/Viterbi with minimum duration) → segments. We evaluate on three benchmarks:
- **B1:** synthetic same-concert splices
- **B2:** real concerts in item order
- **B3:** small self-annotated rāgamālika set

Metrics: boundary F, frame and segment accuracy, false switches, and online detection latency.

---

## 2. Non-negotiable rules

1. **Artist-disjoint splits.** No artist appears in more than one of train/val/test. `tests/test_splits.py` must pass before any training.
2. **No tuning on test.** Choose hyperparameters on val (classifier val, or B1-val for decoders) only. Test runs happen once per frozen config, from `scripts/run_final.py`, after the config is committed.
3. **No raw data in git.** `data/` and `models/` are gitignored. Never redistribute Saraga or CMD audio, pitch or tonic files. Release only code, manifests (IDs + cut times + seeds) and our own annotations.
4. **Inspect, don't assume, file formats.** Pitch file columns, hop size, tonic units and rāga name spellings differ between sources. Read a sample and print its shape before writing a parser. Record what you found in `docs/data_formats.md`.
5. **Determinism.** Every random step takes a seed from the config. Benchmarks rebuild byte-identically from their manifests.
6. **Never show model output to annotators.** B3 annotation tools must not display predictions.
7. **Every number in `results/` is produced by a script** that is committed, with the command that produced it written in the result file's header.
8. **Report negative results.** If D3 doesn't beat D1, write it up honestly. Don't tweak until it looks good on test.

---

## 3. Repo layout

```
raga-tracking/
├── CLAUDE.md, PLAN.md, README.md, Makefile, pyproject.toml
├── configs/
│   ├── default.yaml          # paths, seeds, window L/hop, model + decoder params
│   ├── raga_set.yaml         # final K rāgas (T2)
│   ├── raga_aliases.yaml     # name normalisation across sources (T1)
│   └── scales.yaml           # svarasthāna set per rāga, for easy/hard (T2)
├── data/                     # gitignored
│   ├── raw/{cmd,saraga}/
│   ├── interim/pitch/*.npz   # cached cents + voiced mask at 100 Hz
│   └── processed/{inventory.csv,splits.csv,b1/,b2/}
├── src/raga_tracking/
│   ├── io.py                 # loaders for CMD / Saraga
│   ├── pitch.py              # Hz→cents, resample, voiced mask
│   ├── windows.py            # windowing
│   ├── features/{histogram.py,tdms.py,sequence.py}
│   ├── models/{linear.py,cnn.py,calibration.py}
│   ├── decode/{argmax.py,majority.py,novelty.py,hmm.py}
│   ├── bench/{synthetic.py,concerts.py}
│   └── eval/{metrics.py,latency.py,bootstrap.py}
├── scripts/                  # numbered entry points 01_… to 15_…
├── annotations/ragamalika/   # our B3 CSVs + README (released)
├── results/                  # tables (.csv/.md) and figures (.png)
├── docs/{related_work.md,data_formats.md,decisions.md}
├── paper/
└── tests/
```

Log every non-obvious choice (threshold changes, a backup plan invoked, a dropped task) as one dated line in `docs/decisions.md`.

---

## 4. Environment and commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy pandas scikit-learn matplotlib pyyaml joblib \
            mirdata mir_eval librosa torch tqdm pytest
# optional: compiam essentia (not needed: we use provided pitch + tonic)
```

| Command | Does |
|---|---|
| `make test` | `pytest -q` |
| `make inventory` | T1 |
| `make splits` | T2 |
| `make cache` | T3 |
| `make train W=w1` | T4/T5/T6 |
| `make bench` | T8 + T9 |
| `make tune` | T10 decoder tuning on B1-val |
| `make final` | T11 frozen test run |
| `make reproduce` | everything from cached data to the main table |

Code style: Python 3.10+, type hints, docstrings on public functions, functions < 60 lines, no notebooks as the only source of a result.

---

## 5. Key definitions (must match PLAN.md §3)

- `cents = 1200 * log2(f_hz / tonic_hz)` for `f_hz > 0`; unvoiced frames are masked, never set to 0 cents.
- Pitch class = `cents % 1200` (Python's `%` already returns a non-negative result for negative cents).
- Window centre time = `start + L/2`. A window is valid if voiced fraction ≥ 0.4.
- Histogram smoothing: add `1e-6` to every bin, then renormalise (prevents log(0) in KL).
- Boundary scoring: `mir_eval.segment.detection(ref_intervals, est_intervals, window=5.0)` and `window=10.0`. Intervals in seconds, shape (n, 2).
- Latency: online prediction must equal the new rāga for 2 consecutive hops (10 s). Latency = time of the first of those hops − reference boundary. No match within 60 s = miss.
- D3 minimum duration via chained sub-states: `m = round(min_dur / hop)`; states = K × m; inside a chain move forward with probability 1; from the last sub-state stay with 1−ε or jump to the first sub-state of another rāga with ε/(K−1). Emission log-score = `log p(r|x) − log prior(r)` for every sub-state of rāga r. Invalid (silent) windows give uniform emission.

---

## 6. Known pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| Tonic in wrong units or octave | No peak at pitch class 0 / ~702 | Check tonic file format; compare with the pitch median; fix units |
| Unvoiced frames kept as 0 Hz | log2(0) → −inf, NaNs | Mask f ≤ 0 before conversion |
| Different hop sizes across sources | Windows contain different numbers of frames | Resample everything to 100 Hz in T3 |
| Rāga spelling variants (Kalyani / Kalyāṇi / Kalyaani) | Same rāga counted twice | `raga_aliases.yaml`, normalise lowercase ASCII |
| Leakage through splices | B1 accuracy suspiciously high | Assert that no B1 source item's artist is in train |
| Over-confident posteriors | D3 behaves like D0 | Temperature calibration (T7); posterior^β |
| Tuning on test | Inflated results | Configs frozen + committed before `make final` |
| Applause/percussion inside items | Spurious boundaries | Trim using Saraga sections; voiced-fraction threshold |

---

## 7. Task prompts

Each prompt below is self-contained. Use it as the instruction for that task. Every task follows the same loop:

1. Read the matching task in `PLAN.md §8` and this section.
2. Inspect real data before writing code.
3. Write tests first for anything with a definite expected answer.
4. Implement, run, save outputs to the listed paths.
5. Run **all Checks**. If any fails, apply the **Backup** in order, log it in `docs/decisions.md`, and re-run the checks.
6. Finish with a short report: what was produced, check results, any backup invoked, open questions.

---

### T0 — Setup and literature check

**Prompt**
> Create the repo structure in CLAUDE.md §3 with `pyproject.toml`, `Makefile` targets from §4, a gitignore for `data/` and `models/`, and an empty passing test. Then search Google Scholar, arXiv and the ISMIR archives for: "ragamalika segmentation", "raga change detection", "raga boundary detection", "continuous raga recognition", "raga tracking audio". For each relevant hit since 2015, add a row to `docs/related_work.md`: citation, link, task, data, whether it evaluates boundaries over time, overlap with us (none / partial / full).

**Checks**
- [ ] `make test` passes
- [ ] `docs/related_work.md` has ≥ 10 rows, including Koduri 2012, Gulati 2016 (TDMS), Chordia & Rae 2008 (real-time), Rāgawise, DeepSRGM
- [ ] No row marked "full overlap"

**Backup**
1. Partial overlap → note how we differ (benchmarks B2/B3, latency, decoder comparison) in `docs/decisions.md`; continue.
2. Full overlap → stop and report to the team. The fallback project is the grahabhēdam stress test.

---

### T1 — Data inventory

**Prompt**
> Download CMD features and Saraga Carnatic using mirdata (`compmusic_raga`, `saraga_carnatic`); if a loader fails, download from Zenodo manually. Open 3 sample pitch and tonic files per source and write their exact formats (columns, units, hop) to `docs/data_formats.md`. Write `scripts/01_inventory.py` that produces `data/processed/inventory.csv` with columns: `rec_id, source, artist, concert, item_order, raga_raw, raga, n_ragas, duration_s, hop_s, has_pitch, has_tonic, has_sections`. Build `configs/raga_aliases.yaml` to normalise names. Write `results/inventory_summary.md` with counts per rāga (recordings, artists) and the list of rāgamālika candidates.

**Checks**
- [ ] Every listed pitch file parses; failures logged with reason
- [ ] `hop_s` is constant within each source (report the values)
- [ ] Tonic present for ≥ 95% of recordings
- [ ] No two alias groups collapse different rāgas (manual review of the alias file)
- [ ] Rāgamālika candidate count reported

**Backup**
1. mirdata loader broken → manual Zenodo download + minimal parser in `io.py`.
2. CMD features unavailable → Saraga only; note the reduced K in decisions.
3. Concert or item order missing in Saraga metadata → derive order from track numbering in the file structure; if impossible, B2 is dropped (see T9 backup).

---

### T2 — Rāga set and artist-disjoint splits

**Prompt**
> From `inventory.csv`, choose rāgas with ≥ 8 recordings and ≥ 3 distinct artists. Assign artists to train/val/test (≈70/15/15) with a greedy, seeded procedure that (a) gives each chosen rāga ≥ 1 artist in each split where possible and (b) favours Saraga concerts with ≥ 3 in-set items for val/test. Write `configs/raga_set.yaml` and `data/processed/splits.csv`. Write `configs/scales.yaml` with each rāga's svarasthāna set (ārōhaṇa ∪ avarōhaṇa, as 12-position indices) from a standard reference, citing the source in the file. Add `tests/test_splits.py` asserting artist disjointness.

**Checks**
- [ ] **Gate G1:** K ≥ 15
- [ ] `test_splits.py` passes
- [ ] ≥ 8 test concerts with ≥ 3 in-set items
- [ ] Every rāga in `raga_set.yaml` has a scale in `scales.yaml`

**Backup** (in order)
1. Relax to ≥ 6 recordings, ≥ 2 artists.
2. Accept K ≥ 10 and record it.
3. Too few test concerts → artist-disjoint at recording level, and plan B2 as cross-validation over concerts (T9 backup 1).

---

### T3 — Pitch preprocessing and windowing

**Prompt**
> Implement `pitch.py`: load pitch + tonic, convert to cents (mask f ≤ 0), resample onto a 100 Hz grid with a boolean voiced mask, cache to `data/interim/pitch/{rec_id}.npz` (`cents`, `voiced`, `t0`, `fs`). Implement `windows.py`: given cents/voiced and (L, hop), return window arrays, centre times and voiced fractions. Write tests first (see Checks). Then plot pitch-class histograms for 3 random recordings to `results/figures/t3_sanity.png`.

**Checks**
- [ ] Test: 220 Hz at tonic 220 → 0 cents; 440 → 1200; pitch class of 1200 is 0; −100 cents → pitch class 1100
- [ ] Test: number of windows = floor((dur − L)/hop) + 1
- [ ] Test: masked frames never appear in features
- [ ] Sanity plot shows clear peaks near 0 and ~700 cents for all 3 recordings

**Backup**
1. Missing Sa/Pa peaks → check tonic units and octave in `docs/data_formats.md`; fix the parser.
2. Recording still bad → exclude it, list it in `docs/decisions.md`.

---

### T4 — W1 histogram classifier

**Prompt**
> Implement `features/histogram.py` (n_bins configurable, default 24; L1 + 1e-6 smoothing). Train multinomial logistic regression on train-split windows, sampling at most 200 valid windows per recording (seeded). Tune C ∈ {0.01, 0.1, 1, 10, 100} on val windows. Save the model and `results/e1_w1.json` with val window accuracy, macro-F1, recording-level majority-vote accuracy, and a confusion matrix figure.

**Checks**
- [ ] Val window accuracy ≫ 1/K (report both)
- [ ] Recording-level majority vote ≥ 60% on val
- [ ] No test data touched (grep the script for split usage)

**Backup**
1. Try 12 and 48 bins; L = 60 s.
2. Try k-NN with symmetric KL (the base paper's method) as an alternative W1.
3. Add class weighting if per-class recall is very uneven.

---

### T5 — W2 time-delayed melody surface classifier

**Prompt**
> Implement `features/tdms.py`: for each window, build a 2D histogram over (pc(t), pc(t+τ)) using voiced pairs only; apply power compression (default 0.75) and Gaussian smoothing (σ in bins, default 1); flatten and L1-normalise. Check the defaults against Gulati et al., ISMIR 2016, and note any differences in `docs/decisions.md`. Tune τ ∈ {0.2, 0.3, 0.5} s and bins ∈ {24, 40, 60} on val with logistic regression. Save `results/e1_w2.json` in the same format as W1.

**Checks**
- [ ] Unit test: constant pitch → all mass on one diagonal cell
- [ ] Unit test: shifting all pitch by exactly one bin rolls the surface along both axes
- [ ] **Gate G2:** max(W1, W2) val window accuracy ≥ 60% at L = 30 s
- [ ] W2 ≥ W1, or the gap is explained

**Backup (if G2 fails, in order)**
1. Default L = 60 s.
2. Reduce K to the most-recorded rāgas until ≥ 60%.
3. Both; record which in decisions.

---

### T6 — W3 CNN (optional; cut first if behind)

**Prompt**
> Implement `features/sequence.py` (cents resampled to 50 Hz, masked) and `models/cnn.py`: 4 Conv1d blocks (kernel 5, channels 32→64→128→128, BatchNorm, ReLU, stride-2 pooling), masked mean pooling, linear head. Augmentation: random crop within the window, global shift ±20 cents. AdamW lr 1e-3, weight decay 1e-4, batch 64, early stopping on val macro-F1 (patience 5). Seeded. Save `results/e1_w3.json`.

**Checks**
- [ ] Overfits a 64-window subset to ~100% (pipeline sanity check)
- [ ] Training ≤ 2 GPU-hours on a Colab T4
- [ ] Beats W2 on val, else excluded from the main table (note it)

**Backup**
1. Underperforms → try L = 60 s, or a GRU over CNN features.
2. Still behind or out of time at the end of week 3 → cut W3; log it.

---

### T7 — Calibration

**Prompt**
> Implement temperature scaling in `models/calibration.py` (fit T on val log-likelihood). Apply to W1/W2/W3. Report ECE (15 bins) before and after, and the median max-probability, in `results/calibration.md`.

**Checks**
- [ ] ECE decreases for each model
- [ ] Median max-probability < 0.99 after calibration

**Backup**
1. No improvement → use posterior^β in D3, with β tuned on B1-val in T10.

---

### T8 — B1 synthetic benchmark

**Prompt**
> Implement `bench/synthetic.py`. From val- and test-split Saraga concerts, generate sequences of 2–5 excerpts (60–240 s each) from items in distinct in-set rāgas of the same concert. Skip non-melodic sections using Saraga section annotations where present; otherwise skip excerpts with voiced fraction < 0.6. Concatenate cached cents/voiced arrays with re-based time. Label each sequence easy (all neighbouring pairs share ≤ 4 svaras, using `scales.yaml`) or hard (≥ 1 pair shares ≥ 6). Save `.npz` per sequence plus `b1_manifest.json` (source rec_ids, cut times, seed, label). Target 500 test + 200 val.

**Checks**
- [ ] Test: no source artist in the train split
- [ ] Test: rebuilding from the manifest is byte-identical
- [ ] All segments ≥ 60 s; neighbouring rāgas differ
- [ ] ≥ 100 easy and ≥ 100 hard test sequences

**Backup**
1. Too few hard → define hard as ≥ 5 shared svaras.
2. Too few same-concert combinations → allow same-artist across concerts as a separate subset "B1-xc", reported separately.

---

### T9 — B2 concert benchmark

**Prompt**
> Implement `bench/concerts.py`. For each test-split Saraga concert, concatenate items in performance order, trimming non-melodic sections. Label items whose rāga is out-of-set or rāgamālika as `other`. Save `.npz` + `b2_manifest.json`. Plot reference timelines for 2 concerts to `results/figures/t9_timelines.png`.

**Checks**
- [ ] Item order matches the metadata for every concert
- [ ] ≥ 8 concerts and ≥ 40 boundaries in total
- [ ] Timelines look plausible (durations, no zero-length segments)

**Backup**
1. Too few test concerts → evaluate B2 by artist-grouped cross-validation over all Saraga concerts, retraining the chosen classifier per fold.
2. No section annotations → trim only leading and trailing unvoiced regions.

---

### T10 — Decoders D0–D3

**Prompt**
> Implement `decode/argmax.py` (D0), `majority.py` (D1, width 5), `novelty.py` (D2: symmetric KL between mean W1 histograms of the k windows before and after each point, k = 3; peak-pick with a threshold tuned on B1-val; label segments by mean posterior), and `hmm.py` (D3 exactly as in CLAUDE.md §5, log-space, numpy-vectorised; offline Viterbi and online forward filtering). Each decoder returns `(intervals, labels)`. Tune D3's ε ∈ {1e-4, 1e-3, 1e-2, 5e-2, 1e-1}, min_dur ∈ {30, 45, 60, 90} s (and β if T7 used the backup) on **B1-val** by boundary F@5. Save `results/d3_tuning.csv`.

**Checks**
- [ ] Test: clean posteriors with one switch → exactly one boundary at the correct window
- [ ] Test: isolated single-window flips are removed by D3
- [ ] Test: online filtering is causal (changing future posteriors leaves past outputs unchanged)
- [ ] Test: D3 never outputs a segment shorter than min_dur (except at sequence edges)
- [ ] D3 > D0 on B1-val boundary F@5

**Backup**
1. Misses short segments → lower min_dur, raise ε.
2. D3 ≈ D0 → posteriors are too sharp; revisit T7 or apply β.
3. Slow → profile; the state space is K·m (~300–400), so it should run in milliseconds per sequence when vectorised.

---

### T11 — Main evaluation (E2)

**Prompt**
> Freeze configs (commit them). Implement `eval/metrics.py` (boundary P/R/F at 5 s and 10 s via mir_eval, frame accuracy excluding `other`, segment accuracy, false switches per 10 min) and `eval/bootstrap.py` (1,000 resamples over sequences; paired bootstrap for D3 vs D1). Run every available W × D on B1-easy, B1-hard and B2 test **once** via `scripts/run_final.py`. Write `results/e2_main.csv` and a formatted `results/e2_main.md` with 95% confidence intervals.

**Checks**
- [ ] Config commit predates the result files (record the commit hash in the result header)
- [ ] An independent 20-line script recomputes boundary F for 5 sequences and matches
- [ ] Every cell has a confidence interval

**Backup**
1. D3 doesn't beat D1 → report it as-is; diagnose in T14 (calibration, min_dur, window length). **Do not re-tune on test.**

---

### T12 — Window length and latency (E3, E4)

**Prompt**
> E3: retrain the best classifier at L ∈ {10, 20, 30, 60} s (hop 5 s), re-tune D3 on B1-val for each L, evaluate on B1/B2 test. E4: implement `eval/latency.py` per CLAUDE.md §5 and compute online latency (median, IQR, miss rate) on B1 and B2, split easy/hard. Save CSVs and figures (latency CDF; boundary F vs L).

**Checks**
- [ ] Latency only computed at reference boundaries; misses counted, not dropped
- [ ] Figures have confidence bands
- [ ] Results consistent with E2 at L = 30 s

**Backup**
1. Behind schedule → L ∈ {20, 30, 60} only, latency on B1 only.

---

### T13 — B3 rāgamālika case study

**Prompt**
> From `inventory_summary.md`, list rāgamālika candidates. For each, find the section-wise rāga order in published lyrics or notation (e.g. karnatik.com) and cite the source. Prepare Sonic Visualiser sessions with the audio (if locally available under the Saraga licence) or the pitch track only, **without any model output**. Each annotator independently writes `annotations/ragamalika/{rec_id}.{annotator}.csv` with `start_s,end_s,raga,confidence,notes`. Compute inter-annotator boundary F@3 s, settle disagreements into `{rec_id}.csv`, flag disagreements > 10 s. Write `annotations/README.md` (protocol, sources, agreement). Evaluate the frozen best configuration; write `results/e5_ragamalika.md`.

**Checks**
- [ ] Rāga order in every CSV matches the cited notation
- [ ] Agreement reported per recording and overall
- [ ] ≥ 5 recordings annotated
- [ ] Model predictions were not visible during annotation (state this in the README)

**Backup**
1. Overall agreement F@3 s < 0.7 → evaluate B3 at ±10 s only and say so.
2. < 5 usable recordings → drop B3, add it to limitations.

---

### T14 — Error analysis (E6)

**Prompt**
> For the best configuration, bucket missed and false boundaries by the number of shared svaras between neighbouring rāgas (`scales.yaml`) and by rāga pair. For the 10 worst sequences on B2, plot the pitch track, window posteriors (top 3 rāgas), and reference vs predicted segments. Summarise patterns in `results/e6_errors.md`, each claim pointing to a figure or table.

**Checks**
- [ ] Every claim in `e6_errors.md` cites a figure or table
- [ ] Error rate vs shared-svara count is plotted

**Backup**
1. Short on time → do the shared-svara plot and 3 worst-case figures only.

---

### T15 — Write-up and release

**Prompt**
> Draft `paper/` (4–6 pages) following PLAN.md §11. Every number must come from `results/`. Write `README.md` with setup, data download steps (from the original sources), and `make reproduce`. Verify by cloning into a fresh directory and running `make reproduce` against cached data. Tag a release containing code, manifests and our annotations only.

**Checks**
- [ ] Fresh clone + `make reproduce` regenerates `e2_main.csv` within bootstrap noise
- [ ] No Saraga/CMD pitch, tonic or audio in the repo (`git ls-files` check)
- [ ] Limitations section covers: non-expert B3, synthetic sharpness, closed set, provided tonic

**Backup**
1. No suitable venue in time → course report + arXiv preprint.

---

## 8. Review checklist (run before any result is reported)

- [ ] Splits test passes; no train artist in any benchmark
- [ ] Results produced by a committed script; command + commit hash in the header
- [ ] Only val / B1-val used for tuning
- [ ] Seeds set; benchmarks rebuilt identically
- [ ] Confidence intervals present
- [ ] Decisions and backups logged in `docs/decisions.md`
- [ ] Figures readable: axis labels with units, legend, no default titles

## 9. Glossary

| Term | Meaning |
|---|---|
| Rāga | Melodic framework: svaras, their use, movement, gamakas, phrases |
| Rāgamālika | Composition moving through several rāgas in sequence |
| Svara / svarasthāna | Note name / its pitch position (12 per octave) |
| Tonic (Sa) | The performer's reference pitch; all pitch measured relative to it |
| Gamaka | Pitch oscillation or glide on a svara |
| TDMS | Time-delayed melody surface: 2D histogram of pitch at t vs t+τ |
| Item | One piece within a concert, normally in one rāga |
| ECE | Expected calibration error |
| Boundary F@w | F-measure of boundary detection with ±w s tolerance |
