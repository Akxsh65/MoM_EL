# Raga tracking: which Carnatic raga is being sung, moment by moment

Time-resolved Carnatic raga recognition and change detection from open pitch tracks and tonics (CMD features + Saraga Carnatic).
Pipeline: tonic-normalised pitch -> sliding windows -> window classifier (W1 pitch-class histogram, W2 time-delayed melody surface)
-> temporal decoder (D0 argmax, D1 majority filter, D2 novelty change-points, D3 HMM/Viterbi with minimum duration) -> segments.
Evaluated on B1 (synthetic same-concert splices) and B2 (real concerts in item order). **B3 (ragamalika case study) was not done**:
it needs independent human annotators; see `annotations/README.md`.

Start with `paper/paper.md` for the write-up, `results/e2_main.md` for the main table, and `docs/decisions.md` for every non-obvious choice,
deviation and backup that was invoked. `PLAN.md` is the plan, `CLAUDE.md` the working rules.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
python -m pytest -q                                      # 45 tests, no data needed except where marked skip
```

## Data (not in this repository, and must not be redistributed)

Raw data is **not** in git (`data/` and `models/` are gitignored; `python scripts/check_release.py` enforces that no audio, pitch, tonic, weights or arrays are tracked).
Obtain it from the original sources and respect their licences:

1. **CMD (Carnatic Music Dataset, Raga recognition features)**: Zenodo record 7278506. Unzip so that `data/raw/cmd/RagaDataset/Carnatic/features/...` exists.
2. **Saraga Carnatic 1.5**: Zenodo record 4301737. We only need the non-audio files (pitch, tonic, sections, metadata, about 275 MB of the 14.4 GB archive);
   `python scripts/fetch_saraga_meta.py` extracts them from the remote zip into `data/raw/saraga/`. `scripts/pdl.py` is a resumable parallel downloader for the CMD archive.
3. **Concert item order**: `python scripts/01b_fetch_item_order.py` (MusicBrainz public API, 1 request/s) writes `data/raw/musicbrainz/`.

Formats we found in the real files (hop size, units, unvoiced = 0 Hz, rāga spellings) are in `docs/data_formats.md`.

## Reproduce

`make` targets wrap the scripts (on Windows without `make`, run the commands in the Makefile directly).

| Command | Does | Needs |
|---|---|---|
| `make test` | unit tests | nothing |
| `make reproduce-eval` | frozen B1/B2 test evaluation (E2), window length + latency (E3/E4), error analysis (E6), release check | cached pitch in `data/interim/`, trained models in `models/`, benchmark arrays in `data/processed/` |
| `make reproduce` | everything from raw data: inventory, splits, pitch cache, W1/W2 training, benchmarks, decoder tuning, tests | raw data above; **hours** (W2 training over its C grid dominates) |

Order and freezing: the pipeline tunes on val / B1-val, then writes the frozen configs `configs/decoders_frozen.yaml` and `configs/window_length_frozen.yaml`,
which must be **committed before** `scripts/run_final.py` and `scripts/run_e3_e4.py` are run (both refuse to run on uncommitted configs, scripts or manifests, and refuse to overwrite existing results).
`make reproduce*` pass the explicit `--allow_rerun` override because they regenerate the committed numbers from the same frozen configs.
Benchmarks are rebuilt byte-identically (content hashes) from `manifests/b1_manifest.json` and `manifests/b2_manifest.json` (IDs, cut times, seeds only).

Every file in `results/` states the command and commit that produced it in its header. CSV files start with a `#` header line: read them with `pd.read_csv(path, comment="#")`.

## Layout

```
configs/      default.yaml (paths, seeds, window, model params), raga_set.yaml (K = 40), raga_aliases.yaml, scales.yaml (with sources),
              decoders_frozen.yaml, window_length_frozen.yaml
manifests/    B1 and B2 manifests (IDs + cut times + seeds): enough to rebuild the benchmarks from the raw data
src/raga_tracking/   io, pitch, windows, features/{histogram,tdms}, models/{linear,calibration,posterior}, decode/{argmax,majority,novelty,hmm},
                     bench/{synthetic,concerts}, eval/{metrics,bootstrap,latency}
scripts/      numbered entry points 01 ... 12, run_final.py (E2), run_e3_e4.py (E3/E4), check_e2_independent.py, check_release.py
results/      tables (.csv/.md) and figures; docs/ related work, data formats, decisions; paper/ draft; annotations/ B3 status
```

## Release contents

Code, configs, manifests, results, our docs and (when they exist) our own annotations. No Saraga or CMD audio, pitch or tonic files.
The paper draft's author list, venue and the open items in `docs/decisions.md` still need the team's confirmation before a tagged release.
