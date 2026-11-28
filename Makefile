PY ?= python
W ?= w1
# `final`, `windows-test` re-run only on a frozen, committed config and refuse to overwrite committed results (CLAUDE.md rule 2).
# `reproduce` / `reproduce-eval` regenerate the committed numbers from the same frozen configs, so they pass the explicit override.
RERUN ?= --allow_rerun "make reproduce: regenerating committed results from the same frozen config"
L_EXTRA = 10 20 60
.PHONY: app test inventory splits cache train train-w2 train-w2-L calibrate bench tune final windows-tune windows-test errors release-check reproduce reproduce-eval
test:         ; $(PY) -m pytest -q
app:          ; $(PY) app/serve.py --open
inventory:    ; $(PY) scripts/01_inventory.py
splits:       ; $(PY) scripts/02_splits.py
cache:        ; $(PY) scripts/03_cache_pitch.py
train:        ; $(PY) scripts/04_train.py --model $(W)
train-w2:     ; $(PY) scripts/05_train_w2.py
train-w2-L:   ; for L in $(L_EXTRA); do $(PY) scripts/05_train_w2.py --L $$L --tag _L$$L --fix 0.3,24; done
calibrate:    ; $(PY) scripts/07_calibrate.py
bench:        ; $(PY) scripts/08_build_b1.py --hard_min 6 && $(PY) scripts/09_build_b2.py
tune:         ; $(PY) scripts/10_tune_decoders.py
final:        ; $(PY) scripts/run_final.py $(RERUN)
windows-tune: ; $(PY) scripts/11_tune_window_length.py --L 10 --L 20 --L 30 --L 60 --freeze
windows-test: ; $(PY) scripts/run_e3_e4.py $(RERUN)
errors:       ; $(PY) scripts/12_error_analysis.py
release-check: ; $(PY) scripts/check_release.py
# Everything from cached pitch (data/interim) and trained models (models/): benchmarks, frozen test runs, error analysis.
reproduce-eval: final windows-test errors release-check
# Everything from raw data (hours: W2 training over the C grid dominates). Commit configs/decoders_frozen.yaml and
# configs/window_length_frozen.yaml between the tuning steps and the test runs, as done in the original runs.
reproduce:    inventory splits cache train train-w2 train-w2-L calibrate bench tune windows-tune reproduce-eval
