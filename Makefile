PY ?= python
W ?= w1
.PHONY: test inventory splits cache train bench tune final reproduce
test:       ; $(PY) -m pytest -q
inventory:  ; $(PY) scripts/01_inventory.py
splits:     ; $(PY) scripts/02_splits.py
cache:      ; $(PY) scripts/03_cache_pitch.py
train:      ; $(PY) scripts/04_train.py --model $(W)
bench:      ; $(PY) scripts/08_build_b1.py && $(PY) scripts/09_build_b2.py
tune:       ; $(PY) scripts/10_tune_decoders.py
final:      ; $(PY) scripts/run_final.py
reproduce:  inventory splits cache train bench tune final
