# Decisions log (one dated line per non-obvious choice)
- 2026-11-27: PLAN (1).md copied to PLAN.md so the filename matches CLAUDE.md.
- 2026-11-27: Project dir sits inside the unrelated Desktop git repo; nothing committed. `make` is not installed on this Windows machine; Makefile kept as spec, run the python commands directly.
- 2026-11-27: T0 lit check: partial overlap only (online systems w/o boundary metrics; concert item segmentation). We differ via B1-B3 benchmarks, latency, decoder comparison. Rows tagged [verify] still need manual reading.
- 2026-11-27: Zenodo throttles single connections (~70 KB/s). Used parallel range downloads (scripts/pdl.py) for CMD; for Saraga fetched only non-audio files via HTTP range reads of the remote zip (scripts/fetch_saraga_meta.py) - no audio downloaded, saves ~14 GB.
- 2026-11-27: Tonic source = CMD `.tonic` and Saraga `.ctonic.txt` (not `.tonicFine`).
- 2026-11-27: Saraga has no item/track order in its metadata (T1 backup 3: derive from file structure - impossible here). B2 item order needs an external source (e.g. MusicBrainz release track positions via the concert mbid) - OPEN, needs team decision. Otherwise B2 is dropped / replaced per T9 backup.
- 2026-11-27: Saraga is small (197 items with pitch, 26 concerts) - risk for G1 (K>=15), B1/B2 sizes (>=8 test concerts) - check in T2.
- 2026-11-27: 37 CMD items duplicate Saraga items (same mbid); dedupe by mbid when counting. Tonic >= 95%: 100% of pitch-bearing recordings; 92.7% of all 729 only because 52 Saraga items ship without pitch/tonic.
