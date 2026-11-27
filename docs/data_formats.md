# Data formats (inspected 2026-11-27, T1)

Everything below was read from real files, not assumed.

## CMD Carnatic features (Zenodo 7278506)
Layout: `RagaDataset/Carnatic/features/{raga_uuid}/{artist}/{concert}/{item}/{item}.{ext}`; 480 items, 40 ragas, 63 artists (normalised).
Metadata: `_info_/path_mbid_ragaid.json` (item path -> MusicBrainz recording id + raga uuid), `_info_/ragaId_to_ragaName_mapping.json` (71 uuid -> name with diacritics).
- `.pitch`: whitespace-separated, 2 columns `time_s  f_hz`, no header. Constant hop 0.0044444 s (~225 Hz). **Unvoiced = 0.0 Hz** (24-38% of frames). Durations 360-900+ s in the 3 samples.
- `.tonic`: single float, Hz (e.g. 195.997718). `.tonicFine`: single float in scientific notation, Hz, slightly different (195.32 vs 195.998). **Decision: use `.tonic`** (matches Saraga ctonic); revisit in T3 if Sa/Pa peaks are off.
- Other files (`.pitchSilIntrpPP`, `.flatSegNyas`, `.taniSegKNN`) are not used. No section/item-order metadata.
- Artist and concert come from the path only (underscores -> spaces).

## Saraga Carnatic 1.5 (Zenodo 4301737)
We fetch only non-audio files (about 275 MB of the 14.4 GB zip; mp3s are skipped) with `scripts/fetch_saraga_meta.py`.
Layout: `{Concert title} by {artist}/{item}/{item}.{ext}`; 249 item JSONs in 26 concerts, 24 lead artists.
- `.json`: keys `mbid,title,length(ms),artists[{artist{mbid,name},instrument,lead}],raaga[{uuid,name,common_name}],taala,form,work,concert[{mbid,title}],album_artists`. **`raaga` is empty for 65 of 249 items**; 184 have exactly one. `common_name` is ASCII lowercase; `name` has diacritics and can list alternates ("Kamavardani/Pantuvarali"). 8 items have raga "Ragamalika". **No track/performance-order field exists** (order comes from MusicBrainz, see decisions.md).
- `.pitch.txt`: same format as CMD `.pitch` (2 columns, time_s, f_hz, hop 0.0044444 s, 0 = unvoiced; some files write floats in scientific notation).
- `.ctonic.txt`: single float, Hz.
- `.sections-manual.txt` / `.sections-manual-p.txt`: tab-separated `start_s  id  duration_s  label` (no header); `-p` has diacritic labels, the plain one slugs. Rāgamālika items label sections by raga (e.g. "Caranam vasanta", "Caranam yamuna kalyani").
- Only 197 of 249 items have pitch + ctonic; 119 have sections; `pitch-vocal.txt` exists for 56 (not used).

## Cross-source findings
- Hop is identical (0.0044444 s) in both sources, still resampled to 100 Hz in T3.
- 37 recordings appear in both CMD and Saraga (same MusicBrainz mbid): `dup_in_other_source` in `inventory.csv`. Count unique mbids when selecting ragas; prefer the Saraga copy (it has sections and concert metadata).
- Artists overlap across sources (e.g. Sanjay Subrahmanyan, Mahati, Rithvik Raja): split on a pooled normalised artist key. Manual review still needed for "Akkarai Sisters" (CMD) vs "Akkarai Subhalakshmi" (Saraga).
