# B3: ragamalika case study (not done)

**Status: not annotated, not evaluated.** The plan (T13) needs at least two independent human annotators who mark rāga changes in
rāgamālika recordings **without seeing any model output**, an inter-annotator agreement of boundary F@3 s, and at least 5 usable recordings.
No annotators were available, and annotating from the pitch tracks alone (no audio is downloaded; only pitch, tonic and metadata) by the
same toolchain that built and evaluated the models would not be independent ground truth. Following the T13 backup ("fewer than 5 usable recordings:
drop B3 and add it to the limitations"), B3 is dropped. This is logged in `docs/decisions.md`.

No file in `annotations/ragamalika/` exists, and no result in `results/` refers to B3.

## What a future B3 needs (unchanged from the plan)

- Candidates: the ragamalika recordings listed in `results/inventory_summary.md` (two ragamalika items also appear in B2 as `other`).
- Each annotator independently writes `annotations/ragamalika/{rec_id}.{annotator}.csv` with `start_s,end_s,raga,confidence,notes`, working from a Sonic Visualiser
  session with the audio (only if licensed locally) or the pitch track, plus the published section-wise rāga order (cite the source for every recording, e.g. karnatik.com).
- **No model predictions may be shown to annotators.** State this in this README when annotation happens.
- Compute inter-annotator boundary F@3 s per recording and overall; settle disagreements into `{rec_id}.csv`; flag disagreements above 10 s.
  If overall F@3 s < 0.7, evaluate at +-10 s only and say so. Evaluate the frozen best configuration once and write `results/e5_ragamalika.md`.
