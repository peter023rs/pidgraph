# 08 — Batch runs with resume and honest failure/gap reporting

**What to build:** An operator processes a 200–500 Sheet Document as a single batch run with per-Sheet progress and resumable state — one bad Sheet doesn't cost the night's run. Failed Sheets and low-confidence extractions are reported as failures and gaps in the run report, never silently dropped or papered over, so the output's coverage is honest.

**Blocked by:** 02 (PDF Document intake).

**Status:** resolved

- [x] A multi-Sheet Document runs as a batch with per-Sheet progress visible to the operator during the run.
- [x] A Sheet that fails is recorded as failed and the batch continues; the run completes with the remaining Sheets' artifacts intact.
- [x] Interrupting a run and starting it again resumes from persisted state: completed Sheets are not re-processed, and the resumed run converges to the same artifacts as an uninterrupted one.
- [x] The run report lists every failed Sheet with its reason, and surfaces low-confidence extractions as gaps — coverage is stated, not implied.
- [x] Nothing is silently dropped: every Sheet of the Document is accounted for in the report as succeeded, failed, or skipped-by-resume.
- [x] Tests drive a small multi-Sheet fixture Document through failure injection and interrupt/resume; offline test invariant holds.

## Comments

2026-08-18 (agent): Implemented as `pidgraph.batch.run_batch(document,
profile, run_dir, ...)` — digitize()'s operator-scale sibling. Design
points worth keeping:

- Resume state is one atomically written (write-then-rename) JSON file
  per Sheet under `<run_dir>/state/`, holding the Sheet's detection
  record on success or the failure reason. Completed Sheets are skipped
  ("skipped-by-resume"); failed Sheets are retried. Final assembly
  always rebuilds from the persisted record form — fresh and resumed
  Sheets share one path, so convergence is structural, and the resume
  test byte-compares the plant model, Cypher, and detection records
  against an uninterrupted run's.
- `run_report.json` is deliberately outside the byte-convergence
  guarantee: a resumed run's report says "skipped-by-resume" where the
  uninterrupted one says "succeeded", because the report records honest
  history (acceptance line 5); gaps and coverage totals converge.
- A `state/manifest.json` pins the run to its (Document, Convention
  Profile, PipelineConfig) triple — including a content fingerprint of
  the profile parts, so a profile edited under an unchanged version is
  refused rather than silently mixed into resumed state (two-axis
  review finding). Unreadable state files are refused by name, never
  treated as absent.
- Gaps: fail-closed lexicon reads surface as `unresolved-text`;
  anything below `low_confidence_threshold` (default 0.7 — above the
  0.5 unresolved discount, below the 0.9 corrected-read discount)
  surfaces as `low-confidence`. Interrupts (KeyboardInterrupt) stop the
  run; only `Exception` counts as a per-Sheet failure.
- Note for the spec: "One top seam: digitize()" now has a sanctioned
  sibling seam at `run_batch()` (this ticket / user story 3); the spec
  decision text predates it and wasn't edited here.
