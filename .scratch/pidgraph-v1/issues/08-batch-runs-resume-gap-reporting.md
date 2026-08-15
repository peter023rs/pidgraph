# 08 — Batch runs with resume and honest failure/gap reporting

**What to build:** An operator processes a 200–500 Sheet Document as a single batch run with per-Sheet progress and resumable state — one bad Sheet doesn't cost the night's run. Failed Sheets and low-confidence extractions are reported as failures and gaps in the run report, never silently dropped or papered over, so the output's coverage is honest.

**Blocked by:** 02 (PDF Document intake).

**Status:** ready-for-agent

- [ ] A multi-Sheet Document runs as a batch with per-Sheet progress visible to the operator during the run.
- [ ] A Sheet that fails is recorded as failed and the batch continues; the run completes with the remaining Sheets' artifacts intact.
- [ ] Interrupting a run and starting it again resumes from persisted state: completed Sheets are not re-processed, and the resumed run converges to the same artifacts as an uninterrupted one.
- [ ] The run report lists every failed Sheet with its reason, and surfaces low-confidence extractions as gaps — coverage is stated, not implied.
- [ ] Nothing is silently dropped: every Sheet of the Document is accounted for in the report as succeeded, failed, or skipped-by-resume.
- [ ] Tests drive a small multi-Sheet fixture Document through failure injection and interrupt/resume; offline test invariant holds.

## Comments
