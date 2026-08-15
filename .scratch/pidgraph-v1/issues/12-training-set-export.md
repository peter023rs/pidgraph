# 12 — Training-set export from reviewer verdicts

**What to build:** An ML developer exports reviewer verdicts as a supervised training set per Convention Profile, so per-company fine-tuning has data drawn from exactly the deployed corpus. Pass verdicts become positive examples, rejects become negatives, and edits carry the human-corrected label.

**Blocked by:** 10 (Review Workbench: overlay and verdicts).

**Status:** ready-for-agent

- [ ] An export produces a training set filtered to one Convention Profile (identity + version), from the stored labeled examples.
- [ ] The export represents all three verdict kinds usefully: pass (confirmed detection), reject (negative), edit (detection with corrected tag text or geometry as the label).
- [ ] Examples reference Sheets by identifier and carry the geometry/text needed for training; exports containing drawing-derived content live outside git per ADR-0001.
- [ ] The export is deterministic over a given verdict store — re-export yields the same set.
- [ ] Tests export from a prepared verdict store fixture and assert content and filtering; offline test invariant holds.

## Comments
