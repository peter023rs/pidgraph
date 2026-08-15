# 11 — Workbench confidence queues and per-Sheet review progress

**What to build:** A reviewer's minutes go to the elements most likely to be wrong: detections are presented in confidence-sorted queues. Across a Document, the reviewer sees which Sheets are fully reviewed, which are partially done, and which are untouched — so a 400-Sheet review can be split across days.

**Blocked by:** 10 (Review Workbench: overlay and verdicts).

**Status:** ready-for-agent

- [ ] Detections are presented in queues sorted by confidence, lowest first; giving a verdict advances the queue.
- [ ] Queues can be scoped (at minimum per Sheet; per detection kind if the artifacts distinguish them).
- [ ] A Document-level view shows each Sheet's review state — fully reviewed, in progress, untouched — derived from verdict coverage over that Sheet's detections.
- [ ] Review state is accurate after restarts and across multi-day sessions (recomputed from persisted verdicts, not session memory).
- [ ] Tests use the Flask test client against prepared run artifacts, asserting queues sort by confidence and progress states reflect verdict coverage; offline test invariant holds.

## Comments
