# 11 — Workbench confidence queues and per-Sheet review progress

**What to build:** A reviewer's minutes go to the elements most likely to be wrong: detections are presented in confidence-sorted queues. Across a Document, the reviewer sees which Sheets are fully reviewed, which are partially done, and which are untouched — so a 400-Sheet review can be split across days.

**Blocked by:** 10 (Review Workbench: overlay and verdicts).

**Status:** resolved

- [x] Detections are presented in queues sorted by confidence, lowest first; giving a verdict advances the queue.
- [x] Queues can be scoped (at minimum per Sheet; per detection kind if the artifacts distinguish them).
- [x] A Document-level view shows each Sheet's review state — fully reviewed, in progress, untouched — derived from verdict coverage over that Sheet's detections.
- [x] Review state is accurate after restarts and across multi-day sessions (recomputed from persisted verdicts, not session memory).
- [x] Tests use the Flask test client against prepared run artifacts, asserting queues sort by confidence and progress states reflect verdict coverage; offline test invariant holds.

## Comments

Implemented in `src/pidgraph/workbench.py`, on top of ticket 10's app.
`GET /sheet/<n>/queue` returns the Sheet's undecided detections lowest
confidence first (ties keep record order — the sort is stable, so the
queue is deterministic across requests and restarts); `?kind=symbol|
line|text` scopes it per detection kind, which the artifacts distinguish.
A verdict advances the queue by construction: the queue is derived from
the persisted labels, so a decided detection leaves it. The Sheet page
drives review from the queue — status and lowest-confidence selection on
load, a scope select, a Next button, and auto-advance after each saved
verdict.

`GET /progress` (and the index page) is the Document-level view: per
Sheet, `untouched` / `in progress` / `reviewed` with decided/total
counts, recomputed from the persisted verdicts on every request — no
session memory, so multi-day and post-restart state is exact. Coverage
counts only verdicts on detections the Sheet's record actually contains
(stale labels from an older run never inflate it); a Sheet with nothing
detected counts as reviewed; a truncated/corrupt record degrades to an
`unreadable` row instead of breaking the whole Document's view. A
Sheet's identity is its artifact filename (the URL number) everywhere —
labels, queue, progress — so a renamed record cannot split the keying.
Because detection records never change during review, the progress view
caches each record's profile + detection ids against the file's stat;
labels are deliberately never cached.

Tests: `tests/test_workbench_review_flow.py` (Flask test client over
prepared artifacts of a four-Sheet run — three drawn Sheets plus a blank
one — with confidences scrambled in the record to pin sort order),
offline throughout. Ticket 10's invariants (only writable route is the
verdict endpoint; no path into the extraction engine) still hold and
stay asserted in `tests/test_workbench.py`.
