# 19 — Product KPI computed from Workbench activity

**What to build:** The operator sees the product KPI — corrections per 100 symbols and reviewer minutes per accepted Sheet — computed from Review Workbench activity, so "the software works" is measurable on real corpus Sheets. Together with the component gates, this is the definition of "works" that triggers hazop-ai reintegration.

**Blocked by:** 11 (Workbench queues and review progress).

**Status:** resolved

- [x] Corrections per 100 symbols is computed from verdict records (reject + edit counted as corrections against symbols reviewed), reportable per Document and per Convention Profile.
- [x] Reviewer minutes per accepted Sheet is computed from Workbench activity timing and Sheet review completion, with the measurement basis stated alongside the number.
- [x] The KPI is visible to the operator (in the Workbench or its report output) and recomputable from persisted verdicts and activity — not session memory.
- [x] KPI output references Sheets and Documents by identifier only; no drawing content leaves the local store per ADR-0001.
- [x] Tests compute KPIs from prepared verdict/activity fixtures with known expected values, under the offline invariant.

## Comments

Implemented as `src/pidgraph/kpi.py` (Flask-free) plus the Workbench's
activity logging and KPI views in `src/pidgraph/workbench.py`.

**Activity.** The Workbench now logs its own activity — every Sheet
opened (`GET /sheet/<n>`) and every verdict saved — as one JSON line per
event at the server's clock, appended to `activity.jsonl` beside the
verdict store (`<run_dir>/labels/` by default, or wherever
`--labels-dir` points), outside git with the rest of the review data.
The verdict store keeps only the latest verdict per detection; the log
keeps every action with its time, and is what reviewer minutes are
measured from. Events carry identifiers only (`at`, `event`, `sheet`,
and for verdicts `detection_id` + `verdict`). Refused verdicts, 404s and
reads (queue, progress, raster, index, the KPI itself) log nothing. The
only writable route is still the verdict endpoint; `create_app` takes a
`clock` so the tests drive time.

**Corrections per 100 symbols** = 100 × (reject + edit verdicts on
symbol detections) / symbol detections with a verdict — the latest
verdict per detection in the persisted store, counting only detections
the run's artifacts contain (a stale label from another run's artifacts
counts nowhere, as in progress). Lines and texts are reported beside it
per kind (`by_kind`), never mixed into the headline.

**Reviewer minutes per accepted Sheet.** An accepted Sheet has at least
one detection and a verdict on every one — the Workbench's "reviewed"
state (`artifacts.review_state`, the one rule progress and the KPI both
derive from) on a Sheet with something to review; a blank Sheet was
never reviewed, so it is never accepted. The activity timeline splits at
every event; each interval of at most the idle threshold (default 10
minutes, `--idle-minutes` on both the Workbench and the CLI) belongs to
the Sheet of the event that opened it — from an action on a Sheet until
the next action anywhere, the reviewer is on that Sheet, so interleaved
Sheets never double-count; a longer interval is a break and counts
nothing; the tail after the last event is unmeasured. The ratio is over
the accepted Sheets the log *timed* — at least one of the Sheet's
verdicts was saved through the Workbench; opening a Sheet to look is not
reviewing it, so an accepted Sheet whose verdicts came from another tool
(or whose log was lost) is counted as `accepted_untimed`, never zeroed
into the average. Activity on a Sheet number the run has no record for
is reported as `minutes_outside_artifacts`; an unreadable record is one
unreadable row, not a broken Document; a stored example whose kind or
verdict the store's contract does not allow (tampered with outside the
Workbench) is refused naming the Sheet and detection, as ticket 12's
export does. Every report carries `basis` — the rules above in words,
with the threshold — and a number that is not yet measurable is `null`
/ "not yet measurable", never a zero.

**Visibility.** `GET /kpi` serves the JSON report for the run; the
Workbench index shows both headline numbers with their counts and basis
text, recomputed from the persisted verdicts and activity on every
request (a restarted Workbench reports the same KPI). The CLI `python -m
pidgraph.kpi <run_dir> [<run_dir> ...] [--idle-minutes M] [--out PATH]
[--labels-dir D]` prints the summary and optionally writes the full
report; given several runs it reports per Document (the batch manifest's
or plant model's Document name, else the run directory name) and per
Convention Profile aggregated across the Documents reviewed under it,
plus overall. `kpi_report` refuses to combine documents computed under
different thresholds, so one report states one basis.

**Identifiers only.** Reports hold Sheet numbers, Document names,
Convention Profile identity + version, counts and minutes — no detection
snapshots, geometry, tag strings or corrections; the test walks every
key of the report to prove it.

**Refactor on the way.** The Workbench's inline run-artifact readers
(sheet numbers, the stat-keyed record-summary cache — now a
`RecordSummary(profile, ids)` — `iter_detections`, and the review-state
rule) moved to `src/pidgraph/artifacts.py` so the KPI CLI and the
Workbench read the artifacts one way (and the index computes the KPI
from the same cache as progress). Behaviour is unchanged; ticket 10/11
tests still pass as written.

Tests: `tests/test_kpi.py` (prepared verdict stores and activity logs
with known expected values — 20.0 corrections per 100 symbols, 8.0
minutes per accepted Sheet — idle-threshold edges, interleaved-Sheet
attribution, an opened-but-not-reviewed-here accepted Sheet,
per-Document/per-Profile aggregation across three runs, identifier-only
output, malformed-store refusal, CLI) and `tests/test_workbench_kpi.py`
(Flask test client with `conftest.FakeClock`: what is logged and what is
not, `/kpi`, the index block, restart, `--labels-dir`, threshold).
Offline throughout; full suite 371 passed, mypy clean. Reviewed with
`/code-review` (standards + spec axes); the timing denominator was
tightened from "any logged event" to "a logged verdict" on its finding.
