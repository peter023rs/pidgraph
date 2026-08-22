"""Product KPI seam tests (ticket 19): corrections per 100 symbols and
reviewer minutes per accepted Sheet, computed from persisted verdicts and
Workbench activity — never session memory — with known expected values
from prepared fixtures. Offline throughout: run artifacts come from the
stub pipeline on synthetic Sheets; verdicts are written through the
LabelStore and activity through the ActivityLog with a fake clock."""

import json
import re
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from conftest import FakeClock, T0

from pidgraph.kpi import (
    ActivityLog,
    Event,
    attribute_minutes,
    kpi_report,
    main,
    run_kpi,
)
from pidgraph.labels import LabelStore, make_example
from pidgraph.model import Document, SheetAnnotations
from pidgraph.pipeline import digitize

_KIND_FIELDS = (("symbol", "symbols"), ("line", "lines"), ("text", "texts"))


def _record(run_dir: Path, sheet: int) -> dict:
    return json.loads((run_dir / "detections" / f"sheet_{sheet}.json")
                      .read_text(encoding="utf-8"))


def _detections(record: dict, kind: str) -> list[dict]:
    return record[dict(_KIND_FIELDS)[kind]]


def _correction(kind: str) -> dict:
    return {"symbol": {"bbox": [0.0, 0.0, 10.0, 10.0]},
            "line": {"polyline": [[0.0, 0.0], [10.0, 10.0]]},
            "text": {"string": "X-1"}}[kind]


def _digitize(out: Path, profile, name: str = "kpi-fixture.pdf") -> Path:
    """A four-Sheet run: three drawn Sheets (16 detections each — 7
    symbols, 3 lines, 6 texts) and one blank Sheet."""
    from conftest import build_sheet, build_synthetic_sheet

    document = Document(
        name=name,
        sheets=(build_synthetic_sheet(1), build_synthetic_sheet(2),
                build_synthetic_sheet(3),
                build_sheet(4, SheetAnnotations())))
    digitize(document, profile, out_dir=out)
    return out


def _review(store: LabelStore, log: ActivityLog | None, clock: FakeClock,
            record: dict, sheet: int, plan: dict[str, list[str]],
            seconds_apart: float) -> None:
    """Give verdicts per the plan (kind -> verdict per detection, in
    record order), logging each as Workbench activity when a log is given,
    the clock advancing seconds_apart between verdicts."""
    for kind, verdicts in plan.items():
        for detection, verdict in zip(_detections(record, kind), verdicts):
            clock.tick(seconds_apart)
            example = make_example(
                kind, detection, verdict,
                _correction(kind) if verdict == "edit" else None)
            store.record(record["profile"], sheet, example)
            if log is not None:
                log.record("verdict", sheet, detection_id=detection["id"],
                           verdict=verdict)


@pytest.fixture
def run_dir(tmp_path, synthetic_profile) -> Path:
    return _digitize(tmp_path / "run", synthetic_profile)


@pytest.fixture
def reviewed_run(run_dir) -> Path:
    """The fixture review with known expected values.

    Sheet 3 — fully reviewed in the Workbench: opened, then 16 verdicts
      30 s apart (symbols 5 pass / 1 reject / 1 edit, lines 3 pass, texts
      4 pass / 2 edit) -> 8.0 reviewer minutes, accepted.
    Sheet 1 — a 60-minute break later: opened, then 6 symbol verdicts
      60 s apart (4 pass / 2 reject) -> 6.0 minutes, in progress. A stale
      verdict on a detection the record does not contain must not count.
    Sheet 2 — opened in the Workbench after another break, then fully
      reviewed (all pass) straight through the store: accepted, but the
      log never timed its review.
    Sheet 4 — blank: nothing to review, never an accepted Sheet.
    """
    store = LabelStore(run_dir / "labels")
    clock = FakeClock()
    log = ActivityLog(run_dir / "labels", clock=clock)

    log.record("open", 3)
    _review(store, log, clock, _record(run_dir, 3), 3, {
        "symbol": ["pass", "pass", "pass", "pass", "pass", "reject", "edit"],
        "line": ["pass", "pass", "pass"],
        "text": ["pass", "pass", "pass", "pass", "edit", "edit"],
    }, seconds_apart=30)

    clock.tick(60 * 60)
    log.record("open", 1)
    record_1 = _record(run_dir, 1)
    _review(store, log, clock, record_1, 1, {
        "symbol": ["pass", "pass", "pass", "pass", "reject", "reject"],
    }, seconds_apart=60)
    stale = {"id": "p9-sym99", "sheet": 1, "symbol_class": "tank",
             "bbox": [0.0, 0.0, 1.0, 1.0], "confidence": 1.0}
    store.record(record_1["profile"], 1, make_example("symbol", stale,
                                                      "reject"))

    # Sheet 2: opened in the Workbench after another break, but every
    # verdict written straight through the store — the log never timed
    # its review (opening a Sheet to look is not reviewing it)
    clock.tick(60 * 60)
    log.record("open", 2)
    _review(store, None, clock, _record(run_dir, 2), 2, {
        "symbol": ["pass"] * 7, "line": ["pass"] * 3, "text": ["pass"] * 6,
    }, seconds_apart=0)
    return run_dir


def _counts(reviewed, pass_, reject, edit) -> dict:
    return {"reviewed": reviewed, "pass": pass_, "reject": reject,
            "edit": edit, "corrections": reject + edit}


def test_corrections_per_100_symbols_from_verdict_records(reviewed_run):
    document = run_kpi(reviewed_run)

    # reject + edit on symbol detections, against symbols reviewed: Sheet 1
    # 6 reviewed / 2 corrected, Sheet 2 7 / 0, Sheet 3 7 / 2 -> 4 of 20
    assert document["by_kind"]["symbol"] == _counts(20, 16, 3, 1)
    assert document["corrections_per_100_symbols"] == pytest.approx(20.0)
    # the other kinds are reported beside the headline, never mixed in
    assert document["by_kind"]["line"] == _counts(6, 6, 0, 0)
    assert document["by_kind"]["text"] == _counts(12, 10, 0, 2)

    rows = {row["sheet"]: row for row in document["sheets"]}
    assert rows[1]["by_kind"]["symbol"] == _counts(6, 4, 2, 0)
    assert rows[3]["by_kind"]["symbol"] == _counts(7, 5, 1, 1)
    assert rows[3]["by_kind"]["text"] == _counts(6, 4, 0, 2)
    # the stale verdict on a detection Sheet 1's record lacks counts
    # nowhere: decided stays at the six real verdicts
    assert rows[1]["decided"] == 6
    assert rows[1]["detections"] == 16


def test_reviewer_minutes_per_accepted_sheet_from_activity_and_completion(
        reviewed_run):
    document = run_kpi(reviewed_run)
    rows = {row["sheet"]: row for row in document["sheets"]}

    # Sheet 3: open + 16 verdicts 30 s apart = 16 intervals of 0.5 min
    assert rows[3] == {
        "sheet": 3, "profile": {"name": "synthetic-test", "version": "0"},
        "detections": 16, "decided": 16, "accepted": True,
        "timed": True, "minutes": pytest.approx(8.0),
        "by_kind": rows[3]["by_kind"]}
    # Sheet 1: the hour-long break before it is not reviewer time; then
    # open + 6 verdicts a minute apart; not accepted (6 of 16 decided)
    assert rows[1]["minutes"] == pytest.approx(6.0)
    assert rows[1]["accepted"] is False
    assert rows[1]["timed"] is True
    # Sheet 2: accepted and opened in the Workbench, but none of its
    # verdicts went through it — the log did not time its review
    assert rows[2]["accepted"] is True
    assert rows[2]["timed"] is False
    assert rows[2]["minutes"] == 0
    # Sheet 4: nothing detected, so nothing was reviewed — not accepted
    assert rows[4]["detections"] == 0
    assert rows[4]["accepted"] is False

    assert document["sheets_summary"] == {
        "total": 4, "accepted": 2, "accepted_timed": 1,
        "accepted_untimed": 1, "unreadable": 0}
    assert document["minutes"] == {
        "total": pytest.approx(14.0),
        "on_timed_accepted_sheets": pytest.approx(8.0)}
    # minutes per accepted Sheet is over the accepted Sheets the activity
    # log timed — the untimed accepted Sheet is counted, not zeroed in
    assert document["reviewer_minutes_per_accepted_sheet"] == \
        pytest.approx(8.0)
    assert document["activity_log"] == {
        "events": 25, "unreadable_lines": 0, "minutes_outside_artifacts": 0}


def test_the_measurement_basis_is_stated_alongside_the_numbers(
        reviewed_run):
    report = kpi_report([run_kpi(reviewed_run)])
    basis = report["basis"]
    assert basis["idle_threshold_minutes"] == 10.0
    assert "reject" in basis["corrections_per_100_symbols"]
    assert "edit" in basis["corrections_per_100_symbols"]
    assert "symbol" in basis["corrections_per_100_symbols"]
    minutes_basis = basis["reviewer_minutes_per_accepted_sheet"]
    assert "accepted Sheet" in minutes_basis
    assert "every detection" in minutes_basis
    assert "10" in minutes_basis          # the idle threshold, in words
    assert "break" in minutes_basis
    assert "unmeasured" in minutes_basis

    # a different threshold is stated as such, and changes the numbers:
    # at a 30 s threshold Sheet 1's one-minute intervals are breaks
    tight = run_kpi(reviewed_run, idle_minutes=0.5)
    assert kpi_report([tight])["basis"]["idle_threshold_minutes"] == 0.5
    # one report states one basis: documents computed under different
    # thresholds cannot share it
    with pytest.raises(ValueError):
        kpi_report([tight, run_kpi(reviewed_run)])
    rows = {row["sheet"]: row for row in tight["sheets"]}
    assert rows[3]["minutes"] == pytest.approx(8.0)
    assert rows[1]["minutes"] == 0


def test_intervals_belong_to_the_sheet_of_the_event_that_opened_them():
    def at(seconds: float) -> datetime:
        return datetime.fromisoformat(T0) + timedelta(seconds=seconds)

    events = [
        Event(at(0), "open", 1),
        Event(at(60), "verdict", 1),       # 1 min on Sheet 1
        Event(at(120), "open", 2),         # 1 min more on Sheet 1 (the
                                           # reviewer was on 1 until then)
        Event(at(720), "verdict", 2),      # exactly the threshold: counted
        Event(at(1321), "verdict", 1),     # 601 s > threshold: a break
        Event(at(1351), "verdict", 1),     # 0.5 min on Sheet 1
                                           # the tail after the last event
                                           # is unmeasured
    ]
    assert attribute_minutes(events, idle_minutes=10.0) == {
        1: pytest.approx(2.5), 2: pytest.approx(10.0)}
    # order of recording is irrelevant: the timeline is sorted by time
    assert attribute_minutes(list(reversed(events)), idle_minutes=10.0) \
        == attribute_minutes(events, idle_minutes=10.0)
    assert attribute_minutes([], idle_minutes=10.0) == {}


def test_activity_persists_as_lines_beside_the_labels_and_rereads(
        tmp_path):
    clock = FakeClock()
    log = ActivityLog(tmp_path / "labels", clock=clock)
    opened = log.record("open", 7)
    clock.tick(12)
    given = log.record("verdict", 7, detection_id="p7-sym1", verdict="edit")

    assert opened == {"at": T0, "event": "open", "sheet": 7}
    assert given == {"at": "2026-08-21T09:00:12+00:00", "event": "verdict",
                     "sheet": 7, "detection_id": "p7-sym1",
                     "verdict": "edit"}
    path = tmp_path / "labels" / "activity.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [opened, given]

    # a fresh log over the same directory reads what was persisted
    events, unreadable = ActivityLog(tmp_path / "labels").events()
    assert unreadable == 0
    assert [(e.event, e.sheet, e.at.isoformat()) for e in events] == [
        ("open", 7, T0), ("verdict", 7, "2026-08-21T09:00:12+00:00")]

    # a line a killed process left half-written is counted, not fatal,
    # and never hides the lines around it
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"at": "2026-08-21T09:01:00+00:00", "ev')
    events, unreadable = ActivityLog(tmp_path / "labels").events()
    assert (len(events), unreadable) == (2, 1)
    # no log at all is simply no activity
    assert ActivityLog(tmp_path / "nowhere").events() == ([], 0)


def test_kpi_recomputes_from_persisted_verdicts_and_activity(reviewed_run):
    # nothing is held in memory between computations: two independent
    # computations over the same store agree, and a verdict reconsidered
    # through the store alone moves the next computation
    first = run_kpi(reviewed_run)
    assert run_kpi(reviewed_run) == first

    record = _record(reviewed_run, 2)
    symbol = record["symbols"][0]
    LabelStore(reviewed_run / "labels").record(
        record["profile"], 2, make_example("symbol", symbol, "reject"))
    second = run_kpi(reviewed_run)
    assert second["by_kind"]["symbol"]["corrections"] == \
        first["by_kind"]["symbol"]["corrections"] + 1
    assert second["corrections_per_100_symbols"] == pytest.approx(25.0)


def test_kpi_is_reportable_per_document_and_per_convention_profile(
        tmp_path, synthetic_profile):
    other_version = replace(synthetic_profile, version="1")
    runs = {
        "a": _digitize(tmp_path / "a", synthetic_profile, "doc-a.pdf"),
        "b": _digitize(tmp_path / "b", synthetic_profile, "doc-b.pdf"),
        "c": _digitize(tmp_path / "c", other_version, "doc-c.pdf"),
    }
    # each Document: Sheet 1 fully reviewed with Workbench activity (open,
    # then 16 verdicts n seconds apart) and a known symbol-correction count
    plans = {
        "a": (["pass"] * 6 + ["reject"], 15),   # 1 of 7, 4.0 minutes
        "b": (["pass"] * 5 + ["edit"] * 2, 30),  # 2 of 7, 8.0 minutes
        "c": (["reject"] * 7, 60),               # 7 of 7, 16.0 minutes
    }
    for key, run in runs.items():
        symbols, apart = plans[key]
        clock = FakeClock()
        log = ActivityLog(run / "labels", clock=clock)
        log.record("open", 1)
        _review(LabelStore(run / "labels"), log, clock, _record(run, 1), 1,
                {"symbol": symbols, "line": ["pass"] * 3,
                 "text": ["pass"] * 6}, seconds_apart=apart)

    report = kpi_report([run_kpi(runs[k]) for k in ("a", "b", "c")])

    by_document = {d["document"]: d for d in report["documents"]}
    assert list(by_document) == ["doc-a.pdf", "doc-b.pdf", "doc-c.pdf"]
    assert by_document["doc-a.pdf"]["corrections_per_100_symbols"] == \
        pytest.approx(100 / 7)
    assert by_document["doc-a.pdf"]["reviewer_minutes_per_accepted_sheet"] \
        == pytest.approx(4.0)
    assert by_document["doc-c.pdf"]["corrections_per_100_symbols"] == \
        pytest.approx(100.0)
    assert by_document["doc-c.pdf"]["reviewer_minutes_per_accepted_sheet"] \
        == pytest.approx(16.0)
    assert by_document["doc-a.pdf"]["profiles"] == [
        {"name": "synthetic-test", "version": "0"}]

    # per Convention Profile: identity + version, aggregated across the
    # Documents reviewed under it
    by_profile = {(p["profile"]["name"], p["profile"]["version"]): p
                  for p in report["profiles"]}
    assert set(by_profile) == {("synthetic-test", "0"),
                               ("synthetic-test", "1")}
    v0 = by_profile[("synthetic-test", "0")]
    assert v0["documents"] == ["doc-a.pdf", "doc-b.pdf"]
    assert v0["by_kind"]["symbol"] == _counts(14, 11, 1, 2)
    assert v0["corrections_per_100_symbols"] == pytest.approx(300 / 14)
    assert v0["sheets_summary"]["accepted"] == 2
    assert v0["reviewer_minutes_per_accepted_sheet"] == pytest.approx(6.0)
    v1 = by_profile[("synthetic-test", "1")]
    assert v1["documents"] == ["doc-c.pdf"]
    assert v1["corrections_per_100_symbols"] == pytest.approx(100.0)

    # overall, across every Document given
    overall = report["overall"]
    assert overall["by_kind"]["symbol"] == _counts(21, 11, 8, 2)
    assert overall["corrections_per_100_symbols"] == pytest.approx(1000 / 21)
    assert overall["reviewer_minutes_per_accepted_sheet"] == \
        pytest.approx(28 / 3)
    assert overall["sheets_summary"]["total"] == 12


def test_an_unreviewed_run_has_no_kpi_yet_not_a_zero(run_dir):
    document = run_kpi(run_dir)
    assert document["corrections_per_100_symbols"] is None
    assert document["reviewer_minutes_per_accepted_sheet"] is None
    assert document["by_kind"]["symbol"] == _counts(0, 0, 0, 0)
    assert document["sheets_summary"]["accepted"] == 0
    assert document["activity_log"]["events"] == 0
    report = kpi_report([document])
    assert report["overall"]["corrections_per_100_symbols"] is None
    assert report["profiles"][0]["reviewer_minutes_per_accepted_sheet"] \
        is None


def test_a_corrupt_record_marks_one_sheet_unreadable_not_the_document(
        reviewed_run):
    (reviewed_run / "detections" / "sheet_2.json").write_text(
        "{truncated", encoding="utf-8")
    document = run_kpi(reviewed_run)
    rows = {row["sheet"]: row for row in document["sheets"]}
    assert rows[2]["detections"] is None
    assert rows[2]["accepted"] is False
    assert rows[2]["profile"] is None
    assert document["sheets_summary"]["unreadable"] == 1
    # the rest of the Document still reports
    assert rows[3]["accepted"] is True
    assert document["by_kind"]["symbol"] == _counts(13, 9, 3, 1)


def test_a_malformed_stored_verdict_is_refused_by_name(run_dir):
    # the store's write contract validates verdicts and corrections, not
    # kinds; an example tampered with outside the Workbench is refused
    # naming the Sheet and detection, never counted under a guess
    record = _record(run_dir, 1)
    symbol = record["symbols"][0]
    store = LabelStore(run_dir / "labels")
    for kind, verdict in (("blob", "pass"), ("symbol", "approve")):
        store.record(record["profile"], 1, {
            "verdict": verdict, "kind": kind, "detection": symbol,
            "correction": None})
        with pytest.raises(ValueError,
                           match=rf"Sheet 1 .*{re.escape(symbol['id'])}"):
            run_kpi(run_dir)


def test_activity_on_sheets_outside_the_artifacts_is_accounted_for(
        run_dir):
    clock = FakeClock()
    log = ActivityLog(run_dir / "labels", clock=clock)
    log.record("open", 42)        # no such Sheet in this run's artifacts
    clock.tick(90)
    log.record("open", 1)
    document = run_kpi(run_dir)
    assert document["activity_log"]["minutes_outside_artifacts"] == \
        pytest.approx(1.5)
    assert document["minutes"]["total"] == 0
    assert {row["sheet"] for row in document["sheets"]} == {1, 2, 3, 4}


def test_kpi_output_references_sheets_and_documents_by_identifier_only(
        reviewed_run):
    """No drawing content leaves the local store (ADR-0001): the report
    names Sheets, Documents and Convention Profiles, and carries counts
    and minutes — never detection snapshots, geometry or tag strings."""
    report = kpi_report([run_kpi(reviewed_run)])
    serialized = json.dumps(report, ensure_ascii=False)

    assert "kpi-fixture.pdf" in serialized
    assert '"synthetic-test"' in serialized
    for tag in ("T-101", "V-101", "T-102", "PI-100", "150-GA-001",
                "DW02-0003", "X-1"):
        assert tag not in serialized
    # no corrected or recorded geometry either: a number like 180 (the
    # valve's x) may legitimately appear as a count, so check structure
    # rather than substrings — no key that holds drawing-derived content
    forbidden = {"bbox", "polyline", "string", "detection", "correction",
                 "examples", "raster", "symbol_class", "text_class",
                 "line_class", "evidence", "provenance", "candidates"}

    def keys(value) -> set:
        if isinstance(value, dict):
            return set(value) | {k for v in value.values() for k in keys(v)}
        if isinstance(value, list):
            return {k for v in value for k in keys(v)}
        return set()

    assert not keys(report) & forbidden
    # and the detection identifiers the activity log holds stay in the
    # local store: the report counts events, it does not list them
    assert "p3-" not in serialized and "detection_id" not in serialized


def test_cli_prints_the_kpi_with_its_basis_and_writes_the_report(
        reviewed_run, tmp_path, synthetic_profile, capsys):
    out = tmp_path / "kpi.json"
    main([str(reviewed_run), "--out", str(out)])
    printed = capsys.readouterr().out
    assert "kpi-fixture.pdf" in printed
    assert "corrections per 100 symbols: 20.0" in printed
    assert "reviewer minutes per accepted Sheet: 8.0" in printed
    assert "basis" in printed and "10" in printed
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written == kpi_report([run_kpi(reviewed_run)])

    # not yet measurable is said, never printed as a zero
    main([str(_digitize(tmp_path / "fresh", synthetic_profile))])
    printed = capsys.readouterr().out
    assert "corrections per 100 symbols: not yet measurable" in printed
    assert "reviewer minutes per accepted Sheet: not yet measurable" \
        in printed


def test_cli_takes_many_runs_and_an_override_labels_dir_for_one(
        tmp_path, synthetic_profile, capsys):
    a = _digitize(tmp_path / "a", synthetic_profile, "doc-a.pdf")
    b = _digitize(tmp_path / "b", synthetic_profile, "doc-b.pdf")
    main([str(a), str(b), "--idle-minutes", "3"])
    printed = capsys.readouterr().out
    assert "doc-a.pdf" in printed and "doc-b.pdf" in printed
    assert "3" in printed  # the threshold the basis names

    # verdicts kept under an overridden labels directory (the Workbench's
    # --labels-dir) are found there — for one run; two would be ambiguous
    elsewhere = tmp_path / "elsewhere"
    record = _record(a, 1)
    LabelStore(elsewhere).record(record["profile"], 1, make_example(
        "symbol", record["symbols"][0], "reject"))
    main([str(a), "--labels-dir", str(elsewhere)])
    assert "corrections per 100 symbols: 100.0" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main([str(a), str(b), "--labels-dir", str(elsewhere)])
