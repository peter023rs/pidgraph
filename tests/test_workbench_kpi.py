"""Workbench KPI seam tests (ticket 19): the Review Workbench records its
activity — Sheets opened, verdicts saved — at the server's clock beside
the verdicts, and shows the operator the product KPI recomputed from the
persisted verdicts and activity on every request, with its measurement
basis. Flask test client over prepared run artifacts, a fake clock
standing in for the server's; offline throughout."""

import json
import re
from pathlib import Path

import pytest
from conftest import FakeClock, T0

from pidgraph.model import Document, SheetAnnotations
from pidgraph.pipeline import digitize
from pidgraph.workbench import create_app

_KIND_FIELDS = (("symbol", "symbols"), ("line", "lines"), ("text", "texts"))


@pytest.fixture
def run_dir(tmp_path, synthetic_profile) -> Path:
    """Two drawn Sheets (16 detections each) and one blank Sheet."""
    from conftest import build_sheet, build_synthetic_sheet

    document = Document(
        name="workbench-kpi.pdf",
        sheets=(build_synthetic_sheet(1), build_synthetic_sheet(2),
                build_sheet(3, SheetAnnotations())))
    out = tmp_path / "run"
    digitize(document, synthetic_profile, out_dir=out)
    return out


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def client(run_dir, clock):
    app = create_app(run_dir, clock=clock)
    app.testing = True
    return app.test_client()


def _activity(run_dir: Path, labels: str = "labels") -> list[dict]:
    path = run_dir / labels / "activity.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()]


def _record(run_dir: Path, sheet: int) -> dict:
    return json.loads((run_dir / "detections" / f"sheet_{sheet}.json")
                      .read_text(encoding="utf-8"))


def _detections(record: dict) -> list[tuple[str, dict]]:
    return [(kind, d) for kind, field in _KIND_FIELDS
            for d in record[field]]


def _queue(client, sheet: int) -> dict:
    response = client.get(f"/sheet/{sheet}/queue")
    assert response.status_code == 200
    return response.get_json()


def _give(client, sheet: int, detection_id: str, verdict: str,
          correction: dict | None = None):
    payload = {"detection_id": detection_id, "verdict": verdict}
    if correction is not None:
        payload["correction"] = correction
    return client.post(f"/sheet/{sheet}/verdicts", json=payload)


def _review_sheet_fully(client, clock: FakeClock, run_dir: Path, sheet: int,
                        seconds_apart: float,
                        reject_first_symbol: bool = False) -> None:
    """Open the Sheet, then give every detection a verdict in record order,
    the clock advancing seconds_apart before each — all pass, or the
    first symbol rejected."""
    assert client.get(f"/sheet/{sheet}").status_code == 200
    for index, (kind, detection) in enumerate(
            _detections(_record(run_dir, sheet))):
        clock.tick(seconds_apart)
        verdict = ("reject" if reject_first_symbol and kind == "symbol"
                   and index == 0 else "pass")
        response = _give(client, sheet, detection["id"], verdict)
        assert response.status_code == 200, response.get_data(as_text=True)
    assert _queue(client, sheet)["remaining"] == 0


def _kpi_value(html: str, key: str) -> str:
    match = re.search(rf'data-kpi="{key}">([^<]*)<', html)
    assert match, key
    return match.group(1)


def test_opening_a_sheet_and_saving_verdicts_are_logged_at_server_time(
        client, clock, run_dir):
    assert _activity(run_dir) == []
    assert client.get("/sheet/1").status_code == 200
    clock.tick(45)
    first = _queue(client, 1)["queue"][0]
    assert _give(client, 1, first["id"], "reject").status_code == 200
    clock.tick(15)
    text = _record(run_dir, 1)["texts"][0]
    assert _give(client, 1, text["id"], "edit",
                 {"string": "T-101A"}).status_code == 200

    # one line per event, in order, at the server's clock — identifiers
    # and verdicts only, never the detection or its correction
    assert _activity(run_dir) == [
        {"at": T0, "event": "open", "sheet": 1},
        {"at": "2026-08-21T09:00:45+00:00", "event": "verdict", "sheet": 1,
         "detection_id": first["id"], "verdict": "reject"},
        {"at": "2026-08-21T09:01:00+00:00", "event": "verdict", "sheet": 1,
         "detection_id": text["id"], "verdict": "edit"},
    ]


def test_refused_verdicts_reads_and_absent_sheets_log_nothing(
        client, run_dir):
    assert client.get("/sheet/9").status_code == 404
    assert _give(client, 1, "p9-sym99", "pass").status_code == 400
    front = _queue(client, 1)["queue"][0]["id"]
    assert _give(client, 1, front, "accept").status_code == 400
    # the queue, verdict listing, progress, raster, index and the KPI
    # itself are reads, not review activity
    for url in ("/sheet/1/queue", "/sheet/1/verdicts", "/progress",
                "/sheet/1/raster.png", "/", "/kpi"):
        assert client.get(url).status_code == 200, url
    assert _activity(run_dir) == []
    assert not (run_dir / "labels").exists()


def test_the_kpi_is_served_from_persisted_verdicts_and_activity(
        client, clock, run_dir):
    # nothing reviewed yet: the KPI says so rather than reporting zeros
    report = client.get("/kpi").get_json()
    assert report["overall"]["corrections_per_100_symbols"] is None
    assert report["overall"]["reviewer_minutes_per_accepted_sheet"] is None
    assert report["documents"][0]["document"] == "workbench-kpi.pdf"
    assert report["basis"]["idle_threshold_minutes"] == 10.0
    assert "break" in report["basis"]["reviewer_minutes_per_accepted_sheet"]

    # Sheet 2 reviewed completely: opened, then 16 verdicts 15 s apart
    # with one symbol rejected -> 1 of 7 symbols corrected, 4.0 minutes
    _review_sheet_fully(client, clock, run_dir, 2, seconds_apart=15,
                        reject_first_symbol=True)
    # Sheet 1 touched only: opened, one symbol passed, nothing more
    assert client.get("/sheet/1").status_code == 200
    clock.tick(30)
    assert _give(client, 1, _record(run_dir, 1)["symbols"][0]["id"],
                 "pass").status_code == 200

    report = client.get("/kpi").get_json()
    overall = report["overall"]
    # 1 correction against 8 symbols reviewed (7 on Sheet 2, 1 on Sheet 1)
    assert overall["corrections_per_100_symbols"] == pytest.approx(12.5)
    assert overall["reviewer_minutes_per_accepted_sheet"] == \
        pytest.approx(4.0)
    assert overall["by_kind"]["symbol"] == {
        "reviewed": 8, "pass": 7, "reject": 1, "edit": 0, "corrections": 1}
    assert overall["sheets_summary"] == {
        "total": 3, "accepted": 1, "accepted_timed": 1,
        "accepted_untimed": 0, "unreadable": 0}
    assert overall["minutes"] == {
        "total": pytest.approx(4.5),
        "on_timed_accepted_sheets": pytest.approx(4.0)}
    rows = {row["sheet"]: row for row in report["documents"][0]["sheets"]}
    assert rows[2]["accepted"] is True
    assert rows[2]["minutes"] == pytest.approx(4.0)
    assert rows[1]["accepted"] is False
    assert rows[1]["minutes"] == pytest.approx(0.5)
    assert rows[3]["accepted"] is False  # blank: nothing was reviewed
    assert report["profiles"][0]["profile"] == {"name": "synthetic-test",
                                                "version": "0"}

    # a Workbench restarted over the same run directory reports the same
    # KPI: it was never session memory
    reopened = create_app(run_dir).test_client()
    assert reopened.get("/kpi").get_json() == report


def test_index_shows_the_kpi_with_its_measurement_basis(
        client, clock, run_dir):
    html = client.get("/").get_data(as_text=True)
    assert "Corrections per 100 symbols" in html
    assert "Reviewer minutes per accepted Sheet" in html
    assert _kpi_value(html, "corrections_per_100_symbols") == \
        "not yet measurable"
    assert _kpi_value(html, "reviewer_minutes_per_accepted_sheet") == \
        "not yet measurable"
    # the basis travels with the numbers, and the full report is a click
    assert "10 minutes" in html and "unmeasured" in html
    assert "reject + edit" in html
    assert 'href="/kpi"' in html

    # Sheet 2 reviewed, all pass, 16 verdicts 30 s apart -> 8.0 minutes
    _review_sheet_fully(client, clock, run_dir, 2, seconds_apart=30)
    html = client.get("/").get_data(as_text=True)
    assert _kpi_value(html, "corrections_per_100_symbols") == "0.0"
    assert _kpi_value(html, "reviewer_minutes_per_accepted_sheet") == "8.0"
    assert "7 symbols reviewed" in html
    assert "1 accepted of 3 Sheets" in html


def test_activity_lives_beside_the_verdicts_wherever_they_are(
        run_dir, tmp_path, clock):
    elsewhere = tmp_path / "elsewhere"
    client = create_app(run_dir, labels_dir=elsewhere,
                        clock=clock).test_client()
    assert client.get("/sheet/1").status_code == 200
    assert (elsewhere / "activity.jsonl").is_file()
    assert not (run_dir / "labels").exists()
    # ...and the KPI reads both from there
    document = client.get("/kpi").get_json()["documents"][0]
    assert document["labels_root"] == str(elsewhere)
    assert document["activity_log"]["events"] == 1


def test_the_idle_threshold_is_configurable_and_stated(run_dir, clock):
    client = create_app(run_dir, clock=clock,
                        idle_minutes=0.5).test_client()
    assert client.get("/kpi").get_json()["basis"][
        "idle_threshold_minutes"] == 0.5
    assert "0.5 minutes" in client.get("/").get_data(as_text=True)
