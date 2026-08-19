"""Review-flow seam tests (ticket 11): confidence-sorted queues and
per-Sheet review progress, via the Flask test client against prepared run
artifacts. The queue endpoint presents undecided detections lowest
confidence first; the progress view derives each Sheet's review state from
persisted verdict coverage, never from session memory. Offline throughout:
artifacts come from the stub pipeline on synthetic Sheets."""

import json
import re
from pathlib import Path

import pytest

from pidgraph.labels import LabelStore, make_example
from pidgraph.model import Document, SheetAnnotations
from pidgraph.pipeline import digitize
from pidgraph.workbench import create_app

# One confidence per detection of the synthetic fixture Sheet, assigned in
# record order (7 symbols, 3 lines, 6 texts). Scrambled so queue order is
# nothing like record order, with exactly one tie (0.30) to pin that ties
# keep record order.
_CONFIDENCES = (0.62, 0.05, 0.93, 0.41, 0.77, 0.18, 0.30,   # symbols
                0.88, 0.30, 0.54,                           # lines
                0.71, 0.09, 0.47, 0.26, 0.99, 0.35)         # texts

_KIND_FIELDS = (("symbol", "symbols"), ("line", "lines"), ("text", "texts"))


def _record(run_dir: Path, sheet: int) -> dict:
    return json.loads((run_dir / "detections" / f"sheet_{sheet}.json")
                      .read_text(encoding="utf-8"))


def _detections(record: dict) -> list[tuple[str, dict]]:
    return [(kind, d) for kind, field in _KIND_FIELDS
            for d in record[field]]


def _scramble_confidences(run_dir: Path, sheet: int) -> None:
    """Prepare the run artifact with known, varied confidences — the
    Workbench reads artifacts only, so the queue order under test is
    entirely determined by what the record says."""
    path = run_dir / "detections" / f"sheet_{sheet}.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    detections = [d for _, d in _detections(record)]
    assert len(detections) == len(_CONFIDENCES)
    for detection, confidence in zip(detections, _CONFIDENCES):
        detection["confidence"] = confidence
    path.write_text(json.dumps(record), encoding="utf-8")


@pytest.fixture
def run_dir(tmp_path, synthetic_profile) -> Path:
    """A four-Sheet run: three drawn Sheets (a 400-Sheet Document in
    miniature) and one blank Sheet with nothing detected on it."""
    from conftest import build_sheet, build_synthetic_sheet

    document = Document(
        name="multi-sheet.pdf",
        sheets=(build_synthetic_sheet(1), build_synthetic_sheet(2),
                build_synthetic_sheet(3),
                build_sheet(4, SheetAnnotations())))
    out = tmp_path / "run"
    digitize(document, synthetic_profile, out_dir=out)
    _scramble_confidences(out, 1)
    return out


@pytest.fixture
def client(run_dir):
    app = create_app(run_dir)
    app.testing = True
    return app.test_client()


def _queue(client, sheet: int, kind: str | None = None) -> dict:
    url = f"/sheet/{sheet}/queue"
    if kind is not None:
        url += f"?kind={kind}"
    response = client.get(url)
    assert response.status_code == 200
    return response.get_json()


def _correction_for(kind: str) -> dict:
    return {"symbol": {"bbox": [0.0, 0.0, 10.0, 10.0]},
            "line": {"polyline": [[0.0, 0.0], [10.0, 10.0]]},
            "text": {"string": "X-1"}}[kind]


def _give(client, sheet: int, detection_id: str, verdict: str,
          correction: dict | None = None) -> None:
    payload = {"detection_id": detection_id, "verdict": verdict}
    if correction is not None:
        payload["correction"] = correction
    response = client.post(f"/sheet/{sheet}/verdicts", json=payload)
    assert response.status_code == 200, response.get_data(as_text=True)


def _progress(client) -> dict:
    response = client.get("/progress")
    assert response.status_code == 200
    return {row["sheet"]: row for row in response.get_json()["sheets"]}


def _review_sheet_fully(client, sheet: int) -> None:
    """Walk the Sheet's queue to empty — bounded, so a regression in
    queue advancement fails the test instead of hanging the suite."""
    for _ in range(_queue(client, sheet)["total"]):
        data = _queue(client, sheet)
        if not data["remaining"]:
            break
        _give(client, sheet, data["queue"][0]["id"], "pass")
    assert _queue(client, sheet)["remaining"] == 0


def test_queue_presents_undecided_detections_lowest_confidence_first(
        client, run_dir):
    data = _queue(client, 1)
    assert data["sheet"] == 1
    assert data["kind"] is None
    assert data["total"] == 16
    assert data["remaining"] == 16

    # sorted ascending by the confidences the artifact records; the one
    # tie (two detections at 0.30) keeps record order — deterministic
    # across requests and restarts
    expected = [d["id"] for _, d in
                sorted(_detections(_record(run_dir, 1)),
                       key=lambda item: item[1]["confidence"])]
    assert [entry["id"] for entry in data["queue"]] == expected
    confidences = [entry["confidence"] for entry in data["queue"]]
    assert confidences == sorted(confidences)
    assert confidences[0] == 0.05

    # each entry names its detection kind, matching the record
    kinds = {d["id"]: kind for kind, d in _detections(_record(run_dir, 1))}
    assert all(entry["kind"] == kinds[entry["id"]]
               for entry in data["queue"])


def test_giving_a_verdict_advances_the_queue(client):
    # pass, reject and edit all decide a detection: each advances the
    # queue to the next-lowest-confidence undecided detection
    for verdict in ("pass", "reject", "edit"):
        before = _queue(client, 1)
        front = before["queue"][0]
        correction = (_correction_for(front["kind"])
                      if verdict == "edit" else None)
        _give(client, 1, front["id"], verdict, correction)

        after = _queue(client, 1)
        assert after["remaining"] == before["remaining"] - 1
        assert after["total"] == before["total"]  # scope unchanged
        assert front["id"] not in {e["id"] for e in after["queue"]}
        assert after["queue"][0] == before["queue"][1]


def test_queue_scopes_per_detection_kind(client, run_dir):
    by_kind = {kind: _queue(client, 1, kind)
               for kind in ("symbol", "line", "text")}
    assert {k: q["total"] for k, q in by_kind.items()} == \
        {"symbol": 7, "line": 3, "text": 6}
    for kind, data in by_kind.items():
        assert data["kind"] == kind
        assert all(entry["kind"] == kind for entry in data["queue"])
        confidences = [entry["confidence"] for entry in data["queue"]]
        assert confidences == sorted(confidences)

    # a kind-scoped queue advances on its own verdicts, not another
    # kind's: deciding a text leaves the symbol queue untouched
    _give(client, 1, by_kind["text"]["queue"][0]["id"], "pass")
    assert _queue(client, 1, "text")["remaining"] == 5
    assert _queue(client, 1, "symbol")["remaining"] == 7

    # the unscoped queue spans every kind, so it advanced too
    assert _queue(client, 1)["remaining"] == 15


def test_unknown_kinds_and_absent_sheets_are_refused(client):
    assert client.get("/sheet/1/queue?kind=bogus").status_code == 400
    assert client.get("/sheet/9/queue").status_code == 404


def test_a_blank_sheet_presents_an_empty_queue(client):
    assert _queue(client, 4) == {"sheet": 4, "kind": None,
                                 "total": 0, "remaining": 0, "queue": []}
    assert _queue(client, 4, "symbol") == {
        "sheet": 4, "kind": "symbol",
        "total": 0, "remaining": 0, "queue": []}


def test_sheet_identity_is_the_artifact_filename(run_dir):
    """The number a Sheet is served under — its artifact filename — keys
    everything: verdicts, queue and progress. A copied or renamed
    artifact whose internal record disagrees stays self-consistent under
    its filename instead of splitting labels across two keys."""
    detections = run_dir / "detections"
    (detections / "sheet_5.json").write_text(
        (detections / "sheet_1.json").read_text(encoding="utf-8"),
        encoding="utf-8")
    client = create_app(run_dir).test_client()

    front = _queue(client, 5)["queue"][0]
    _give(client, 5, front["id"], "pass")

    data = _queue(client, 5)
    assert data["sheet"] == 5
    assert data["remaining"] == 15  # the verdict advanced this queue
    progress = _progress(client)
    assert progress[5] == {"sheet": 5, "state": "in progress",
                           "decided": 1, "total": 16}
    # ...and Sheet 1, whose record the copy carries inside, is untouched
    assert progress[1]["state"] == "untouched"
    assert _queue(client, 1)["remaining"] == 16


def test_a_corrupt_artifact_marks_one_sheet_unreadable_not_the_document(
        run_dir):
    # a run killed mid-write can leave a truncated record behind; the
    # Document-level view degrades to that one Sheet, never to a 500 on
    # the whole review's entry point
    (run_dir / "detections" / "sheet_2.json").write_text(
        "{truncated", encoding="utf-8")
    client = create_app(run_dir).test_client()

    progress = _progress(client)
    assert progress[2] == {"sheet": 2, "state": "unreadable",
                           "decided": 0, "total": 0}
    assert progress[1]["state"] == "untouched"
    assert progress[3]["state"] == "untouched"

    html = client.get("/").get_data(as_text=True)
    assert 'data-state="unreadable"' in html


def test_document_progress_reflects_verdict_coverage(client):
    # untouched run: every drawn Sheet is untouched; the blank Sheet has
    # nothing to review, so no reviewer minutes belong there
    initial = _progress(client)
    assert [initial[n]["state"] for n in (1, 2, 3)] == ["untouched"] * 3
    assert initial[4] == {"sheet": 4, "state": "reviewed",
                          "decided": 0, "total": 0}

    # fully review Sheet 3 by walking its queue to empty
    _review_sheet_fully(client, 3)

    # partially review Sheet 1
    front = _queue(client, 1)["queue"][0]
    _give(client, 1, front["id"], "reject")

    progress = _progress(client)
    assert progress[1] == {"sheet": 1, "state": "in progress",
                           "decided": 1, "total": 16}
    assert progress[2] == {"sheet": 2, "state": "untouched",
                           "decided": 0, "total": 16}
    assert progress[3] == {"sheet": 3, "state": "reviewed",
                           "decided": 16, "total": 16}


def test_progress_recomputes_from_persisted_verdicts_after_restart(
        run_dir):
    first_day = create_app(run_dir).test_client()
    first_two = _queue(first_day, 1)["queue"][:2]
    for entry in first_two:
        _give(first_day, 1, entry["id"], "pass")
    decided = {entry["id"] for entry in first_two}

    # a fresh Workbench over the same run directory — day two
    second_day = create_app(run_dir).test_client()
    progress = _progress(second_day)
    assert progress[1] == {"sheet": 1, "state": "in progress",
                           "decided": 2, "total": 16}
    assert progress[2]["state"] == "untouched"

    # the queue picks up where day one left off: what was decided stays
    # out, and the front is the lowest-confidence undecided detection
    data = _queue(second_day, 1)
    assert data["remaining"] == 14
    assert decided.isdisjoint(entry["id"] for entry in data["queue"])
    confidences = [entry["confidence"] for entry in data["queue"]]
    assert confidences == sorted(confidences)


def test_stale_examples_never_inflate_coverage(run_dir):
    # labels from an older run of the same profile may reference
    # detection ids this run's artifacts no longer contain; review state
    # counts coverage over this Sheet's detections only
    record = _record(run_dir, 1)
    stale = {"id": "p9-sym99", "sheet": 1, "symbol_class": "tank",
             "bbox": [0.0, 0.0, 1.0, 1.0], "confidence": 1.0}
    LabelStore(run_dir / "labels").record(
        record["profile"], 1, make_example("symbol", stale, "pass"))

    client = create_app(run_dir).test_client()
    progress = _progress(client)
    assert progress[1] == {"sheet": 1, "state": "untouched",
                           "decided": 0, "total": 16}
    assert _queue(client, 1)["remaining"] == 16


def test_index_shows_each_sheets_review_state(client):
    _review_sheet_fully(client, 3)
    _give(client, 1, _queue(client, 1)["queue"][0]["id"], "pass")

    html = client.get("/").get_data(as_text=True)
    # each row carries its state and the counts a reviewer splits a
    # multi-day review by — bound to the row's Sheet, not just anywhere
    # on the page
    rows = {int(number): (state, int(decided), int(total))
            for state, number, decided, total in re.findall(
                r'data-state="([^"]+)"><a href="/sheet/(\d+)">'
                r'(?s:.*?)(\d+)/(\d+) reviewed', html)}
    assert rows == {1: ("in progress", 1, 16), 2: ("untouched", 0, 16),
                    3: ("reviewed", 16, 16), 4: ("reviewed", 0, 0)}


def test_sheet_page_offers_queue_driven_review(client):
    html = client.get("/sheet/1").get_data(as_text=True)
    # the queue endpoint drives the page's next-in-queue control
    assert "/sheet/1/queue" in html
    assert 'id="next"' in html
    # scoping control offers every kind the artifacts distinguish
    assert 'id="scope"' in html
    for kind in ("symbol", "line", "text"):
        assert f'value="{kind}"' in html
    # a saved verdict advances the queue, and the queue greets the
    # reviewer on page load rather than waiting for a first click
    assert "await advance()" in html
    assert "\nadvance();" in html
