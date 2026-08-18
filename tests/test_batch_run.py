"""Batch-run seam tests (ticket 08): run_batch() drives a multi-Sheet
Document with per-Sheet progress, per-Sheet failure isolation, resumable
state, and an honest run report — failures with reasons, low-confidence
extractions as gaps, every Sheet accounted for. Offline throughout: stub
components on synthetic fixture Sheets."""

import pytest

from dataclasses import replace

from pidgraph.batch import BatchStateError, run_batch
from pidgraph.model import (
    Document,
    Sheet,
    SheetAnnotations,
    TextAnnotation,
)
from pidgraph.pipeline import digitize
from pidgraph.seams import (
    SYMBOL_DETECTORS,
    PipelineConfig,
    StubSymbolDetector,
)

from conftest import (
    LINES,
    SHEET_H,
    SHEET_W,
    SYMBOLS,
    TEXTS,
    build_sheet,
    build_synthetic_sheet,
)


class FlakySymbolDetector(StubSymbolDetector):
    """Failure injection at the SymbolDetector seam: raises whatever the
    test planted for a Sheet number, otherwise behaves as the stub."""

    raise_on: dict[int, BaseException] = {}

    def detect(self, sheet, profile):
        error = self.raise_on.get(sheet.number)
        if error is not None:
            raise error
        return super().detect(sheet, profile)


@pytest.fixture
def flaky_config(monkeypatch):
    monkeypatch.setitem(SYMBOL_DETECTORS, "flaky-stub", FlakySymbolDetector)
    monkeypatch.setattr(FlakySymbolDetector, "raise_on", {})
    return PipelineConfig(symbol_detector="flaky-stub")


def _document(*numbers: int) -> Document:
    return Document(name="batch-fixture.pdf",
                    sheets=tuple(build_synthetic_sheet(n) for n in numbers))


def test_batch_run_reports_per_sheet_progress_and_writes_run_artifacts(
        tmp_path, synthetic_profile):
    document = _document(1, 2)
    run_dir = tmp_path / "run"
    events = []

    result = run_batch(document, synthetic_profile, run_dir,
                       progress=events.append)

    # per-Sheet progress was visible while the run moved
    assert [(e.sheet, e.index, e.total, e.status) for e in events] == [
        (1, 1, 2, "started"), (1, 1, 2, "succeeded"),
        (2, 2, 2, "started"), (2, 2, 2, "succeeded")]

    # every Sheet is accounted for, and coverage is stated
    assert [(o["sheet"], o["status"]) for o in result.report["sheets"]] == [
        (1, "succeeded"), (2, "succeeded")]
    assert result.report["coverage"] == {
        "total_sheets": 2, "succeeded": 2, "failed": 0,
        "skipped_by_resume": 0, "digitized": 2}
    assert result.report["document"] == "batch-fixture.pdf"

    # the batch produces the same plant model the single-run seam does
    assert (result.artifacts.plant_model
            == digitize(document, synthetic_profile).plant_model)
    assert (run_dir / "plant_model_dexpi.json").exists()
    assert (run_dir / "plant_graph.cypher").exists()
    assert (run_dir / "run_report.json").exists()
    assert (run_dir / "detections" / "sheet_1.json").exists()
    assert (run_dir / "detections" / "sheet_2.json").exists()


def test_failed_sheet_is_recorded_and_the_batch_continues(
        tmp_path, synthetic_profile):
    # A Sheet the stub detector cannot process (no annotations) — the
    # natural offline failure at the SymbolDetector seam.
    bad = Sheet(number=2, width=SHEET_W, height=SHEET_H)
    document = Document(name="batch-fixture.pdf",
                        sheets=(build_synthetic_sheet(1), bad,
                                build_synthetic_sheet(3)))
    run_dir = tmp_path / "run"
    events = []

    result = run_batch(document, synthetic_profile, run_dir,
                       progress=events.append)

    # the failure is recorded with its reason, and nothing is dropped
    assert {o["sheet"]: o["status"] for o in result.report["sheets"]} == {
        1: "succeeded", 2: "failed", 3: "succeeded"}
    [failure] = result.report["failures"]
    assert failure["sheet"] == 2
    assert "annotations" in failure["reason"]
    assert result.report["coverage"] == {
        "total_sheets": 3, "succeeded": 2, "failed": 1,
        "skipped_by_resume": 0, "digitized": 2}
    assert [(e.sheet, e.status) for e in events if e.status == "failed"] == [
        (2, "failed")]

    # the remaining Sheets' artifacts are intact
    assert (run_dir / "detections" / "sheet_1.json").exists()
    assert not (run_dir / "detections" / "sheet_2.json").exists()
    assert (run_dir / "detections" / "sheet_3.json").exists()
    equipment = result.artifacts.plant_model["conceptualModel"]["Equipment"]
    assert {item["sheet"] for item in equipment} == {1, 3}


def test_interrupted_run_resumes_without_reprocessing_and_converges(
        tmp_path, synthetic_profile, flaky_config):
    document = _document(1, 2, 3)
    run_dir = tmp_path / "run"

    # the run is interrupted on Sheet 2; an interrupt is never a
    # per-Sheet failure — it stops the run
    FlakySymbolDetector.raise_on = {2: KeyboardInterrupt()}
    with pytest.raises(KeyboardInterrupt):
        run_batch(document, synthetic_profile, run_dir, config=flaky_config)

    FlakySymbolDetector.raise_on = {}
    events = []
    result = run_batch(document, synthetic_profile, run_dir,
                       config=flaky_config, progress=events.append)

    # the completed Sheet is not re-processed, and is accounted for
    statuses = [(e.sheet, e.status) for e in events]
    assert (1, "skipped-by-resume") in statuses
    assert (1, "started") not in statuses
    assert {o["sheet"]: o["status"] for o in result.report["sheets"]} == {
        1: "skipped-by-resume", 2: "succeeded", 3: "succeeded"}
    assert result.report["coverage"] == {
        "total_sheets": 3, "succeeded": 2, "failed": 0,
        "skipped_by_resume": 1, "digitized": 3}

    # the resumed run converges to the artifacts of an uninterrupted one
    baseline_dir = tmp_path / "baseline"
    run_batch(document, synthetic_profile, baseline_dir,
              config=flaky_config)
    for name in ("plant_model_dexpi.json", "plant_graph.cypher",
                 "detections/sheet_1.json", "detections/sheet_2.json",
                 "detections/sheet_3.json"):
        assert ((run_dir / name).read_bytes()
                == (baseline_dir / name).read_bytes()), name


def test_failed_sheet_is_retried_on_resume(tmp_path, synthetic_profile,
                                           flaky_config):
    document = _document(1, 2)
    run_dir = tmp_path / "run"

    FlakySymbolDetector.raise_on = {
        2: RuntimeError("detector crashed mid-Sheet")}
    first = run_batch(document, synthetic_profile, run_dir,
                      config=flaky_config, progress=lambda event: None)
    assert {o["sheet"]: o["status"] for o in first.report["sheets"]} == {
        1: "succeeded", 2: "failed"}
    assert first.report["failures"] == [
        {"sheet": 2, "reason": "RuntimeError: detector crashed mid-Sheet"}]

    FlakySymbolDetector.raise_on = {}
    second = run_batch(document, synthetic_profile, run_dir,
                       config=flaky_config, progress=lambda event: None)
    assert {o["sheet"]: o["status"] for o in second.report["sheets"]} == {
        1: "skipped-by-resume", 2: "succeeded"}
    assert second.report["failures"] == []
    assert (run_dir / "detections" / "sheet_2.json").exists()


def test_report_surfaces_unresolved_and_low_confidence_gaps(
        tmp_path, synthetic_profile):
    # "T=999" matches no grammar and no confusion-set repair fits, so the
    # lexicon decoder leaves it unresolved (fail-closed) — a gap, never
    # silently dropped or papered over.
    texts = TEXTS + (TextAnnotation("T=999", "equipment_tag",
                                    (250.0, 60.0, 290.0, 72.0)),)
    sheet = build_sheet(1, SheetAnnotations(symbols=SYMBOLS, lines=LINES,
                                            texts=texts))
    document = Document(name="batch-fixture.pdf", sheets=(sheet,))

    result = run_batch(document, synthetic_profile, tmp_path / "run",
                       progress=lambda event: None)

    [unresolved] = [g for g in result.report["gaps"]
                    if g["kind"] == "unresolved-text"]
    assert unresolved["sheet"] == 1
    assert unresolved["string"] == "T=999"
    assert unresolved["text_class"] == "equipment_tag"
    assert "unresolved" in unresolved["detail"]
    # corrected reads (confidence 0.9) are honest output at the default
    # threshold — not gaps
    assert [g["kind"] for g in result.report["gaps"]] == ["unresolved-text"]

    # a stricter threshold surfaces corrected reads as low-confidence gaps
    strict = run_batch(document, synthetic_profile, tmp_path / "strict",
                       progress=lambda event: None,
                       low_confidence_threshold=0.95)
    assert strict.report["low_confidence_threshold"] == 0.95
    low = [g for g in strict.report["gaps"] if g["kind"] == "low-confidence"]
    assert "T-101" in {g["string"] for g in low}
    assert all(g["confidence"] < 0.95 for g in low)
    # the unresolved read is reported once, as unresolved — not twice
    assert [g["string"] for g in strict.report["gaps"]
            if g["kind"] == "unresolved-text"] == ["T=999"]


def test_resuming_against_state_from_a_different_run_is_refused(
        tmp_path, synthetic_profile):
    document = _document(1, 2)
    run_dir = tmp_path / "run"
    run_batch(document, synthetic_profile, run_dir,
              progress=lambda event: None)

    renamed = replace(document, name="other-document.pdf")
    with pytest.raises(BatchStateError) as excinfo:
        run_batch(renamed, synthetic_profile, run_dir,
                  progress=lambda event: None)
    assert "document" in str(excinfo.value)

    other_profile = replace(synthetic_profile, version="v999")
    with pytest.raises(BatchStateError) as excinfo:
        run_batch(document, other_profile, run_dir,
                  progress=lambda event: None)
    assert "profile" in str(excinfo.value)

    # a profile edited under an unchanged version is still a different run
    edited_profile = replace(
        synthetic_profile,
        tag_grammar=dict(synthetic_profile.tag_grammar,
                         equipment_tag=r"[A-Z]{2}-\d{4}"))
    with pytest.raises(BatchStateError) as excinfo:
        run_batch(document, edited_profile, run_dir,
                  progress=lambda event: None)
    assert "profile" in str(excinfo.value)

    # the mismatch is caught before the unknown component would resolve —
    # the state refusal is the message the operator needs
    with pytest.raises(BatchStateError):
        run_batch(document, synthetic_profile, run_dir,
                  config=PipelineConfig(symbol_detector="not-the-stub"),
                  progress=lambda event: None)


def test_default_progress_is_printed_for_the_operator(
        tmp_path, synthetic_profile, capsys):
    run_batch(_document(1, 2), synthetic_profile, tmp_path / "run")

    stderr = capsys.readouterr().err
    assert "Sheet 1 (1/2): succeeded" in stderr
    assert "Sheet 2 (2/2): succeeded" in stderr


def test_unreadable_state_file_is_refused_not_mistaken_for_state(
        tmp_path, synthetic_profile):
    document = _document(1)
    run_dir = tmp_path / "run"
    run_batch(document, synthetic_profile, run_dir,
              progress=lambda event: None)
    state_file = run_dir / "state" / "sheet_1.json"
    state_file.write_text("{truncated", encoding="utf-8")

    with pytest.raises(BatchStateError) as excinfo:
        run_batch(document, synthetic_profile, run_dir,
                  progress=lambda event: None)

    message = str(excinfo.value)
    assert str(state_file) in message
    assert "Sheet 1" in message
