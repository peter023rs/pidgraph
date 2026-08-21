"""The eval harness (ticket 15): a configured pipeline scored against a
labeled eval set, gates stated pass/fail, two configurations side by
side. Verified end to end with the stub seams on the synthetic fixture
Sheet — offline, deterministic, no models."""

import json
from pathlib import Path

import pytest
from conftest import build_synthetic_sheet

from pidgraph import seams
from pidgraph.assemble import build_plant_graph
from pidgraph.eval_harness import (
    HARNESS_VERSION,
    compare,
    evaluate,
    load_eval_set,
    main,
)
from pidgraph.labels import LabelStore, make_example
from pidgraph.pipeline import digitize_sheet
from pidgraph.pngio import encode_gray_png
from pidgraph.seams import PipelineConfig, build_components

BUNDLE = str(Path(__file__).parent / "fixtures" / "profiles"
             / "synthetic-test")


def _write_eval_set(root, profile, record, assembly, sheet):
    """Lay an eval set out the way the label factory does: labels as
    pass verdicts in the LabelStore schema, connectivity as plant-graph
    edges keyed by symbol detection ids, the Sheet raster as PNG."""
    identity = profile.identity_record()
    store = LabelStore(root / "labels")
    store.record_many(identity, sheet.number, (
        [make_example("symbol", s, "pass") for s in record["symbols"]]
        + [make_example("text", t, "pass") for t in record["texts"]]
        + [make_example("line", l, "pass") for l in record["lines"]]))

    tag_to_id = {t.node_tag(): t.detection.id for t in assembly.terminals}
    links = [{"source": tag_to_id[edge["source"]],
              "target": tag_to_id[edge["target"]],
              "attributes": edge["attributes"]}
             for edge in build_plant_graph([assembly])["edges"]]
    connectivity = root / "connectivity" / f"sheet_{sheet.number}.json"
    connectivity.parent.mkdir(parents=True, exist_ok=True)
    connectivity.write_text(json.dumps(
        {"profile": identity, "sheet": sheet.number, "links": links}))

    png = root / "sheets" / f"sheet_{sheet.number}.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    png.write_bytes(encode_gray_png(sheet.width, sheet.height,
                                    sheet.raster))
    return root


@pytest.fixture
def eval_root(tmp_path, synthetic_profile):
    sheet = build_synthetic_sheet(1)
    detector, recognizer, _ = build_components(PipelineConfig())
    record, assembly = digitize_sheet(sheet, synthetic_profile,
                                      detector, recognizer)
    return _write_eval_set(tmp_path / "evalset", synthetic_profile,
                           record, assembly, sheet)


class _DropInstrumentation:
    """A deliberately lossy detector: the stub minus the instrument
    bubble and the flow arrow — 5 of the fixture's 7 symbols, so its
    symbol F1 of 10/12 sits under the 0.90 gate while the piping
    topology (and with it the connectivity gate) survives."""

    def detect(self, sheet, profile):
        return [s for s in seams.StubSymbolDetector().detect(sheet, profile)
                if s.symbol_class not in ("instrument_bubble", "flow_arrow")]


class _Explodes:
    def detect(self, sheet, profile):
        raise RuntimeError("no model behind this seam")


# --------------------------------------------------------------------------
# The stub round trip: the harness is verifiable before real models exist

def test_the_stub_pipeline_passes_every_gate(eval_root, synthetic_profile):
    eval_sheets = load_eval_set(eval_root, synthetic_profile)
    report = evaluate(eval_sheets, synthetic_profile, PipelineConfig())

    assert report["harness_version"] == HARNESS_VERSION
    assert report["passed"] is True
    assert [g["passed"] for g in report["gates"]] == [True, True, True]
    assert report["metrics"]["symbol"]["f1"] == 1.0
    assert report["metrics"]["tag"]["exact_match"] == 1.0
    assert report["metrics"]["connectivity"]["f1"] == 1.0
    assert report["coverage"] == {"sheets": 1, "scored": 1, "failed": 0}
    assert report["failures"] == []
    assert report["config"] == {"symbol_detector": "stub",
                                "text_recognizer": "stub",
                                "graph_store": "cypher-script"}


def test_direction_evidence_is_reported_beside_the_gate(
        eval_root, synthetic_profile):
    eval_sheets = load_eval_set(eval_root, synthetic_profile)
    report = evaluate(eval_sheets, synthetic_profile, PipelineConfig())
    direction = report["metrics"]["connectivity"]["direction"]
    assert direction["truth_known"] >= 1  # run1 carries a flow arrow
    assert direction["agreeing"] == direction["truth_known"]
    assert direction["conflicting"] == 0


def test_the_report_carries_per_sheet_scores(eval_root, synthetic_profile):
    eval_sheets = load_eval_set(eval_root, synthetic_profile)
    report = evaluate(eval_sheets, synthetic_profile, PipelineConfig())
    assert [s["sheet"] for s in report["per_sheet"]] == [1]
    assert report["per_sheet"][0]["symbol"]["f1"] == 1.0


# --------------------------------------------------------------------------
# Side by side: any configured seam implementation is scorable

def test_two_configurations_score_side_by_side(
        eval_root, synthetic_profile, monkeypatch):
    monkeypatch.setitem(seams.SYMBOL_DETECTORS, "lossy",
                        _DropInstrumentation)
    eval_sheets = load_eval_set(eval_root, synthetic_profile)
    report = compare(eval_sheets, synthetic_profile, {
        "stub": PipelineConfig(),
        "lossy": PipelineConfig(symbol_detector="lossy")})

    assert report["harness_version"] == HARNESS_VERSION
    by_name = {c["name"]: c for c in report["configurations"]}
    assert set(by_name) == {"stub", "lossy"}
    assert by_name["stub"]["passed"] is True

    lossy_gates = {g["metric"]: g for g in by_name["lossy"]["gates"]}
    assert by_name["lossy"]["passed"] is False
    assert lossy_gates["symbol_f1"]["score"] == pytest.approx(10 / 12)
    assert lossy_gates["symbol_f1"]["passed"] is False
    assert lossy_gates["connectivity_f1"]["passed"] is True


def test_a_comparison_needs_two_configurations(eval_root, synthetic_profile):
    eval_sheets = load_eval_set(eval_root, synthetic_profile)
    with pytest.raises(ValueError, match="two"):
        compare(eval_sheets, synthetic_profile,
                {"only": PipelineConfig()})


# --------------------------------------------------------------------------
# Honest accounting: a Sheet the pipeline cannot digitize scores as misses

def test_a_failing_sheet_counts_its_truth_as_misses(
        eval_root, synthetic_profile, monkeypatch):
    monkeypatch.setitem(seams.SYMBOL_DETECTORS, "explodes", _Explodes)
    eval_sheets = load_eval_set(eval_root, synthetic_profile)
    report = evaluate(eval_sheets, synthetic_profile,
                      PipelineConfig(symbol_detector="explodes"))

    assert report["coverage"] == {"sheets": 1, "scored": 0, "failed": 1}
    assert report["failures"] == [
        {"sheet": 1, "reason": "RuntimeError: no model behind this seam"}]
    assert report["metrics"]["symbol"]["f1"] == 0.0
    assert report["metrics"]["tag"]["exact_match"] == 0.0
    assert report["metrics"]["connectivity"]["f1"] == 0.0
    assert report["passed"] is False


# --------------------------------------------------------------------------
# Ground truth honors verdicts; the eval set fails closed when incomplete

def test_verdicts_shape_the_ground_truth(tmp_path, synthetic_profile):
    identity = synthetic_profile.identity_record()
    root = tmp_path / "evalset"
    detection = {"id": "p1-sym0", "sheet": 1, "symbol_class": "tank",
                 "bbox": [20.0, 80.0, 60.0, 140.0], "confidence": 1.0,
                 "provenance": {"component": "c", "evidence": "e"},
                 "direction": None}
    edited = dict(detection, id="p1-sym1", bbox=[180.0, 100.0, 200.0, 120.0])
    rejected = dict(detection, id="p1-sym2", bbox=[300.0, 100.0, 320.0, 120.0])
    LabelStore(root / "labels").record_many(identity, 1, [
        make_example("symbol", detection, "pass"),
        make_example("symbol", edited, "edit",
                     {"bbox": [185.0, 100.0, 205.0, 120.0]}),
        make_example("symbol", rejected, "reject")])
    (root / "connectivity").mkdir(parents=True)
    (root / "connectivity" / "sheet_1.json").write_text(json.dumps(
        {"profile": identity, "sheet": 1, "links": []}))
    (root / "sheets").mkdir()
    (root / "sheets" / "sheet_1.png").write_bytes(
        encode_gray_png(4, 4, bytes(16)))

    (eval_sheet,) = load_eval_set(root, synthetic_profile)
    truth = {s["id"]: s for s in eval_sheet.symbols}
    assert set(truth) == {"p1-sym0", "p1-sym1"}  # the reject is no truth
    assert truth["p1-sym1"]["bbox"] == (185.0, 100.0, 205.0, 120.0)
    annotated = {a.bbox for a in eval_sheet.sheet.annotations.symbols}
    assert (185.0, 100.0, 205.0, 120.0) in annotated  # correction applied
    assert (300.0, 100.0, 320.0, 120.0) not in annotated


def test_only_grammar_classes_are_tags(eval_root, synthetic_profile):
    (eval_sheet,) = load_eval_set(eval_root, synthetic_profile)
    classes = {t["text_class"] for t in eval_sheet.tags}
    assert classes <= set(synthetic_profile.tag_grammar)


def test_an_eval_sheet_missing_connectivity_is_refused(
        eval_root, synthetic_profile):
    (eval_root / "connectivity" / "sheet_1.json").unlink()
    with pytest.raises(ValueError, match="connectivity"):
        load_eval_set(eval_root, synthetic_profile)


def test_an_eval_sheet_missing_its_raster_is_refused(
        eval_root, synthetic_profile):
    (eval_root / "sheets" / "sheet_1.png").unlink()
    with pytest.raises(ValueError, match="raster"):
        load_eval_set(eval_root, synthetic_profile)


def test_an_empty_eval_set_is_refused(tmp_path, synthetic_profile):
    with pytest.raises(ValueError, match="no labeled Sheets"):
        load_eval_set(tmp_path / "nothing", synthetic_profile)


def test_an_unknown_sheet_selection_is_refused(eval_root, synthetic_profile):
    with pytest.raises(ValueError, match="Sheet"):
        load_eval_set(eval_root, synthetic_profile, sheets=[2])


# --------------------------------------------------------------------------
# CLI: on-demand local runs, pass/fail stated, exit status carries the gate

def test_cli_writes_the_report_and_states_pass(
        eval_root, synthetic_profile, tmp_path, capsys):
    out = tmp_path / "report.json"
    main([str(eval_root), BUNDLE, "--out", str(out)])

    report = json.loads(out.read_text())
    assert report["passed"] is True
    printed = capsys.readouterr().out
    assert "symbol_f1" in printed and "PASS" in printed


def test_cli_exits_nonzero_when_a_gate_fails(
        eval_root, synthetic_profile, monkeypatch, capsys):
    monkeypatch.setitem(seams.SYMBOL_DETECTORS, "lossy",
                        _DropInstrumentation)
    with pytest.raises(SystemExit) as excinfo:
        main([str(eval_root), BUNDLE,
              "--config", "lossy:symbol_detector=lossy"])
    assert excinfo.value.code == 1
    assert "FAIL" in capsys.readouterr().out


def test_cli_compares_two_configurations(
        eval_root, synthetic_profile, monkeypatch, tmp_path, capsys):
    monkeypatch.setitem(seams.SYMBOL_DETECTORS, "lossy",
                        _DropInstrumentation)
    out = tmp_path / "comparison.json"
    with pytest.raises(SystemExit):  # the lossy configuration fails
        main([str(eval_root), BUNDLE, "--out", str(out),
              "--config", "stub",
              "--config", "lossy:symbol_detector=lossy"])

    report = json.loads(out.read_text())
    assert [c["name"] for c in report["configurations"]] == ["stub", "lossy"]
    printed = capsys.readouterr().out
    assert "stub" in printed and "lossy" in printed


def test_cli_refuses_an_unknown_seam_key(eval_root):
    with pytest.raises(SystemExit) as excinfo:
        main([str(eval_root), BUNDLE, "--config", "bad:not_a_seam=x"])
    assert excinfo.value.code == 2  # argparse usage error


def test_cli_refuses_a_nameless_configuration(eval_root):
    with pytest.raises(SystemExit) as excinfo:
        main([str(eval_root), BUNDLE, "--config", ""])
    assert excinfo.value.code == 2  # argparse usage error
