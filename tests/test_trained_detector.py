"""The trained SymbolDetector (ticket 16): a training procedure turns
label-factory data (clean + degraded) into a versioned detector artifact,
the detector runs behind the SymbolDetector seam selected by
configuration, and the eval harness gates it — all offline on synthetic
glyph Sheets, no GPU, no stored models, no network.

The glyph Sheets drawn here give each symbol class genuinely distinct
ink (rectangle tank, bowtie valve, circle bubble, diamond OPC, filled
nozzle stub, arrowhead) at the pipeline's operating scale, so training
and detection are exercised on real pixels rather than echoed
annotations."""

import json
import math
import re
from dataclasses import replace

import pytest

from pidgraph.assemble import build_plant_graph
from pidgraph.degrade import (
    Noise,
    Raster,
    Skew,
    degrade_bbox,
    degrade_polyline,
    degrade_raster,
)
from pidgraph.detector import TrainedSymbolDetector, main, train_detector
from pidgraph.eval_harness import bbox_iou, evaluate, load_eval_set
from pidgraph.labels import LabelStore, make_example
from pidgraph.model import (
    ConventionProfile,
    Document,
    LineAnnotation,
    Sheet,
    SheetAnnotations,
    SymbolAnnotation,
    TextAnnotation,
)
from pidgraph.normalize import normalize_sheet
from pidgraph.pipeline import digitize, digitize_sheet
from pidgraph.pngio import encode_gray_png
from pidgraph.seams import PipelineConfig, build_components

SHEET_W, SHEET_H = 400, 200


# --------------------------------------------------------------------------
# Glyph Sheets: distinct ink per symbol class at the operating scale

def _stroke(grid: bytearray, width: int, height: int,
            p: tuple, q: tuple) -> None:
    (x0, y0), (x1, y1) = p, q
    steps = max(int(abs(x1 - x0)), int(abs(y1 - y0)), 1)
    for i in range(steps + 1):
        t = i / steps
        x, y = int(round(x0 + (x1 - x0) * t)), int(round(y0 + (y1 - y0) * t))
        if 0 <= x < width and 0 <= y < height:
            grid[y * width + x] = 0


def _draw_symbol(grid: bytearray, width: int, height: int,
                 sym: SymbolAnnotation) -> None:
    x0, y0, x1, y1 = sym.bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    edges = (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
             ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0)))
    if sym.symbol_class == "tank":
        for p, q in edges:
            _stroke(grid, width, height, p, q)
    elif sym.symbol_class == "gate_valve":
        # bowtie: two triangles meeting at the center
        _stroke(grid, width, height, (x0, y0), (x1, y1))
        _stroke(grid, width, height, (x0, y1), (x1, y0))
        _stroke(grid, width, height, (x0, y0), (x0, y1))
        _stroke(grid, width, height, (x1, y0), (x1, y1))
    elif sym.symbol_class == "instrument_bubble":
        radius = (x1 - x0) / 2
        for i in range(64):
            a0 = 2 * math.pi * i / 64
            a1 = 2 * math.pi * (i + 1) / 64
            _stroke(grid, width, height,
                    (cx + radius * math.cos(a0), cy + radius * math.sin(a0)),
                    (cx + radius * math.cos(a1), cy + radius * math.sin(a1)))
    elif sym.symbol_class == "opc":
        _stroke(grid, width, height, (x0, cy), (cx, y0))
        _stroke(grid, width, height, (cx, y0), (x1, cy))
        _stroke(grid, width, height, (x1, cy), (cx, y1))
        _stroke(grid, width, height, (cx, y1), (x0, cy))
    elif sym.symbol_class == "nozzle":
        for yy in range(int(y0), int(y1)):
            _stroke(grid, width, height, (x0, yy), (x1, yy))  # filled stub
    elif sym.symbol_class == "flow_arrow":
        # filled arrowhead over the run's shaft (integer spans, so the
        # glyph is identical at every integer offset)
        length = int(x1 - x0)
        for i in range(length):
            half = (length - i) // 2
            _stroke(grid, width, height,
                    (x0 + i, cy - half), (x0 + i, cy + half))
    else:
        raise AssertionError(f"no glyph for {sym.symbol_class!r}")


def _shifted(bbox: tuple, dx: float, dy: float) -> tuple:
    x0, y0, x1, y1 = bbox
    return (x0 + dx, y0 + dy, x1 + dx, y1 + dy)


def glyph_annotations(dx: float = 0.0, dy: float = 0.0) -> SheetAnnotations:
    """The fixture topology drawn with class-distinct glyphs; dx/dy shift
    the whole drawing so Sheets are not pixel-identical layouts."""
    symbols = (
        SymbolAnnotation("tank", _shifted((30.0, 70.0, 70.0, 130.0), dx, dy)),
        SymbolAnnotation("nozzle", _shifted((66.0, 96.0, 74.0, 104.0), dx, dy)),
        SymbolAnnotation("flow_arrow",
                         _shifted((115.0, 95.0, 125.0, 105.0), dx, dy),
                         direction=(1.0, 0.0)),
        SymbolAnnotation("gate_valve",
                         _shifted((180.0, 90.0, 200.0, 110.0), dx, dy)),
        SymbolAnnotation("instrument_bubble",
                         _shifted((180.0, 30.0, 200.0, 50.0), dx, dy)),
        SymbolAnnotation("tank", _shifted((320.0, 70.0, 360.0, 130.0), dx, dy)),
        SymbolAnnotation("opc", _shifted((355.0, 150.0, 385.0, 170.0), dx, dy)),
    )
    lines = (
        LineAnnotation(((74.0 + dx, 100.0 + dy), (180.0 + dx, 100.0 + dy))),
        LineAnnotation(((200.0 + dx, 100.0 + dy), (320.0 + dx, 100.0 + dy))),
        LineAnnotation(((340.0 + dx, 130.0 + dy), (340.0 + dx, 160.0 + dy),
                        (355.0 + dx, 160.0 + dy))),
    )
    texts = (
        TextAnnotation("T-101", "equipment_tag",
                       _shifted((30.0, 50.0, 70.0, 62.0), dx, dy)),
        TextAnnotation("V-101", "equipment_tag",
                       _shifted((175.0, 115.0, 205.0, 127.0), dx, dy)),
        TextAnnotation("T-102", "equipment_tag",
                       _shifted((320.0, 50.0, 360.0, 62.0), dx, dy)),
        TextAnnotation("PI-100", "instrument_tag",
                       _shifted((178.0, 50.0, 202.0, 62.0), dx, dy)),
        TextAnnotation("150-GA-001", "line_number",
                       _shifted((90.0, 80.0, 150.0, 90.0), dx, dy)),
        TextAnnotation("DW02-0003", "opc_label",
                       _shifted((357.0, 153.0, 383.0, 167.0), dx, dy)),
    )
    return SheetAnnotations(symbols=symbols, lines=lines, texts=texts)


def build_glyph_sheet(number: int, dx: float = 0.0, dy: float = 0.0) -> Sheet:
    annotations = glyph_annotations(dx, dy)
    grid = bytearray(b"\xff" * (SHEET_W * SHEET_H))
    for line in annotations.lines:
        for p, q in zip(line.polyline, line.polyline[1:]):
            _stroke(grid, SHEET_W, SHEET_H, p, q)
    for sym in annotations.symbols:
        _draw_symbol(grid, SHEET_W, SHEET_H, sym)
    return Sheet(number=number, width=SHEET_W, height=SHEET_H,
                 raster=bytes(grid), annotations=annotations)


# --------------------------------------------------------------------------
# Datasets in the label-factory layout (labels/, sheets/)

def _symbol_detection(sheet: Sheet, index: int,
                      sym: SymbolAnnotation) -> dict:
    return {"id": f"p{sheet.number}-sym{index}", "sheet": sheet.number,
            "symbol_class": sym.symbol_class, "bbox": list(sym.bbox),
            "confidence": 1.0,
            "provenance": {"component": "test:glyphs",
                           "evidence": f"drawn {sym.symbol_class}"},
            "direction": (None if sym.direction is None
                          else list(sym.direction))}


def write_dataset(root, profile, sheets):
    """A dataset the way the label factory lays one out: symbol labels as
    pass verdicts in the LabelStore schema, Sheet rasters as PNGs."""
    identity = profile.identity_record()
    store = LabelStore(root / "labels")
    for sheet in sheets:
        store.record_many(identity, sheet.number, [
            make_example("symbol", _symbol_detection(sheet, i, sym), "pass")
            for i, sym in enumerate(sheet.annotations.symbols)])
        png = root / "sheets" / f"sheet_{sheet.number}.png"
        png.parent.mkdir(parents=True, exist_ok=True)
        png.write_bytes(encode_gray_png(sheet.width, sheet.height,
                                        sheet.raster))
    return root


ALL_CLASSES = {"tank", "nozzle", "gate_valve", "instrument_bubble",
               "opc", "flow_arrow"}


# --------------------------------------------------------------------------
# Training: label-factory data in, versioned detector artifact out

def test_training_produces_a_versioned_artifact(tmp_path, synthetic_profile):
    root = write_dataset(tmp_path / "clean", synthetic_profile,
                         [build_glyph_sheet(1)])
    out = tmp_path / "detector"

    manifest = train_detector([root], synthetic_profile.identity_record(),
                              out)

    assert set(manifest["classes"]) == ALL_CLASSES
    for variants in manifest["classes"].values():
        assert variants
        for entry in variants:
            assert (out / entry["prototype"]).is_file()
            assert 0.0 < entry["threshold"] < 1.0
            assert entry["examples"] >= 1
            assert entry["width"] >= 2 and entry["height"] >= 2
    # the labeled arrow direction is learned as the variant's evidence
    (arrow,) = manifest["classes"]["flow_arrow"]
    assert arrow["direction"] == pytest.approx([1.0, 0.0])
    (tank,) = manifest["classes"]["tank"]
    assert tank["direction"] is None
    assert manifest["profile"] == synthetic_profile.identity_record()
    assert re.fullmatch(r"[0-9a-f]{12}", manifest["version"])
    assert json.loads((out / "manifest.json").read_text()) == manifest


def test_training_refuses_an_empty_dataset(tmp_path, synthetic_profile):
    with pytest.raises(ValueError, match="no labeled Sheets"):
        train_detector([tmp_path / "nothing"],
                       synthetic_profile.identity_record(),
                       tmp_path / "detector")


def test_training_refuses_labels_recorded_for_another_profile(
        tmp_path, synthetic_profile):
    """A store file whose recorded identity disagrees with its directory
    key (a case-insensitive filesystem can resolve one profile's
    directory for another's name) is refused, not re-stamped."""
    root = write_dataset(tmp_path / "clean", synthetic_profile,
                         [build_glyph_sheet(1)])
    stored = next((root / "labels").rglob("sheet_1.json"))
    labels = json.loads(stored.read_text())
    labels["profile"] = {"name": "someone-else", "version": "v9"}
    stored.write_text(json.dumps(labels))

    with pytest.raises(ValueError, match="refusing to train"):
        train_detector([root], synthetic_profile.identity_record(),
                       tmp_path / "detector")


def test_training_refuses_a_sheet_missing_its_raster(
        tmp_path, synthetic_profile):
    root = write_dataset(tmp_path / "clean", synthetic_profile,
                         [build_glyph_sheet(1)])
    (root / "sheets" / "sheet_1.png").unlink()
    with pytest.raises(ValueError, match="raster"):
        train_detector([root], synthetic_profile.identity_record(),
                       tmp_path / "detector")


def test_training_refuses_a_dataset_with_no_symbol_examples(
        tmp_path, synthetic_profile):
    identity = synthetic_profile.identity_record()
    root = tmp_path / "textonly"
    text = {"id": "p1-text0", "sheet": 1, "string": "T-101",
            "text_class": "equipment_tag", "bbox": [0.0, 0.0, 10.0, 10.0],
            "confidence": 1.0,
            "provenance": {"component": "c", "evidence": "e"}}
    LabelStore(root / "labels").record_many(
        identity, 1, [make_example("text", text, "pass")])
    (root / "sheets").mkdir(parents=True)
    (root / "sheets" / "sheet_1.png").write_bytes(
        encode_gray_png(4, 4, bytes(b"\xff" * 16)))
    with pytest.raises(ValueError, match="no symbol examples"):
        train_detector([root], identity, tmp_path / "detector")


def test_training_needs_at_least_one_dataset_root(tmp_path,
                                                  synthetic_profile):
    with pytest.raises(ValueError, match="at least one dataset root"):
        train_detector([], synthetic_profile.identity_record(),
                       tmp_path / "detector")


def test_training_is_deterministic(tmp_path, synthetic_profile):
    root = write_dataset(tmp_path / "clean", synthetic_profile,
                         [build_glyph_sheet(1), build_glyph_sheet(2, 6, 4)])

    first = train_detector([root], synthetic_profile.identity_record(),
                           tmp_path / "a")
    second = train_detector([root], synthetic_profile.identity_record(),
                            tmp_path / "b")

    assert first["version"] == second["version"]
    assert first["classes"] == second["classes"]


# --------------------------------------------------------------------------
# Detection: real pixels in, classified boxes with provenance out

@pytest.fixture
def trained_artifact(tmp_path, synthetic_profile):
    """Trained over layouts at both raster parities, so calibrated
    thresholds price in the fixture renderer's rounding variation."""
    root = write_dataset(tmp_path / "clean", synthetic_profile,
                         [build_glyph_sheet(1), build_glyph_sheet(2, 6, 4),
                          build_glyph_sheet(3, 7, 3)])
    out = tmp_path / "detector"
    train_detector([root], synthetic_profile.identity_record(), out)
    return out


def test_the_trained_detector_finds_the_glyphs(tmp_path, synthetic_profile,
                                               trained_artifact):
    detector = TrainedSymbolDetector(str(trained_artifact))
    held_out = build_glyph_sheet(3, 9.0, 5.0)
    normalized, _ = normalize_sheet(held_out)

    detections = detector.detect(normalized, synthetic_profile)

    truth = list(held_out.annotations.symbols)
    assert len(detections) == len(truth)
    matched = set()
    for sym in truth:
        hits = [d for d in detections
                if d.symbol_class == sym.symbol_class
                and bbox_iou(d.bbox, sym.bbox) >= 0.5
                and d.id not in matched]
        assert hits, f"{sym.symbol_class} at {sym.bbox} not detected"
        matched.add(hits[0].id)
    version = json.loads(
        (trained_artifact / "manifest.json").read_text())["version"]
    for detection in detections:
        assert detection.sheet == 3
        assert 0.0 < detection.confidence <= 1.0
        assert detection.provenance.component == (
            f"symbol_detector:trained@{version}")
        if detection.symbol_class == "flow_arrow":
            # the matched oriented prototype is the direction evidence
            assert detection.direction == pytest.approx((1.0, 0.0))
        else:
            assert detection.direction is None
    assert len({d.id for d in detections}) == len(detections)


def test_detection_refuses_a_rasterless_sheet(synthetic_profile,
                                              trained_artifact):
    detector = TrainedSymbolDetector(str(trained_artifact))
    with pytest.raises(ValueError, match="raster"):
        detector.detect(Sheet(number=1, width=10, height=10),
                        synthetic_profile)


def test_detection_refuses_another_profiles_sheets(trained_artifact):
    detector = TrainedSymbolDetector(str(trained_artifact))
    other = ConventionProfile(name="other-company", version="v1",
                              legend={}, tag_grammar={})
    sheet = build_glyph_sheet(1)
    with pytest.raises(ValueError, match="other-company"):
        detector.detect(sheet, other)


def test_a_tampered_artifact_is_refused(trained_artifact):
    manifest_path = trained_artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["classes"]["tank"][0]["threshold"] = 0.01
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="modified after training"):
        TrainedSymbolDetector(str(trained_artifact))


def test_a_missing_artifact_is_refused_by_name(tmp_path):
    with pytest.raises(ValueError, match="manifest.json"):
        TrainedSymbolDetector(str(tmp_path / "nowhere"))


def test_an_artifact_from_another_operating_scale_is_refused(
        trained_artifact, monkeypatch):
    """Scale is the contract: when the pipeline's normalized scale moves
    (the TARGET_LONG_SIDE revisit the spec plans for the real corpus),
    an artifact trained at the old scale is refused, not loaded to
    detect garbage."""
    monkeypatch.setattr("pidgraph.detector.TARGET_LONG_SIDE", 1600)
    with pytest.raises(ValueError, match="target_long_side=400"):
        TrainedSymbolDetector(str(trained_artifact))


# --------------------------------------------------------------------------
# Selected by configuration: nothing above the seam changes

def test_the_trained_detector_is_selected_by_configuration(
        synthetic_profile, trained_artifact):
    artifacts = digitize(
        Document(name="glyphs.pdf", sheets=(build_glyph_sheet(1),)),
        synthetic_profile,
        config=PipelineConfig(symbol_detector=f"trained:{trained_artifact}"))

    (record,) = artifacts.detection_records
    assert record["symbols"], "the trained detector ran and detected"
    components = {s["provenance"]["component"] for s in record["symbols"]}
    (component,) = components
    assert component.startswith("symbol_detector:trained@")
    assert {s["symbol_class"] for s in record["symbols"]} <= set(
        synthetic_profile.legend)
    assert artifacts.plant_graph["nodes"], "assembly above the seam ran"


def test_bare_trained_selection_names_the_syntax():
    with pytest.raises(ValueError, match=r"trained:<artifact dir>"):
        build_components(PipelineConfig(symbol_detector="trained"))


def test_options_on_a_component_that_takes_none_are_refused():
    with pytest.raises(ValueError, match="takes no"):
        build_components(PipelineConfig(symbol_detector="stub:extra"))


def test_the_stub_stays_the_default(synthetic_document, synthetic_profile):
    artifacts = digitize(synthetic_document, synthetic_profile)
    components = {s["provenance"]["component"]
                  for s in artifacts.detection_records[0]["symbols"]}
    assert components == {"symbol_detector:stub"}


# --------------------------------------------------------------------------
# The acceptance bar: the eval harness gates the trained detector

def degrade_glyph_sheet(sheet, transforms, seed):
    """The Sheet under synthetic degradation, its annotations pushed
    through the same point map (the label-factory variant treatment)."""
    width, height = sheet.width, sheet.height
    raster = degrade_raster(Raster(width, height, sheet.raster),
                            transforms, seed)
    annotations = sheet.annotations
    return Sheet(
        number=sheet.number, width=width, height=height,
        raster=raster.pixels,
        annotations=SheetAnnotations(
            symbols=tuple(replace(s, bbox=tuple(degrade_bbox(
                transforms, s.bbox, width, height)))
                for s in annotations.symbols),
            lines=tuple(replace(l, polyline=tuple(
                tuple(p) for p in degrade_polyline(
                    transforms, l.polyline, width, height)))
                for l in annotations.lines),
            texts=tuple(replace(t, bbox=tuple(degrade_bbox(
                transforms, t.bbox, width, height)))
                for t in annotations.texts)))


def write_eval_dataset(root, profile, sheets):
    """An eval set the way the label factory lays one out — labels,
    connectivity in the s2_pml edge shape, Sheet rasters."""
    detector, recognizer, _ = build_components(PipelineConfig())
    identity = profile.identity_record()
    store = LabelStore(root / "labels")
    for sheet in sheets:
        record, assembly = digitize_sheet(sheet, profile, detector,
                                          recognizer)
        store.record_many(identity, sheet.number, (
            [make_example("symbol", s, "pass")
             for s in record["symbols"]]
            + [make_example("text", t, "pass") for t in record["texts"]]
            + [make_example("line", l, "pass")
               for l in record["lines"]]))
        tag_to_id = {t.node_tag(): t.detection.id
                     for t in assembly.terminals}
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


def test_the_trained_detector_passes_the_symbol_gate(tmp_path,
                                                     synthetic_profile):
    """The ticket's acceptance bar, end to end and offline: train on
    clean + degraded label-factory-shaped data, score held-out layouts
    with the eval harness, and clear symbol F1 >= 0.90 — on the clean
    eval set and on a degraded one.

    Deliberate exception to "the harness never runs in CI": what runs
    here is the harness *machinery* on tiny synthetic Sheets (the
    test_eval_harness precedent) — no real models, no stored weights,
    no network. On-demand gate runs against real eval sets stay outside
    the suite."""
    identity = synthetic_profile.identity_record()
    transforms = (Skew(angle_deg=0.8), Noise(sigma=16.0))
    training_sheets = [build_glyph_sheet(1), build_glyph_sheet(2, 6, 4),
                       build_glyph_sheet(3, 7, 3),
                       build_glyph_sheet(4, 12, 8)]
    clean = write_dataset(tmp_path / "clean", synthetic_profile,
                          training_sheets)
    degraded = write_dataset(
        tmp_path / "degraded", synthetic_profile,
        [degrade_glyph_sheet(sheet, transforms, f"train/{sheet.number}")
         for sheet in training_sheets])
    artifact = tmp_path / "det"
    train_detector([clean, degraded], identity, artifact)
    config = PipelineConfig(symbol_detector=f"trained:{artifact}")

    held_out = [build_glyph_sheet(11, 11, 7), build_glyph_sheet(12, 3, 9)]
    reports = {}
    for name, sheets in (
            ("clean", held_out),
            ("degraded", [degrade_glyph_sheet(s, transforms,
                                              f"eval/{s.number}")
                          for s in held_out])):
        root = write_eval_dataset(tmp_path / f"eval-{name}",
                                  synthetic_profile, sheets)
        eval_sheets = load_eval_set(root, synthetic_profile)
        reports[name] = evaluate(eval_sheets, synthetic_profile, config)

    for name, report in reports.items():
        assert report["coverage"]["failed"] == 0, report["failures"]
        (symbol_gate,) = [g for g in report["gates"]
                          if g["metric"] == "symbol_f1"]
        assert symbol_gate["passed"], (name, report["metrics"]["symbol"])
        assert report["metrics"]["symbol"]["f1"] >= 0.90


# --------------------------------------------------------------------------
# CLI: the ML developer's training entry point, fully local

def test_cli_trains_and_names_the_artifact(tmp_path, synthetic_profile,
                                           capsys):
    root = write_dataset(tmp_path / "clean", synthetic_profile,
                         [build_glyph_sheet(1), build_glyph_sheet(2, 6, 4)])
    out = tmp_path / "det"

    main([str(root), "--name", synthetic_profile.name,
          "--version", synthetic_profile.version, "--out", str(out)])

    manifest = json.loads((out / "manifest.json").read_text())
    printed = capsys.readouterr().out
    assert manifest["version"] in printed
    assert str(out) in printed
    detector = TrainedSymbolDetector(str(out))
    assert detector.version == manifest["version"]
