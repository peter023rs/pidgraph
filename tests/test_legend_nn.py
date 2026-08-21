"""The Legend Dictionary nearest-neighbor classifier (ticket 17): symbol
candidates classified by nearest-neighbor match against the Convention
Profile's Legend Dictionary glyphs, behind the same SymbolDetector seam as
the trained detector, selected by configuration. Below-acceptance matches
surface as the reserved unclassified class — fail-closed, never
force-assigned — and the eval harness scores the classifier beside the
fine-tuned path in one comparison report.

All offline and deterministic: glyph crops are rendered with the same
class-distinct ink the trained-detector suite draws its Sheets with."""

import shutil
from pathlib import Path
from urllib.parse import quote

import pytest

from pidgraph.assemble import assemble_sheet
from pidgraph.detector import LegendNNSymbolDetector, train_detector
from pidgraph.eval_harness import bbox_iou, compare, evaluate, load_eval_set
from pidgraph.model import (
    ConventionProfile,
    Document,
    Provenance,
    Sheet,
    SheetAnnotations,
    SymbolAnnotation,
    SymbolDetection,
    UNCLASSIFIED_SYMBOL,
)
from pidgraph.normalize import normalize_sheet
from pidgraph.pipeline import digitize
from pidgraph.pngio import encode_gray_png
from pidgraph.seams import PipelineConfig, build_components
from test_trained_detector import (
    SHEET_H,
    SHEET_W,
    _draw_symbol,
    _stroke,
    build_glyph_sheet,
    glyph_annotations,
    write_dataset,
    write_eval_dataset,
)

FIXTURE_BUNDLE = (Path(__file__).parent / "fixtures" / "profiles"
                  / "synthetic-test")

# one glyph per Legend Dictionary class, at the size the glyph Sheets
# draw it — no flow_arrow: a static glyph match asserts no orientation
GLYPH_SIZES = {"tank": (40, 60), "nozzle": (8, 8), "gate_valve": (20, 20),
               "instrument_bubble": (20, 20), "opc": (30, 20)}

GLYPH_MARGIN = 2  # the pinned paper ring the classifier trims back off


def write_glyph(glyphs_dir: Path, symbol_class: str, width: int,
                height: int, draw_class: "str | None" = None) -> Path:
    """One Legend Dictionary glyph crop: the symbol's ink inside the
    pinned 2 px paper ring, the way a hand-crop from a normalized Legend
    Sheet carries it."""
    canvas_w = width + 2 * GLYPH_MARGIN
    canvas_h = height + 2 * GLYPH_MARGIN
    grid = bytearray(b"\xff" * (canvas_w * canvas_h))
    if draw_class != "":  # "" writes a blank (all-paper) glyph
        _draw_symbol(grid, canvas_w, canvas_h, SymbolAnnotation(
            draw_class or symbol_class,
            (GLYPH_MARGIN, GLYPH_MARGIN,
             GLYPH_MARGIN + width, GLYPH_MARGIN + height)))
    glyphs_dir.mkdir(parents=True, exist_ok=True)
    path = glyphs_dir / f"{quote(symbol_class, safe='')}.png"
    path.write_bytes(encode_gray_png(canvas_w, canvas_h, bytes(grid)))
    return path


@pytest.fixture
def nn_bundle(tmp_path):
    """The checked-in fixture profile bundle plus a glyphs/ directory —
    the Convention Profile the classifier is selected with."""
    bundle = shutil.copytree(FIXTURE_BUNDLE, tmp_path / "bundle")
    for symbol_class, (width, height) in GLYPH_SIZES.items():
        write_glyph(bundle / "glyphs", symbol_class, width, height)
    return bundle


def build_nn_sheet(number: int, dx: float = 0.0, dy: float = 0.0) -> Sheet:
    """The glyph-Sheet fixture minus its flow arrow — the one class the
    classifier carries no glyph for."""
    full = glyph_annotations(dx, dy)
    annotations = SheetAnnotations(
        symbols=tuple(s for s in full.symbols
                      if s.symbol_class != "flow_arrow"),
        lines=full.lines, texts=full.texts)
    grid = bytearray(b"\xff" * (SHEET_W * SHEET_H))
    for line in annotations.lines:
        for p, q in zip(line.polyline, line.polyline[1:]):
            _stroke(grid, SHEET_W, SHEET_H, p, q)
    for sym in annotations.symbols:
        _draw_symbol(grid, SHEET_W, SHEET_H, sym)
    return Sheet(number=number, width=SHEET_W, height=SHEET_H,
                 raster=bytes(grid), annotations=annotations)


def shape_sheet(symbol_class: str, bbox: tuple, number: int = 1) -> Sheet:
    """One lone shape on an otherwise blank operating-scale Sheet."""
    grid = bytearray(b"\xff" * (SHEET_W * SHEET_H))
    _draw_symbol(grid, SHEET_W, SHEET_H,
                 SymbolAnnotation(symbol_class, bbox))
    return Sheet(number=number, width=SHEET_W, height=SHEET_H,
                 raster=bytes(grid))


# --------------------------------------------------------------------------
# Loading: the profile bundle's glyphs, validated fail-closed

def test_bare_legend_nn_selection_names_the_syntax():
    with pytest.raises(ValueError, match=r"legend-nn:<profile bundle dir>"):
        build_components(PipelineConfig(symbol_detector="legend-nn"))


def test_a_bundle_without_glyphs_is_refused(tmp_path):
    bundle = shutil.copytree(FIXTURE_BUNDLE, tmp_path / "bundle")
    with pytest.raises(ValueError, match="no Legend Dictionary glyphs"):
        LegendNNSymbolDetector(str(bundle))


def test_a_glyph_for_a_class_outside_the_legend_is_refused(nn_bundle):
    write_glyph(nn_bundle / "glyphs", "mystery_widget", 10, 10,
                draw_class="tank")
    with pytest.raises(ValueError, match="mystery_widget"):
        LegendNNSymbolDetector(str(nn_bundle))


def test_a_flow_arrow_glyph_is_refused(nn_bundle):
    """A nearest-neighbor glyph match asserts no orientation, and a
    direction is never guessed — an arrow glyph is refused loudly rather
    than matched into direction-less FlowArrow detections."""
    write_glyph(nn_bundle / "glyphs", "flow_arrow", 10, 10)
    with pytest.raises(ValueError, match="orientation"):
        LegendNNSymbolDetector(str(nn_bundle))


def test_a_uniform_glyph_is_refused(nn_bundle):
    write_glyph(nn_bundle / "glyphs", "opc", 30, 20, draw_class="")
    with pytest.raises(ValueError, match="uniform"):
        LegendNNSymbolDetector(str(nn_bundle))


def test_the_version_is_the_glyph_content(tmp_path, nn_bundle):
    """Two bundles with the same glyphs share a version; a changed glyph
    changes it — provenance distinguishes what was matched against."""
    same = LegendNNSymbolDetector(str(nn_bundle)).version
    other = shutil.copytree(FIXTURE_BUNDLE, tmp_path / "other")
    for symbol_class, (width, height) in GLYPH_SIZES.items():
        write_glyph(other / "glyphs", symbol_class, width, height)
    assert LegendNNSymbolDetector(str(other)).version == same
    write_glyph(other / "glyphs", "gate_valve", 24, 24)
    assert LegendNNSymbolDetector(str(other)).version != same


# --------------------------------------------------------------------------
# Classification: candidates assigned their nearest Legend Dictionary entry

def test_synthetic_glyphs_classify_by_nearest_neighbor(nn_bundle,
                                                       synthetic_profile):
    detector = LegendNNSymbolDetector(str(nn_bundle))
    sheet = build_nn_sheet(3, 9.0, 5.0)
    normalized, _ = normalize_sheet(sheet)

    detections = detector.detect(normalized, synthetic_profile)

    truth = list(sheet.annotations.symbols)
    assert len(detections) == len(truth)
    matched = set()
    for sym in truth:
        hits = [d for d in detections
                if d.symbol_class == sym.symbol_class
                and bbox_iou(d.bbox, sym.bbox) >= 0.5
                and d.id not in matched]
        assert hits, f"{sym.symbol_class} at {sym.bbox} not classified"
        matched.add(hits[0].id)
    for detection in detections:
        assert 0.0 < detection.confidence <= 1.0
        assert detection.provenance.component == detector.component
        assert detection.symbol_class in detection.provenance.evidence
        assert "distance" in detection.provenance.evidence
        assert detection.direction is None
    assert len({d.id for d in detections}) == len(detections)


def test_detection_refuses_a_rasterless_sheet(nn_bundle,
                                              synthetic_profile):
    detector = LegendNNSymbolDetector(str(nn_bundle))
    with pytest.raises(ValueError, match="raster"):
        detector.detect(Sheet(number=1, width=10, height=10),
                        synthetic_profile)


def test_detection_refuses_another_profiles_sheets(nn_bundle):
    detector = LegendNNSymbolDetector(str(nn_bundle))
    other = ConventionProfile(name="other-company", version="v1",
                              legend={}, tag_grammar={})
    with pytest.raises(ValueError, match="other-company"):
        detector.detect(build_nn_sheet(1), other)


# --------------------------------------------------------------------------
# Fail-closed: a below-acceptance nearest neighbor is surfaced, not assigned

@pytest.fixture
def tank_only_bundle(tmp_path):
    bundle = shutil.copytree(FIXTURE_BUNDLE, tmp_path / "tank-only")
    write_glyph(bundle / "glyphs", "tank", 20, 20)
    return bundle


def test_a_below_acceptance_match_surfaces_unclassified(tank_only_bundle,
                                                        synthetic_profile):
    """A squashed tank shares the glyph's top and bottom runs but not its
    verticals: near enough to be a candidate, too far to assign — it
    surfaces as the reserved unclassified class with the nearest entry
    named in its provenance."""
    detector = LegendNNSymbolDetector(str(tank_only_bundle))
    sheet = shape_sheet("tank", (100.0, 100.0, 120.0, 108.0))
    normalized, _ = normalize_sheet(sheet)

    (detection,) = detector.detect(normalized, synthetic_profile)

    assert detection.symbol_class == UNCLASSIFIED_SYMBOL
    assert detection.confidence < 0.6
    assert "'tank'" in detection.provenance.evidence
    assert "unclassified" in detection.provenance.evidence


def test_ink_far_from_every_glyph_is_not_a_candidate(tank_only_bundle,
                                                     synthetic_profile):
    """A diamond shares almost no ink with the square tank glyph — under
    the candidate floor, it is not a symbol report at all."""
    detector = LegendNNSymbolDetector(str(tank_only_bundle))
    sheet = shape_sheet("opc", (100.0, 100.0, 120.0, 120.0))
    normalized, _ = normalize_sheet(sheet)

    assert detector.detect(normalized, synthetic_profile) == []


def test_an_exact_match_is_assigned_its_entry(tank_only_bundle,
                                              synthetic_profile):
    detector = LegendNNSymbolDetector(str(tank_only_bundle))
    sheet = shape_sheet("tank", (100.0, 100.0, 120.0, 120.0))
    normalized, _ = normalize_sheet(sheet)

    (detection,) = detector.detect(normalized, synthetic_profile)

    assert detection.symbol_class == "tank"
    assert detection.confidence >= 0.6
    assert bbox_iou(detection.bbox, (100.0, 100.0, 120.0, 120.0)) >= 0.5


def test_an_unclassified_detection_never_becomes_a_plant_item(
        synthetic_profile):
    unclassified = SymbolDetection(
        id="p1-sym0", sheet=1, symbol_class=UNCLASSIFIED_SYMBOL,
        bbox=(0.0, 0.0, 10.0, 10.0), confidence=0.4,
        provenance=Provenance(component="symbol_detector:legend-nn@0",
                              evidence="nearest entry too far"))
    assembly = assemble_sheet(Sheet(number=1, width=50, height=50),
                              [unclassified], [], [], synthetic_profile)
    assert assembly.terminals == []
    assert assembly.arrows == []


# --------------------------------------------------------------------------
# Selected by configuration: nothing above the seam changes

def test_the_classifier_is_selected_by_configuration(synthetic_profile,
                                                     nn_bundle):
    artifacts = digitize(
        Document(name="glyphs.pdf", sheets=(build_nn_sheet(1),)),
        synthetic_profile,
        config=PipelineConfig(symbol_detector=f"legend-nn:{nn_bundle}"))

    (record,) = artifacts.detection_records
    assert record["symbols"], "the classifier ran and classified"
    components = {s["provenance"]["component"] for s in record["symbols"]}
    (component,) = components
    assert component.startswith("symbol_detector:legend-nn@")
    assert {s["symbol_class"] for s in record["symbols"]} <= (
        set(synthetic_profile.legend) | {UNCLASSIFIED_SYMBOL})
    assert artifacts.plant_graph["nodes"], "assembly above the seam ran"


# --------------------------------------------------------------------------
# The harness scores the classifier; the side-by-side report includes it

def test_the_harness_scores_the_classifier(tmp_path, synthetic_profile,
                                           nn_bundle):
    root = write_eval_dataset(tmp_path / "eval", synthetic_profile,
                              [build_nn_sheet(11, 11, 7),
                               build_nn_sheet(12, 3, 9)])
    eval_sheets = load_eval_set(root, synthetic_profile)

    report = evaluate(eval_sheets, synthetic_profile,
                      PipelineConfig(symbol_detector=f"legend-nn:{nn_bundle}"))

    assert report["coverage"]["failed"] == 0, report["failures"]
    assert report["metrics"]["symbol"]["f1"] >= 0.90


def test_the_side_by_side_report_includes_the_classifier(
        tmp_path, synthetic_profile, nn_bundle):
    """The comparison the spec asks the harness for: per-company
    fine-tuning beside Legend Dictionary nearest-neighbor classification,
    on the same eval set — arrows included, which the classifier honestly
    misses (no glyph asserts orientation) rather than crashing on."""
    identity = synthetic_profile.identity_record()
    clean = write_dataset(tmp_path / "clean", synthetic_profile,
                          [build_glyph_sheet(1), build_glyph_sheet(2, 6, 4),
                           build_glyph_sheet(3, 7, 3)])
    artifact = tmp_path / "det"
    train_detector([clean], identity, artifact)
    root = write_eval_dataset(tmp_path / "eval", synthetic_profile,
                              [build_glyph_sheet(11, 11, 7)])
    eval_sheets = load_eval_set(root, synthetic_profile)

    report = compare(eval_sheets, synthetic_profile, {
        "trained": PipelineConfig(symbol_detector=f"trained:{artifact}"),
        "legend-nn": PipelineConfig(
            symbol_detector=f"legend-nn:{nn_bundle}")})

    by_name = {c["name"]: c for c in report["configurations"]}
    assert set(by_name) == {"trained", "legend-nn"}
    for name, config_report in by_name.items():
        assert config_report["coverage"]["failed"] == 0, (
            name, config_report["failures"])
        assert config_report["metrics"]["symbol"]["f1"] is not None
    # the arrow the classifier carries no glyph for stays a miss —
    # priced into its score, never guessed and never a crash
    assert by_name["legend-nn"]["metrics"]["symbol"]["fn"] >= 1
