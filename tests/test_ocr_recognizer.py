"""OCR engines behind the TextRecognizer seam (ticket 18).

A real engine reads pixels without knowing which tag class it is looking
at, so the adapter around it assigns each raw read its tag-grammar
class, stamps engine + version provenance, and hands the raw string to
the lexicon-constrained decoder unchanged. All of that is proven here
with an in-test fake backend — the suite never needs an OCR engine, a
model, or the network; the real engines are scored on demand by the
eval harness."""

import json
import socket
import sys
from pathlib import Path

import pytest
from conftest import TEXTS, build_synthetic_sheet
from test_trained_detector import write_eval_dataset

from pidgraph.eval_harness import compare, load_eval_set
from pidgraph.eval_harness import main as harness_main
from pidgraph.lexicon import classify_candidate
from pidgraph.model import (
    UNCLASSIFIED_TEXT,
    Sheet,
    SheetAnnotations,
)
from pidgraph.ocr import (
    ENGINES,
    EngineTextRecognizer,
    OfflineViolation,
    RawRead,
    engine_options,
)
from pidgraph.pipeline import digitize
from pidgraph.seams import TEXT_RECOGNIZERS, PipelineConfig, build_components

BUNDLE = str(Path(__file__).parent / "fixtures" / "profiles"
             / "synthetic-test")

GRAMMAR = {
    "equipment_tag": r"[A-Z]-\d{3}",
    "instrument_tag": r"[A-Z]{2}-\d{3}",
    "line_number": r"\d{2,3}-[A-Z]{2}-\d{3}",
    "equipment_label": r"[一-鿿]{2,6}",
}


# --------------------------------------------------------------------------
# Class assignment: which grammar a raw read belongs to

def test_a_read_matching_exactly_one_grammar_is_that_class():
    assert classify_candidate("PI-100", GRAMMAR) == "instrument_tag"
    assert classify_candidate("原料罐", GRAMMAR) == "equipment_label"


def test_a_smudged_read_repairable_into_exactly_one_class_is_that_class():
    # the decoder's own confusion set (O/0, S/5, ...) decides reachability;
    # the string itself is left for the decoder to repair
    assert classify_candidate("PI-1OO", GRAMMAR) == "instrument_tag"
    assert classify_candidate("15O-GA-OO1", GRAMMAR) == "line_number"


def test_a_read_fitting_no_grammar_is_unclassified():
    assert classify_candidate("冷却水 CW", GRAMMAR) is None
    assert classify_candidate("", GRAMMAR) is None


def test_a_read_fitting_several_classes_is_unclassified():
    # fail-closed: the adapter never guesses between grammars
    overlapping = {"a": r"[A-Z]-\d{3}", "b": r"[A-Z]-\d{3}"}
    assert classify_candidate("T-101", overlapping) is None
    # ...nor between repairs into different classes
    repairs = {"letters": r"[A-Z]{5}", "digits": r"\d{5}"}
    assert classify_candidate("OS8IZ", repairs) is None


def test_an_exact_class_beats_repairs_into_others():
    grammar = {"code": r"[A-Z0-9]{5}", "digits": r"\d{5}"}
    assert classify_candidate("1O2S3", grammar) == "code"


def test_a_read_too_degraded_to_enumerate_is_unclassified():
    assert classify_candidate("O" * 40, {"zeros": r"0{40}"}) is None


# --------------------------------------------------------------------------
# The adapter around an engine: raw reads in, TextDetections out


class FakeBackend:
    """An engine that answers with canned reads — the adapter's contract
    is what is under test, not any engine."""
    name = "fake"
    version = "v1"
    rotations = (0,)
    scale = 1

    def __init__(self, reads):
        self.reads = list(reads)
        self.calls = []

    def read(self, width, height, pixels):
        self.calls.append((width, height, pixels))
        return list(self.reads)


def _bbox_of(string: str):
    return next(t.bbox for t in TEXTS if t.string == string)


@pytest.fixture
def fixture_reads():
    # the fixture Sheet's tags as an engine would read them: one clean,
    # one smudged (0 read as O), plus a Chinese label no grammar covers
    return [
        RawRead(_bbox_of("PI-100"), "PI-100", 0.98, "quad a"),
        RawRead(_bbox_of("T-101"), "T-1O1", 0.91, "quad b"),
        RawRead((100.0, 150.0, 140.0, 162.0), "冷却水", 0.99, "quad c"),
    ]


def test_reads_become_classified_detections_with_engine_provenance(
        synthetic_profile, fixture_reads):
    recognizer = EngineTextRecognizer(FakeBackend(fixture_reads))
    sheet = build_synthetic_sheet(1)

    detections = recognizer.recognize(sheet, synthetic_profile)

    by_string = {d.string: d for d in detections}
    assert set(by_string) == {"PI-100", "T-1O1", "冷却水"}  # raw, untouched
    assert by_string["PI-100"].text_class == "instrument_tag"
    assert by_string["T-1O1"].text_class == "equipment_tag"
    assert by_string["冷却水"].text_class == UNCLASSIFIED_TEXT
    assert by_string["T-1O1"].bbox == _bbox_of("T-101")
    assert by_string["PI-100"].confidence == pytest.approx(0.98)
    for detection in detections:
        assert detection.sheet == 1
        assert detection.resolved is False      # only the decoder grants trust
        assert detection.provenance.component == "text_recognizer:fake@v1"
    assert [d.id for d in detections] == ["p1-text0", "p1-text1", "p1-text2"]
    smudged = by_string["T-1O1"].provenance.evidence
    assert "'T-1O1'" in smudged and "0.91" in smudged and "quad b" in smudged
    assert "no tag-grammar class" in by_string["冷却水"].provenance.evidence


def test_the_engine_sees_the_sheet_raster(synthetic_profile, fixture_reads):
    backend = FakeBackend(fixture_reads)
    sheet = build_synthetic_sheet(1)

    EngineTextRecognizer(backend).recognize(sheet, synthetic_profile)

    [(width, height, pixels)] = backend.calls
    assert (width, height) == (sheet.width, sheet.height)
    assert pixels == sheet.raster


def test_a_rasterless_sheet_is_refused(synthetic_profile):
    recognizer = EngineTextRecognizer(FakeBackend([]))
    sheet = Sheet(number=3, width=10, height=10, raster=None,
                  annotations=SheetAnnotations())

    with pytest.raises(ValueError, match="text_recognizer:fake@v1.*Sheet 3"):
        recognizer.recognize(sheet, synthetic_profile)


def test_raw_reads_feed_the_decoder_unchanged_end_to_end(
        synthetic_document, synthetic_profile, fixture_reads):
    """The whole point of the seam: the adapter hands the decoder exactly
    what the engine read, and the decoder's correction shows in the DEXPI
    plant model — nothing above the seam changed."""
    TEXT_RECOGNIZERS["fake"] = lambda: EngineTextRecognizer(
        FakeBackend(fixture_reads))
    try:
        artifacts = digitize(synthetic_document, synthetic_profile,
                             config=PipelineConfig(text_recognizer="fake"))
    finally:
        del TEXT_RECOGNIZERS["fake"]

    [record] = artifacts.detection_records
    texts = {t["string"]: t for t in record["texts"]}
    assert texts["T-101"]["raw_string"] == "T-1O1"
    assert texts["T-101"]["correction"] == "O->0 at index 3"
    assert texts["T-101"]["resolved"] is True
    assert texts["冷却水"]["resolved"] is False
    assert texts["冷却水"]["text_class"] == UNCLASSIFIED_TEXT
    equipment = artifacts.plant_model["conceptualModel"]["Equipment"]
    assert "T-101" in {e["tagName"] for e in equipment}


# --------------------------------------------------------------------------
# Rotated and vertical text: the adapter sweeps rotations for engines that
# only read upright, maps boxes back, and keeps the best read per region

def _ink_bbox(width, height, pixels):
    xs = [i % width for i, v in enumerate(pixels) if v == 0]
    ys = [i // width for i, v in enumerate(pixels) if v == 0]
    if not xs:
        return None
    return (float(min(xs)), float(min(ys)),
            float(max(xs) + 1), float(max(ys) + 1))


class UprightOnlyBackend:
    """Reads a bar of ink only when it lies horizontally — a stand-in
    for an engine that cannot read vertical text itself. Upright-wide
    ink reads as 'V-101'; the same bar seen upside down reads worse."""
    name = "upright"
    version = "v1"
    rotations = (0, 90, 270)
    scale = 1

    def __init__(self):
        self.calls = []

    def read(self, width, height, pixels):
        self.calls.append((width, height))
        box = _ink_bbox(width, height, pixels)
        if box is None or (box[2] - box[0]) <= (box[3] - box[1]):
            return []
        # a bar's ink is symmetric, so tell the two readable rotations
        # apart by where the bar lies: the fixture bar sits on the right
        # of the Sheet, which a 90° clockwise turn carries to the right
        # half of the turned frame (upright) and 270° to the left half
        if box[0] >= width / 2:
            return [RawRead(box, "V-101", 0.9, "upright")]
        return [RawRead(box, "101-A", 0.6, "upside down")]


def _bar_sheet(bbox, width=400, height=200):
    x0, y0, x1, y1 = bbox
    grid = bytearray(b"\xff" * (width * height))
    for y in range(int(y0), int(y1)):
        for x in range(int(x0), int(x1)):
            grid[y * width + x] = 0
    return Sheet(number=7, width=width, height=height, raster=bytes(grid),
                 annotations=SheetAnnotations())


def test_vertical_text_is_read_through_a_rotation_and_mapped_back(
        synthetic_profile):
    vertical_bar = (300.0, 40.0, 312.0, 100.0)   # taller than wide
    backend = UprightOnlyBackend()

    detections = EngineTextRecognizer(backend).recognize(
        _bar_sheet(vertical_bar), synthetic_profile)

    assert sorted(backend.calls) == [(200, 400), (200, 400), (400, 200)]
    [detection] = detections                     # merged: one region, one read
    assert detection.string == "V-101"           # the better of the two
    assert detection.confidence == pytest.approx(0.9)
    assert detection.bbox == vertical_bar        # back in Sheet coordinates
    assert "rotated 90" in detection.provenance.evidence
    # the read it set aside is not lost to review
    assert "also read as '101-A' at 0.60 rotated 270°" in \
        detection.provenance.evidence


def test_horizontal_text_needs_no_rotation(synthetic_profile):
    horizontal_bar = (240.0, 50.0, 300.0, 62.0)   # right half: reads upright at 0°

    [detection] = EngineTextRecognizer(UprightOnlyBackend()).recognize(
        _bar_sheet(horizontal_bar), synthetic_profile)

    assert detection.string == "V-101"
    assert detection.bbox == horizontal_bar
    assert "rotated" not in detection.provenance.evidence


class ScaledBackend(FakeBackend):
    """An engine that wants the raster upscaled before it reads."""
    name = "scaled"
    scale = 2


def test_an_upscaled_read_is_mapped_back_to_sheet_coordinates(
        synthetic_profile):
    backend = ScaledBackend([RawRead((40.0, 120.0, 120.0, 144.0), "T-101",
                                     0.8)])
    sheet = build_synthetic_sheet(1)

    [detection] = EngineTextRecognizer(backend).recognize(
        sheet, synthetic_profile)

    [(width, height, pixels)] = backend.calls
    assert (width, height) == (2 * sheet.width, 2 * sheet.height)
    assert len(pixels) == 4 * len(sheet.raster)
    assert detection.bbox == (20.0, 60.0, 60.0, 72.0)


# --------------------------------------------------------------------------
# Fully local: no network endpoint is called at inference time


class PhoningHomeBackend(FakeBackend):
    """An engine that would fetch something while reading — the behavior
    a model auto-download produces on a machine that never pre-fetched."""
    name = "phoning"

    def __init__(self, how):
        super().__init__([])
        self.how = how

    def read(self, width, height, pixels):
        if self.how == "create_connection":
            socket.create_connection(("127.0.0.1", 9), timeout=0.2)
        else:
            with socket.socket() as sock:
                sock.settimeout(0.2)
                sock.connect(("127.0.0.1", 9))
        return []


@pytest.mark.parametrize("how", ["create_connection", "socket.connect"])
def test_an_engine_reaching_for_the_network_is_refused(synthetic_profile,
                                                       how):
    recognizer = EngineTextRecognizer(PhoningHomeBackend(how))

    with pytest.raises(OfflineViolation, match="text_recognizer:phoning@v1"):
        recognizer.recognize(build_synthetic_sheet(1), synthetic_profile)


def test_the_guard_is_lifted_after_recognition(synthetic_profile):
    before = (socket.socket.connect, socket.socket.connect_ex,
              socket.create_connection)
    recognizer = EngineTextRecognizer(PhoningHomeBackend("socket.connect"))
    with pytest.raises(OfflineViolation):
        recognizer.recognize(build_synthetic_sheet(1), synthetic_profile)

    assert (socket.socket.connect, socket.socket.connect_ex,
            socket.create_connection) == before
    # ...and an ordinary recognition leaves it in place too
    EngineTextRecognizer(FakeBackend([])).recognize(build_synthetic_sheet(1),
                                                    synthetic_profile)
    assert (socket.socket.connect, socket.socket.connect_ex,
            socket.create_connection) == before


# --------------------------------------------------------------------------
# Selected by configuration: the candidates are registry names, the stub
# stays the default, and a missing engine names what to install


def test_the_stub_stays_the_default():
    assert PipelineConfig().text_recognizer == "stub"
    _, recognizer, _ = build_components(PipelineConfig())
    assert recognizer.COMPONENT == "text_recognizer:stub"


def test_every_candidate_engine_is_selectable_by_name():
    assert set(ENGINES) == {"rapidocr", "tesseract", "apple-vision",
                            "easyocr"}
    assert set(ENGINES) <= set(TEXT_RECOGNIZERS)


@pytest.mark.parametrize("engine, module, extra", [
    ("rapidocr", "rapidocr", "ocr"),
    ("tesseract", "pytesseract", "ocr-candidates"),
    ("apple-vision", "Vision", "ocr-candidates"),
    ("easyocr", "easyocr", "ocr-candidates"),
])
def test_a_missing_engine_package_names_the_extra_to_install(
        monkeypatch, engine, module, extra):
    monkeypatch.setitem(sys.modules, module, None)   # "not installed"

    with pytest.raises(ModuleNotFoundError,
                       match=rf"pidgraph\[{extra}\]"):
        build_components(PipelineConfig(text_recognizer=engine))


def test_engine_options_select_the_sweep_and_the_upscaling():
    assert engine_options("") == {}
    assert engine_options("scale=3") == {"scale": 3}
    assert engine_options("rotations=0/90/270,scale=2") == {
        "rotations": (0, 90, 270), "scale": 2}


@pytest.mark.parametrize("options", [
    "scale", "scale=", "scale=two", "scale=0", "rotations=45",
    "rotations=", "model=big", "scale=2,scale=3"])
def test_malformed_engine_options_are_refused_by_name(options):
    with pytest.raises(ValueError, match="scale|rotations"):
        engine_options(options)


class EqualConfidenceBackend:
    """Coarse-confidence engines (Apple Vision scores 0.3/0.5/1.0) tie
    across the sweep; geometry must break the tie."""
    name = "coarse"
    version = "v1"
    rotations = (0, 90, 270)
    scale = 1

    def read(self, width, height, pixels):
        box = _ink_bbox(width, height, pixels)
        if box is None:
            return []
        wide = (box[2] - box[0]) > (box[3] - box[1])
        return [RawRead(box, "V-101" if wide else "sOz-L7", 0.5,
                        "upright" if wide else "sideways")]


def test_a_tall_region_read_at_equal_confidence_prefers_the_turned_read(
        synthetic_profile):
    vertical_bar = (300.0, 40.0, 312.0, 100.0)

    [detection] = EngineTextRecognizer(EqualConfidenceBackend()).recognize(
        _bar_sheet(vertical_bar), synthetic_profile)

    assert detection.string == "V-101"
    assert detection.bbox == vertical_bar


# --------------------------------------------------------------------------
# Judged by the harness: engines score side by side on tag exact-match,
# and the report names the component versions that produced the scores


def test_the_harness_scores_an_engine_beside_the_stub(tmp_path,
                                                      synthetic_profile,
                                                      fixture_reads):
    root = write_eval_dataset(tmp_path / "eval", synthetic_profile,
                              [build_synthetic_sheet(1)])
    eval_sheets = load_eval_set(root, synthetic_profile)
    TEXT_RECOGNIZERS["fake"] = lambda: EngineTextRecognizer(
        FakeBackend(fixture_reads))
    try:
        report = compare(eval_sheets, synthetic_profile, {
            "stub": PipelineConfig(),
            "engine": PipelineConfig(text_recognizer="fake")})
    finally:
        del TEXT_RECOGNIZERS["fake"]

    stub, engine = report["configurations"]
    assert stub["metrics"]["tag"]["exact_match"] == 1.0
    # the fake read two of the six tags (one after the decoder's repair)
    assert engine["metrics"]["tag"] == {"truth": 6, "exact": 2,
                                        "exact_match": pytest.approx(2 / 6)}
    assert engine["components"] == {
        "symbol_detector": "symbol_detector:stub",
        "text_recognizer": "text_recognizer:fake@v1"}
    assert stub["components"]["text_recognizer"] == "text_recognizer:stub"
    assert engine["seconds"] >= 0.0


# --------------------------------------------------------------------------
# Through the harness CLI: an engine selection with several options rides
# in one --config string — commas inside an implementation's options are
# the implementation's, not the configuration grammar's

class OptionRecordingBackend(FakeBackend):
    name = "recording"


def test_cli_passes_multi_option_engine_selections_through(
        tmp_path, synthetic_profile, fixture_reads, monkeypatch, capsys):
    seen = {}

    class Recording(EngineTextRecognizer):
        def __init__(self, rotations=None, scale=None):
            super().__init__(OptionRecordingBackend(fixture_reads),
                             rotations=rotations, scale=scale)
            seen.update({"rotations": self.rotations, "scale": self.scale})

        @classmethod
        def from_options(cls, options):
            return cls(**engine_options(options))

    monkeypatch.setitem(TEXT_RECOGNIZERS, "recording", Recording)
    root = write_eval_dataset(tmp_path / "eval", synthetic_profile,
                              [build_synthetic_sheet(1)])
    out = tmp_path / "report.json"
    with pytest.raises(SystemExit):   # the fake engine fails the tag gate
        harness_main([str(root), BUNDLE, "--out", str(out), "--config",
                      "rec:text_recognizer=recording:scale=2,rotations=0/90"
                      ",symbol_detector=stub"])

    assert seen == {"rotations": (0, 90), "scale": 2}
    report = json.loads(out.read_text())
    assert report["config"] == {
        "symbol_detector": "stub",
        "text_recognizer": "recording:scale=2,rotations=0/90",
        "graph_store": "cypher-script"}
