"""Label factory rendering edge (ticket 13): real PyMuPDF rasterization
of a synthetic vector fixture with exactly known geometry, proving the
projection is geometrically faithful — a projected box surrounds its
rendered symbol at the chosen render resolution.

Needs the labelfactory extra (PyMuPDF); skipped where it is absent. No
real drawings are touched: the vector fixture is drawn programmatically
into a pytest tmp dir, keeping the offline test invariant.
"""

import json
import zlib

import pytest

pytest.importorskip("pymupdf")

from pidgraph.label_factory import run_label_factory
from pidgraph.labels import profile_key
from pdf_fixtures import write_pdf

# The fixture page, 400 x 200 pt. PDF content streams use a bottom-left
# origin (y up); the artifacts and PyMuPDF use top-left (y down), so the
# rectangle stroked at PDF y 40..90 sits at artifact y 110..160.
PAGE_W, PAGE_H = 400, 200
OPS = (b"1 w 30 40 100 50 re S "          # the symbol: a stroked rect
       # a valve glyph (bowtie in a 12 pt square) centered at PDF (300,
       # 65), which is artifact (300, 135): its box must be synthesized
       b"294 59 m 306 71 l 294 71 m 306 59 l "
       b"294 59 m 294 71 l 306 59 m 306 71 l S "
       b"BT /F1 12 Tf 250 60 Td (T-101) Tj ET")  # the equipment tag text

# ink hull of the stroked rect in artifact space: the 1 pt line straddles
# the path, so the hull is the rect plus half the stroke; the artifact
# bbox pads a further 0.7 pt the way a detector's hull would
RECT = (30.0, 110.0, 130.0, 160.0)
PAD = 0.5 + 0.7
BBOX = (RECT[0] - PAD, RECT[1] - PAD, RECT[2] + PAD, RECT[3] + PAD)

DPI = 144  # scale 2.0: page renders at 800 x 400 px
SCALE = DPI / 72.0

ARTIFACT = {
    "page": 1,
    "sheet_prefix": "2401-",
    "nodes": [
        {"id": 1, "kind": "equipment", "xy": [80.0, 135.0],
         "tag": "T-101", "shape": "capsule", "bbox": list(BBOX)},
        {"id": 2, "kind": "valve", "xy": [300.0, 135.0],
         "ball": False, "subclass": "gate"},
    ],
    "edges": [],
    "connectors": [],
    "direction_stats": {"pipe_runs": 0, "directed": 0, "conflicts": 0},
}

PROFILE = {"name": "synthetic-vector", "version": "1"}


def decode_gray_png(png: bytes) -> tuple[int, int, bytes]:
    """Decode the factory's own PNG form (8-bit gray, filter 0 rows)."""
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    width = int.from_bytes(png[16:20], "big")
    height = int.from_bytes(png[20:24], "big")
    data, offset = b"", 8
    while offset < len(png):
        length = int.from_bytes(png[offset:offset + 4], "big")
        tag = png[offset + 4:offset + 8]
        if tag == b"IDAT":
            data += png[offset + 8:offset + 8 + length]
        offset += 12 + length
    raw = zlib.decompress(data)
    rows = [raw[y * (width + 1) + 1:(y + 1) * (width + 1)]
            for y in range(height)]
    assert all(raw[y * (width + 1)] == 0 for y in range(height))
    return width, height, b"".join(rows)


@pytest.fixture(scope="module")
def factory_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("labelfactory")
    pdf = write_pdf(tmp / "synthetic.pdf",
                    [("vector_ops", PAGE_W, PAGE_H, OPS)])
    artifact_dir = tmp / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "topology_page1.json").write_text(
        json.dumps(ARTIFACT), encoding="utf-8")
    out = tmp / "out"
    report = run_label_factory(artifact_dir, pdf, out, profile=PROFILE,
                               dpi=DPI, progress=lambda outcome: None)
    return report, out


def ink_pixels(raster, width, window):
    x0, y0, x1, y1 = window
    return [(x, y)
            for y in range(max(0, int(y0)), int(y1) + 1)
            for x in range(max(0, int(x0)), int(x1) + 1)
            if raster[y * width + x] < 100]


def test_render_rasterizes_the_page_at_the_chosen_resolution(factory_run):
    report, out = factory_run
    assert report["coverage"] == {"total_artifacts": 1, "succeeded": 1,
                                  "failed": 0}
    width, height, raster = decode_gray_png(
        (out / "sheets" / "sheet_1.png").read_bytes())
    assert (width, height) == (PAGE_W * 2, PAGE_H * 2)
    assert min(raster) < 100  # the page genuinely has ink


def test_projected_box_surrounds_the_rendered_symbol(factory_run):
    report, out = factory_run
    width, _, raster = decode_gray_png(
        (out / "sheets" / "sheet_1.png").read_bytes())
    labels = json.loads(
        (out / "labels" / profile_key(PROFILE) / "sheet_1.json")
        .read_text(encoding="utf-8"))
    box = labels["examples"]["p1-node1"]["detection"]["bbox"]
    assert box == [v * SCALE for v in BBOX]

    # every ink pixel near the symbol falls inside the projected box,
    # and the box is not vacuously empty
    window = (box[0] - 15, box[1] - 15, box[2] + 15, box[3] + 15)
    ink = ink_pixels(raster, width, window)
    assert ink, "the rendered symbol left no ink where projected"
    assert all(box[0] <= x <= box[2] and box[1] <= y <= box[3]
               for x, y in ink)


def test_synthesized_box_surrounds_the_rendered_valve_glyph(factory_run):
    """A center-only symbol's box is synthesized from convention
    constants — it too must surround its rendered ink at the chosen
    resolution."""
    report, out = factory_run
    width, _, raster = decode_gray_png(
        (out / "sheets" / "sheet_1.png").read_bytes())
    labels = json.loads(
        (out / "labels" / profile_key(PROFILE) / "sheet_1.json")
        .read_text(encoding="utf-8"))
    valve = labels["examples"]["p1-node2"]["detection"]
    assert valve["symbol_class"] == "gate_valve"
    # the default valve half-extent is 8 pt around the artifact center
    assert valve["bbox"] == [(300 - 8) * SCALE, (135 - 8) * SCALE,
                             (300 + 8) * SCALE, (135 + 8) * SCALE]

    box = valve["bbox"]
    window = (box[0] - 15, box[1] - 15, box[2] + 15, box[3] + 15)
    ink = ink_pixels(raster, width, window)
    assert ink, "the rendered valve glyph left no ink where projected"
    assert all(box[0] <= x <= box[2] and box[1] <= y <= box[3]
               for x, y in ink)


def test_text_span_projects_onto_its_rendered_glyphs(factory_run):
    report, out = factory_run
    width, _, raster = decode_gray_png(
        (out / "sheets" / "sheet_1.png").read_bytes())
    labels = json.loads(
        (out / "labels" / profile_key(PROFILE) / "sheet_1.json")
        .read_text(encoding="utf-8"))
    texts = [example["detection"]
             for example in labels["examples"].values()
             if example["kind"] == "text"]
    assert [t["string"] for t in texts] == ["T-101"]
    tag = texts[0]
    assert tag["text_class"] == "equipment_tag"

    box = tag["bbox"]
    window = (box[0] - 15, box[1] - 15, box[2] + 15, box[3] + 15)
    ink = ink_pixels(raster, width, window)
    assert ink, "the rendered tag left no ink where projected"
    pad = 1.0  # antialiasing bleed at the glyph edge
    assert all(box[0] - pad <= x <= box[2] + pad
               and box[1] - pad <= y <= box[3] + pad
               for x, y in ink)
