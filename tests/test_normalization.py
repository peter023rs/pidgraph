"""Raster normalization tests (ticket 03): deskew, resolution-normalize,
binarize — the first Raster Path stage on every Sheet.

Unit tests exercise the normalize_sheet() function directly (it has no seam
— deterministic classical CV is a spec decision); seam tests digitize()
whole Documents and assert the recorded provenance and that all geometry
maps back to the original Sheet frame, so overlays stay correct.
"""

import math
from dataclasses import replace

import pytest

from pidgraph.model import Document, Sheet, SheetAnnotations
from pidgraph.normalize import (
    SKEW_TOLERANCE_DEGREES,
    TARGET_LONG_SIDE,
    normalize_sheet,
)
from pidgraph.pipeline import digitize

from conftest import LINES, SHEET_H, SHEET_W, SYMBOLS, TEXTS, _render

CLEAN_ANNOTATIONS = SheetAnnotations(symbols=SYMBOLS, lines=LINES,
                                     texts=TEXTS)
CLEAN_RASTER = _render(SHEET_W, SHEET_H, CLEAN_ANNOTATIONS)


# --------------------------------------------------------------------------
# Test-local geometry helpers, independent of pidgraph.normalize internals

def _rotate_point(point, degrees, center):
    rad = math.radians(degrees)
    cos_t, sin_t = math.cos(rad), math.sin(rad)
    x, y = point[0] - center[0], point[1] - center[1]
    return (cos_t * x - sin_t * y + center[0],
            sin_t * x + cos_t * y + center[1])


def _rotate_raster(raster, width, height, degrees):
    """Rotate content by `degrees` about the raster center (inverse-map
    nearest neighbor), the same screen coordinates the Sheet uses."""
    center = (width / 2, height / 2)
    out = bytearray(b"\xff" * (width * height))
    for y in range(height):
        for x in range(width):
            sx, sy = _rotate_point((x + 0.5, y + 0.5), -degrees, center)
            xi, yi = int(sx), int(sy)
            if sx >= 0 and sy >= 0 and xi < width and yi < height:
                out[y * width + x] = raster[yi * width + xi]
    return bytes(out)


def _rotate_bbox(bbox, degrees, center):
    x0, y0, x1, y1 = bbox
    corners = [_rotate_point(p, degrees, center)
               for p in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return (min(xs), min(ys), max(xs), max(ys))


def _skewed_annotations(degrees):
    center = (SHEET_W / 2, SHEET_H / 2)
    symbols = tuple(
        replace(s, bbox=_rotate_bbox(s.bbox, degrees, center),
                direction=None if s.direction is None else
                _rotate_point(s.direction, degrees, (0.0, 0.0)))
        for s in SYMBOLS)
    lines = tuple(
        replace(l, polyline=tuple(_rotate_point(p, degrees, center)
                                  for p in l.polyline))
        for l in LINES)
    texts = tuple(replace(t, bbox=_rotate_bbox(t.bbox, degrees, center))
                  for t in TEXTS)
    return SheetAnnotations(symbols=symbols, lines=lines, texts=texts)


def _assert_polyline_close(got, expected, tolerance=1e-12):
    assert len(got) == len(expected)
    for (gx, gy), (ex, ey) in zip(got, expected):
        assert (gx, gy) == pytest.approx((ex, ey), abs=tolerance)


def _assert_endpoints_near(got_polyline, annotated_polyline, tolerance):
    """Lines are traced from pixels (masked at symbol boxes), so extracted
    endpoints sit within a few pixels of the annotated ones — a frame
    mapping error would put them tens of pixels away."""
    for got, want in ((got_polyline[0], annotated_polyline[0]),
                      (got_polyline[-1], annotated_polyline[-1])):
        assert math.dist(got, want) <= tolerance, (got, want)


def _scaled_annotations(factor):
    def bbox(b):
        return (b[0] * factor, b[1] * factor, b[2] * factor, b[3] * factor)

    symbols = tuple(replace(s, bbox=bbox(s.bbox)) for s in SYMBOLS)
    lines = tuple(
        replace(l, polyline=tuple((x * factor, y * factor)
                                  for x, y in l.polyline))
        for l in LINES)
    texts = tuple(replace(t, bbox=bbox(t.bbox)) for t in TEXTS)
    return SheetAnnotations(symbols=symbols, lines=lines, texts=texts)


def _double_resolution_sheet(number=1):
    annotations = _scaled_annotations(2.0)
    return Sheet(number=number, width=SHEET_W * 2, height=SHEET_H * 2,
                 raster=_render(SHEET_W * 2, SHEET_H * 2, annotations),
                 annotations=annotations)


def _skewed_sheet(degrees, number=1):
    return Sheet(number=number, width=SHEET_W, height=SHEET_H,
                 raster=_rotate_raster(CLEAN_RASTER, SHEET_W, SHEET_H,
                                       degrees),
                 annotations=_skewed_annotations(degrees))


# --------------------------------------------------------------------------
# Unit: normalize_sheet()

def test_clean_on_target_sheet_normalizes_to_identity():
    sheet = Sheet(number=1, width=SHEET_W, height=SHEET_H,
                  raster=CLEAN_RASTER, annotations=CLEAN_ANNOTATIONS)

    normalized, norm = normalize_sheet(sheet)

    assert norm.angle_degrees == 0.0
    assert norm.scale == 1.0
    assert norm.binarization == "otsu"
    assert (normalized.width, normalized.height) == (SHEET_W, SHEET_H)
    # already binary and axis-aligned: pixels come through untouched
    assert normalized.raster == CLEAN_RASTER
    assert normalized.annotations == CLEAN_ANNOTATIONS


@pytest.mark.parametrize("degrees", [3.2, -2.4, 0.75])
def test_known_skew_recovered_within_stated_tolerance(degrees):
    skewed = _rotate_raster(CLEAN_RASTER, SHEET_W, SHEET_H, degrees)
    sheet = Sheet(number=1, width=SHEET_W, height=SHEET_H, raster=skewed)

    _, norm = normalize_sheet(sheet)

    assert abs(norm.angle_degrees - degrees) <= SKEW_TOLERANCE_DEGREES


def _ink_pixels(raster, width):
    return {(i % width, i // width)
            for i, value in enumerate(raster) if value == 0}


def _fraction_within_one_pixel(points, reference):
    hits = sum(1 for x, y in points
               if any((x + dx, y + dy) in reference
                      for dx in (-1, 0, 1) for dy in (-1, 0, 1)))
    return hits / len(points)


def test_deskew_recovers_the_clean_drawing_in_pixels():
    """Not just the angle: the corrected raster itself must land back on
    the clean render (a sign or center error would leave it rotated)."""
    skewed = _rotate_raster(CLEAN_RASTER, SHEET_W, SHEET_H, 3.2)
    sheet = Sheet(number=1, width=SHEET_W, height=SHEET_H, raster=skewed)

    normalized, _ = normalize_sheet(sheet)

    clean_ink = _ink_pixels(CLEAN_RASTER, SHEET_W)
    assert _fraction_within_one_pixel(
        _ink_pixels(skewed, SHEET_W), clean_ink) < 0.6
    assert _fraction_within_one_pixel(
        _ink_pixels(normalized.raster, SHEET_W), clean_ink) >= 0.98


def test_off_target_resolution_normalized_to_target_scale():
    sheet = _double_resolution_sheet()

    normalized, norm = normalize_sheet(sheet)

    assert norm.scale == pytest.approx(0.5)
    assert (normalized.width, normalized.height) == \
        (TARGET_LONG_SIDE, TARGET_LONG_SIDE // 2)
    # downstream stages consume the normalized frame: annotations rescaled
    assert normalized.annotations.symbols[0].bbox == \
        pytest.approx(SYMBOLS[0].bbox)
    _assert_polyline_close(normalized.annotations.lines[0].polyline,
                           LINES[0].polyline)
    # ...and the raster keeps its (thin) ink where run1 now lies
    assert normalized.raster[110 * TARGET_LONG_SIDE + 100] == 0
    assert normalized.raster[110 * TARGET_LONG_SIDE + 390] == 255


def test_below_target_resolution_upscaled_to_target_scale():
    annotations = _scaled_annotations(0.5)
    sheet = Sheet(number=1, width=SHEET_W // 2, height=SHEET_H // 2,
                  raster=_render(SHEET_W // 2, SHEET_H // 2, annotations),
                  annotations=annotations)

    normalized, norm = normalize_sheet(sheet)

    assert norm.scale == pytest.approx(2.0)
    assert (normalized.width, normalized.height) == (SHEET_W, SHEET_H)
    _assert_polyline_close(normalized.annotations.lines[0].polyline,
                           LINES[0].polyline)
    # run1's ink came up with the resolution (rows 110-111 after 2x)
    assert normalized.raster[110 * SHEET_W + 100] == 0
    assert normalized.raster[110 * SHEET_W + 390] == 255


def test_rasterless_sheet_passes_through_unchanged():
    sheet = Sheet(number=1, width=SHEET_W, height=SHEET_H,
                  annotations=CLEAN_ANNOTATIONS)

    normalized, norm = normalize_sheet(sheet)

    assert normalized == sheet
    assert norm.is_identity
    assert norm.binarization == "none"
    assert norm.threshold is None


def test_grayscale_sheet_binarized_with_recorded_method():
    gray_table = bytes(64 if v == 0 else 200 for v in range(256))
    sheet = Sheet(number=1, width=SHEET_W, height=SHEET_H,
                  raster=CLEAN_RASTER.translate(gray_table))

    normalized, norm = normalize_sheet(sheet)

    assert set(normalized.raster) <= {0, 255}
    assert norm.binarization == "otsu"
    assert 64 <= norm.threshold < 200
    # ink stayed put: same pixels dark as in the clean binary render
    assert normalized.raster == CLEAN_RASTER


def test_transforms_round_trip_between_frames():
    from pidgraph.normalize import Normalization

    _, from_skewed_sheet = normalize_sheet(_skewed_sheet(3.2))
    skew_and_scale_combined = Normalization(
        angle_degrees=3.2, scale=0.5, binarization="otsu", threshold=0,
        original_size=(SHEET_W * 2, SHEET_H * 2),
        normalized_size=(SHEET_W, SHEET_H))

    for norm in (from_skewed_sheet, skew_and_scale_combined):
        for point in ((66.0, 110.0), (340.0, 152.0), (20.0, 80.0)):
            roundtrip = norm.to_original(norm.to_normalized(point))
            assert roundtrip == pytest.approx(point, abs=1e-9)


# --------------------------------------------------------------------------
# Seam: digitize() — provenance in the run artifacts, overlay-correct
# geometry in the original Sheet frame

def test_normalization_recorded_in_sheet_provenance(tmp_path,
                                                    synthetic_document,
                                                    synthetic_profile):
    import json

    artifacts = digitize(synthetic_document, synthetic_profile,
                         out_dir=tmp_path)

    record = artifacts.detection_records[0]
    assert record["normalization"] == {
        "angle_degrees": 0.0,
        "scale": 1.0,
        "binarization": "otsu",
        "threshold": 0,
        "original_size": [SHEET_W, SHEET_H],
        "normalized_size": [SHEET_W, SHEET_H],
    }

    on_disk = json.loads((tmp_path / "detections" / "sheet_1.json")
                         .read_text(encoding="utf-8"))
    assert on_disk["normalization"] == record["normalization"]


def test_off_target_document_geometry_maps_back_to_original(
        synthetic_profile):
    document = Document(name="double-res.pdf",
                        sheets=(_double_resolution_sheet(),))

    artifacts = digitize(document, synthetic_profile)

    record = artifacts.detection_records[0]
    assert record["normalization"]["scale"] == pytest.approx(0.5)

    # detection-record geometry is in the original 800x400 frame
    doubled = _scaled_annotations(2.0)
    for got, annotated in zip(record["texts"], doubled.texts):
        assert tuple(got["bbox"]) == pytest.approx(annotated.bbox)
    assert len(record["lines"]) == len(doubled.lines)
    for got, annotated in zip(record["lines"], doubled.lines):
        _assert_endpoints_near(got["polyline"], annotated.polyline,
                               tolerance=6.0)

    # and so is the DEXPI model: run3's corner at 2x its base position
    corners = [n for n in
               artifacts.plant_model["conceptualModel"]["PipingNode"]
               if n["nodeType"] == "corner"]
    assert [c["position"] for c in corners] == \
        [pytest.approx([680.0, 340.0], abs=2.0)]


def test_skewed_document_overlay_geometry_maps_back(synthetic_profile):
    degrees = 3.2
    document = Document(name="skewed.pdf",
                        sheets=(_skewed_sheet(degrees),))

    artifacts = digitize(document, synthetic_profile)

    record = artifacts.detection_records[0]
    assert abs(record["normalization"]["angle_degrees"] - degrees) <= \
        SKEW_TOLERANCE_DEGREES

    skewed = _skewed_annotations(degrees)
    # polylines land back on the original (skewed) Sheet
    assert len(record["lines"]) == len(skewed.lines)
    for got, annotated in zip(record["lines"], skewed.lines):
        _assert_endpoints_near(got["polyline"], annotated.polyline,
                               tolerance=5.0)
    # bboxes stay centered on the original symbols and still cover them
    for got, annotated in zip(record["symbols"], skewed.symbols):
        gx0, gy0, gx1, gy1 = got["bbox"]
        ax0, ay0, ax1, ay1 = annotated.bbox
        assert ((gx0 + gx1) / 2, (gy0 + gy1) / 2) == \
            pytest.approx(((ax0 + ax1) / 2, (ay0 + ay1) / 2), abs=1e-6)
        assert gx0 <= ax0 + 1e-6 and gy0 <= ay0 + 1e-6
        assert gx1 >= ax1 - 1e-6 and gy1 >= ay1 - 1e-6

    # downstream assembly still resolves the same topology
    tags = {n["tag"] for n in artifacts.plant_graph["nodes"]}
    assert tags == {"T-101", "V-101", "T-102", "PI-100", "OPC-DW02-0003"}
