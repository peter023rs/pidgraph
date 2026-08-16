"""Raster normalization: deskew, resolution-normalize, binarize (first
Raster Path stage).

Every Sheet is normalized before any extraction stage sees it, so detector
and OCR behavior doesn't depend on scanner quirks: skew is estimated by
projection-profile search and corrected, the raster is rescaled so its
long side hits TARGET_LONG_SIDE (the v1 synthetic-corpus scale; revisited
when the real corpus arrives), and pixels are binarized with Otsu's
threshold. Deterministic classical CV throughout — no seam, no ML (spec
decision).

Two coordinate frames exist from here on. Extraction stages consume the
normalized Sheet — raster and ground-truth annotations both transformed —
while every emitted geometry is mapped back through
Normalization.to_original() so detection records and the plant model
overlay the operator's original Sheet. What was applied to each Sheet is
recorded in its provenance via Normalization.as_record().
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .model import (
    Bbox,
    LineDetection,
    Point,
    Sheet,
    SheetAnnotations,
    SymbolDetection,
    TextDetection,
)

TARGET_LONG_SIDE = 400

SKEW_SEARCH_DEGREES = 15.0
SKEW_TOLERANCE_DEGREES = 0.1   # stated recovery guarantee, held by tests
_COARSE_STEP = 0.5
_FINE_STEP = 0.05
_MAX_SKEW_SAMPLES = 20000


def _rotated(point: Point, degrees: float, center: Point) -> Point:
    rad = math.radians(degrees)
    cos_t, sin_t = math.cos(rad), math.sin(rad)
    x, y = point[0] - center[0], point[1] - center[1]
    return (cos_t * x - sin_t * y + center[0],
            sin_t * x + cos_t * y + center[1])


def _mapped_bbox(bbox: Bbox, mapper) -> Bbox:
    """Axis-aligned hull of a bbox pushed through a point transform."""
    x0, y0, x1, y1 = bbox
    corners = [mapper((x0, y0)), mapper((x1, y0)),
               mapper((x1, y1)), mapper((x0, y1))]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return (min(xs), min(ys), max(xs), max(ys))


@dataclass(frozen=True)
class Normalization:
    """What normalization did to one Sheet, and the map between the
    original and normalized coordinate frames."""

    angle_degrees: float   # detected skew; correction rotated by -angle
    scale: float           # resolution factor applied after deskew
    binarization: str      # "otsu", or "none" for a rasterless Sheet
    threshold: int | None  # ink is <= threshold in the original grayscale
    original_size: tuple[int, int]
    normalized_size: tuple[int, int]

    @property
    def is_identity(self) -> bool:
        return self.angle_degrees == 0.0 and self.scale == 1.0

    def _center(self) -> Point:
        return (self.original_size[0] / 2, self.original_size[1] / 2)

    def to_normalized(self, point: Point) -> Point:
        if self.angle_degrees == 0.0:
            return (point[0] * self.scale, point[1] * self.scale)
        x, y = _rotated(point, -self.angle_degrees, self._center())
        return (x * self.scale, y * self.scale)

    def to_original(self, point: Point) -> Point:
        if self.angle_degrees == 0.0:
            return (point[0] / self.scale, point[1] / self.scale)
        return _rotated((point[0] / self.scale, point[1] / self.scale),
                        self.angle_degrees, self._center())

    def as_record(self) -> dict:
        return {
            "angle_degrees": self.angle_degrees,
            "scale": self.scale,
            "binarization": self.binarization,
            "threshold": self.threshold,
            "original_size": list(self.original_size),
            "normalized_size": list(self.normalized_size),
        }


# --------------------------------------------------------------------------
# Binarization: Otsu's threshold

def _otsu_threshold(raster: bytes) -> int:
    """Threshold maximizing between-class variance; ties break toward the
    lowest threshold, so an already-binary raster keeps ink == 0."""
    histogram = [raster.count(value) for value in range(256)]
    total = len(raster)
    total_sum = sum(value * count for value, count in enumerate(histogram))
    best_threshold, best_variance = 0, -1.0
    ink_count, ink_sum = 0, 0
    for threshold in range(256):
        ink_count += histogram[threshold]
        ink_sum += threshold * histogram[threshold]
        paper_count = total - ink_count
        if ink_count == 0 or paper_count == 0:
            continue
        ink_mean = ink_sum / ink_count
        paper_mean = (total_sum - ink_sum) / paper_count
        variance = ink_count * paper_count * (ink_mean - paper_mean) ** 2
        if variance > best_variance:
            best_variance, best_threshold = variance, threshold
    return best_threshold


def _binarized(raster: bytes, threshold: int) -> bytes:
    table = bytes(0 if value <= threshold else 255 for value in range(256))
    return raster.translate(table)


# --------------------------------------------------------------------------
# Deskew: projection-profile search (rows of ink sharpen at the true angle)

def _ink_points(raster: bytes, width: int,
                threshold: int) -> list[Point]:
    points = [(float(i % width), float(i // width))
              for i, value in enumerate(raster) if value <= threshold]
    if len(points) > _MAX_SKEW_SAMPLES:
        stride = math.ceil(len(points) / _MAX_SKEW_SAMPLES)
        points = points[::stride]
    return points


def _row_alignment(points: list[Point], degrees: float,
                   center: Point) -> int:
    """Sum of squared row occupancies after rotating ink by -degrees:
    maximal when horizontal strokes collapse into single rows."""
    rad = math.radians(-degrees)
    cos_t, sin_t = math.cos(rad), math.sin(rad)
    cx, cy = center
    counts: dict[int, int] = {}
    for x, y in points:
        row = round(sin_t * (x - cx) + cos_t * (y - cy))
        counts[row] = counts.get(row, 0) + 1
    return sum(n * n for n in counts.values())


def _estimate_skew(points: list[Point], center: Point) -> float:
    if len(points) < 2:
        return 0.0

    def best_of(candidates: list[float], seed: float) -> float:
        best, best_score = seed, -1
        for candidate in candidates:
            score = _row_alignment(points, candidate, center)
            # ties prefer the smaller correction, so a clean Sheet stays
            # at exactly 0.0 (the identity path)
            if score > best_score or (score == best_score
                                      and abs(candidate) < abs(best)):
                best, best_score = candidate, score
        return best

    steps = int(SKEW_SEARCH_DEGREES / _COARSE_STEP)
    coarse = best_of(
        [round(i * _COARSE_STEP - SKEW_SEARCH_DEGREES, 1)
         for i in range(2 * steps + 1)], 0.0)
    fine_steps = int(_COARSE_STEP / _FINE_STEP)
    return best_of(
        [round(coarse + (i - fine_steps) * _FINE_STEP, 2)
         for i in range(2 * fine_steps + 1)], coarse)


def _deskewed(raster: bytes, width: int, height: int,
              angle_degrees: float) -> bytes:
    """Rotate content by -angle about the center (inverse-map nearest
    neighbor, same canvas, paper fill)."""
    if angle_degrees == 0.0:
        return raster
    rad = math.radians(angle_degrees)
    cos_t, sin_t = math.cos(rad), math.sin(rad)
    cx, cy = width / 2, height / 2
    out = bytearray(width * height)
    i = 0
    for y in range(height):
        dy = y + 0.5 - cy
        sx = cos_t * (0.5 - cx) - sin_t * dy + cx
        sy = sin_t * (0.5 - cx) + cos_t * dy + cy
        for x in range(width):
            if 0.0 <= sx < width and 0.0 <= sy < height:
                out[i] = raster[int(sy) * width + int(sx)]
            else:
                out[i] = 255
            sx += cos_t
            sy += sin_t
            i += 1
    return bytes(out)


# --------------------------------------------------------------------------
# Resolution normalization

def _rescaled(raster: bytes, width: int, height: int,
              target_width: int, target_height: int,
              scale: float) -> bytes:
    if scale == 1.0:
        return raster
    out = bytearray(target_width * target_height)
    if scale < 1.0:
        # min-pool when shrinking: a thin ink stroke survives, where
        # nearest-neighbor sampling could drop it between pixel centers
        for y in range(target_height):
            row_start = min(height - 1, int(y / scale))
            row_stop = min(height, math.ceil((y + 1) / scale))
            base_out = y * target_width
            for x in range(target_width):
                col_start = min(width - 1, int(x / scale))
                col_stop = min(width, math.ceil((x + 1) / scale))
                out[base_out + x] = min(
                    min(raster[row * width + col_start:
                               row * width + col_stop])
                    for row in range(row_start, row_stop))
    else:
        # nearest neighbor when growing
        for y in range(target_height):
            base_in = min(height - 1, int((y + 0.5) / scale)) * width
            base_out = y * target_width
            for x in range(target_width):
                out[base_out + x] = raster[
                    base_in + min(width - 1, int((x + 0.5) / scale))]
    return bytes(out)


# --------------------------------------------------------------------------
# Frame mapping for annotations (into the normalized frame) and detections
# (back to the original frame)

def _normalized_annotations(annotations: SheetAnnotations | None,
                            norm: Normalization) -> SheetAnnotations | None:
    if annotations is None or norm.is_identity:
        return annotations
    origin = (0.0, 0.0)
    symbols = tuple(
        replace(s, bbox=_mapped_bbox(s.bbox, norm.to_normalized),
                direction=None if s.direction is None else
                _rotated(s.direction, -norm.angle_degrees, origin))
        for s in annotations.symbols)
    lines = tuple(
        replace(l, polyline=tuple(norm.to_normalized(p)
                                  for p in l.polyline))
        for l in annotations.lines)
    texts = tuple(replace(t, bbox=_mapped_bbox(t.bbox, norm.to_normalized))
                  for t in annotations.texts)
    return SheetAnnotations(symbols=symbols, lines=lines, texts=texts)


def map_detections_to_original(
        norm: Normalization,
        symbols: list[SymbolDetection],
        lines: list[LineDetection],
        texts: list[TextDetection],
) -> tuple[list[SymbolDetection], list[LineDetection],
           list[TextDetection]]:
    """Map detection geometry from the normalized frame back onto the
    original Sheet, so overlays land on what the operator scanned."""
    if norm.is_identity:
        return symbols, lines, texts
    origin = (0.0, 0.0)
    return (
        [replace(s, bbox=_mapped_bbox(s.bbox, norm.to_original),
                 direction=None if s.direction is None else
                 _rotated(s.direction, norm.angle_degrees, origin))
         for s in symbols],
        [replace(l, polyline=tuple(norm.to_original(p)
                                   for p in l.polyline))
         for l in lines],
        [replace(t, bbox=_mapped_bbox(t.bbox, norm.to_original))
         for t in texts],
    )


# --------------------------------------------------------------------------
# The stage itself

def normalize_sheet(sheet: Sheet,
                    target_long_side: int = TARGET_LONG_SIDE,
                    ) -> tuple[Sheet, Normalization]:
    """Deskew, resolution-normalize, and binarize one Sheet.

    Returns the normalized Sheet (raster and any annotations in the
    normalized frame) plus the Normalization that was applied — the map
    back to original coordinates and the provenance record.
    """
    size = (sheet.width, sheet.height)
    if sheet.raster is None:
        return sheet, Normalization(
            angle_degrees=0.0, scale=1.0, binarization="none",
            threshold=None, original_size=size, normalized_size=size)

    threshold = _otsu_threshold(sheet.raster)
    ink = _ink_points(sheet.raster, sheet.width, threshold)
    angle = _estimate_skew(ink, (sheet.width / 2, sheet.height / 2))
    scale = target_long_side / max(sheet.width, sheet.height)
    target_width = round(sheet.width * scale)
    target_height = round(sheet.height * scale)

    norm = Normalization(
        angle_degrees=angle, scale=scale, binarization="otsu",
        threshold=threshold, original_size=size,
        normalized_size=(target_width, target_height))

    raster = _deskewed(sheet.raster, sheet.width, sheet.height, angle)
    raster = _rescaled(raster, sheet.width, sheet.height,
                       target_width, target_height, scale)
    raster = _binarized(raster, threshold)

    return Sheet(number=sheet.number, width=target_width,
                 height=target_height, raster=raster,
                 annotations=_normalized_annotations(sheet.annotations,
                                                     norm)), norm
