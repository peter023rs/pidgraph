"""Trained SymbolDetector (ticket 16): per-company fine-tuning behind
the SymbolDetector seam — the plan-of-record adaptation mechanism.

Training reads label-factory datasets (the clean set and any degraded
variants side by side, ticket 14) and produces a versioned detector
artifact: one ink-probability prototype per symbol class — arrows one
per labeled direction sector, so a match carries its orientation as
evidence — learned from the labeled crops, plus an acceptance threshold
calibrated on those same crops, so a detector trained on degraded data
accepts what degradation does to its symbols. Everything runs on the stdlib alone;
training and inference are fully local, and the artifact lives outside
git with the rest of the drawing-derived content (ADR-0001).

Scale is the contract: crops are taken in the pipeline's own normalized
frame (the same normalize_sheet every Sheet passes through before the
detector sees it), so training and inference always operate at the one
operating scale. v1 learns a single canonical patch per class — size
variation beyond resampling is future work for the Workbench corpus.

Inference slides each class prototype over the binarized raster
(ink-count prefilter, Pearson correlation, non-maximum suppression) and
emits detections whose confidence is the match score and whose
provenance names the detector version — the content hash stamped into
the artifact at training time and verified at load, so a hand-edited
artifact is refused rather than silently misreported.

Beside it lives the candidate cheap adaptation mechanism (ticket 17):
the Legend Dictionary nearest-neighbor classifier, which matches the
same way against the Convention Profile bundle's own glyph crops
instead of trained prototypes. Architecture is identical — the same
seam, selected by configuration — only onboarding cost differs, and
the eval harness judges the two side by side on evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping, NamedTuple, Sequence
from urllib.parse import quote, unquote

from .labels import LabelStore, profile_key, supervised_label
from .model import (
    UNCLASSIFIED_SYMBOL,
    Bbox,
    ConventionProfile,
    Provenance,
    Sheet,
    SymbolDetection,
)
from .normalize import TARGET_LONG_SIDE, normalize_sheet
from .pngio import decode_gray_png, encode_gray_png
from .profile import load_profile

DETECTOR_FORMAT = "pidgraph-trained-symbol-detector/1"

# artifacts carry drawing-derived content: data/ is ignored (ADR-0001)
DEFAULT_OUT_DIR = Path("data") / "detectors"

_INK_THRESHOLD = 128   # normalized rasters are binarized 0/255; midpoint
_MIN_SIDE = 2          # a canonical patch side never collapses below 2 px

# the acceptance threshold sits under the worst training score by this
# margin, floored so a destroyed training crop cannot open the gate to
# everything
_THRESHOLD_MARGIN = 0.85
_THRESHOLD_FLOOR = 0.25

# crops learn a ring of context around the label box: a filled symbol
# with no context is a uniform patch, and a uniform patch correlates
# with nothing; the emitted detection trims the ring back off
_CONTEXT_MARGIN = 2

# inference: windows whose ink count is far from the prototype's
# expected ink cannot match and are skipped before scoring
_MIN_INK_RATIO = 0.45
_MAX_INK_RATIO = 2.5

# non-maximum suppression: a candidate overlapping a better-scoring one
# this much (IoU), or contained in one this much (intersection over the
# smaller box), is the same ink reported twice
_NMS_IOU = 0.3
_NMS_CONTAINMENT = 0.65


# --------------------------------------------------------------------------
# Ink crops: binary rasters, ink-preserving resampling, Pearson scores

def _ink_bits(raster: bytes) -> bytes:
    """The raster as ink bits (1 = ink). Normalized rasters are already
    binarized; the midpoint split keeps raw grayscale workable too."""
    table = bytes(1 if value < _INK_THRESHOLD else 0
                  for value in range(256))
    return raster.translate(table)


def _resampled(bits: bytes, width: int, height: int,
               target_width: int, target_height: int) -> bytes:
    """Resample an ink crop: max-pool when shrinking (a thin stroke
    survives, the normalize-stage stance), nearest when growing."""
    if (width, height) == (target_width, target_height):
        return bits
    out = bytearray(target_width * target_height)
    for y in range(target_height):
        y0 = y * height // target_height
        y1 = max(y0 + 1, math.ceil((y + 1) * height / target_height))
        for x in range(target_width):
            x0 = x * width // target_width
            x1 = max(x0 + 1, math.ceil((x + 1) * width / target_width))
            out[y * target_width + x] = int(any(
                bits[yy * width + xx]
                for yy in range(y0, y1) for xx in range(x0, x1)))
    return bytes(out)


def _correlation(n: int, cross: float, ink: int, prototype_sum: float,
                 variance_term: float) -> float:
    """The Pearson correlation given its accumulated terms — the one
    scoring formula calibration and the detection scan both call, so
    the two can never drift apart. A degenerate window (no variance on
    either side) correlates with nothing and scores 0."""
    denominator = math.sqrt(variance_term * (n * ink - ink * ink))
    if denominator <= 0:
        return 0.0
    return (n * cross - prototype_sum * ink) / denominator


def _pearson(bits: bytes, prototype: Sequence[float],
             prototype_sum: float, prototype_sq_sum: float) -> float:
    """Pearson correlation between a binary crop and an ink-probability
    prototype of the same size."""
    n = len(bits)
    ink = sum(bits)
    if ink == 0 or ink == n:
        return 0.0
    cross = sum(p for p, b in zip(prototype, bits) if b)
    return _correlation(n, cross, ink, prototype_sum,
                        n * prototype_sq_sum - prototype_sum * prototype_sum)


def _median(values: Sequence[int]) -> int:
    """Upper median — a size some real crop actually had, never one
    invented by averaging."""
    return sorted(values)[len(values) // 2]


# --------------------------------------------------------------------------
# Direction: labeled arrow orientations become oriented prototype
# variants, so a matched arrow carries its direction as evidence — the
# assembly stage rightly refuses a flow arrow that asserts none

_SECTORS = 8  # compass sectors oriented variants are grouped into


def _unit(direction) -> tuple[float, float] | None:
    if direction is None:
        return None
    x, y = float(direction[0]), float(direction[1])
    length = math.hypot(x, y)
    if length == 0.0:
        return None
    return x / length, y / length


def _sector(unit: tuple[float, float]) -> int:
    return round(math.atan2(unit[1], unit[0])
                 / (2 * math.pi / _SECTORS)) % _SECTORS


def _mean_direction(units: Sequence[tuple[float, float]]) -> list[float]:
    x = sum(u[0] for u in units) / len(units)
    y = sum(u[1] for u in units) / len(units)
    length = math.hypot(x, y)
    return [round(x / length, 6), round(y / length, 6)]


# --------------------------------------------------------------------------
# Collecting labeled crops in the normalized frame

def _normalized_box(norm, bbox: Sequence[float], width: int,
                    height: int) -> tuple[int, int, int, int] | None:
    """A label box mapped into the normalized frame, snapped outward to
    the pixel grid, and grown by the context ring — or None when nothing
    of it survives on the canvas (its ink was lost to deskew fill or the
    canvas edge)."""
    corners = [norm.to_normalized((x, y))
               for x, y in ((bbox[0], bbox[1]), (bbox[2], bbox[1]),
                            (bbox[2], bbox[3]), (bbox[0], bbox[3]))]
    x0 = max(0, math.floor(min(x for x, _ in corners)) - _CONTEXT_MARGIN)
    y0 = max(0, math.floor(min(y for _, y in corners)) - _CONTEXT_MARGIN)
    x1 = min(width,
             math.ceil(max(x for x, _ in corners)) + _CONTEXT_MARGIN)
    y1 = min(height,
             math.ceil(max(y for _, y in corners)) + _CONTEXT_MARGIN)
    if x1 - x0 < _MIN_SIDE or y1 - y0 < _MIN_SIDE:
        return None
    return x0, y0, x1, y1


class _Crop(NamedTuple):
    """One labeled symbol's ink crop in the normalized frame."""
    direction: "tuple[float, float] | None"
    width: int
    height: int
    bits: bytes


def _crop(bits: bytes, width: int,
          box: tuple[int, int, int, int]) -> tuple[int, int, bytes]:
    x0, y0, x1, y1 = box
    crop = b"".join(bits[y * width + x0:y * width + x1]
                    for y in range(y0, y1))
    return x1 - x0, y1 - y0, crop


def _collect_crops(root: Path, profile: Mapping[str, str]) -> dict:
    """One dataset root's labeled symbol crops, taken from the Sheet
    rasters in the pipeline's normalized frame. Fails closed like the
    eval harness: a root with no labeled Sheets, labels recorded for a
    different profile, or a labeled Sheet missing its raster is refused
    rather than silently narrowing the training set."""
    store = LabelStore(root / "labels")
    numbers = store.sheets(profile)
    if not numbers:
        raise ValueError(
            f"no labeled Sheets for profile {profile_key(profile)} under"
            f" {root / 'labels'}; the store holds"
            f" {store.profiles() or 'no profiles'}")

    crops: dict[str, list[_Crop]] = {}
    examples = 0
    off_canvas = 0
    for number in numbers:
        labels = store.sheet_labels(profile, number)
        if labels["profile"] != dict(profile):
            raise ValueError(
                f"labels under {root} for Sheet {number} were recorded"
                f" for profile {labels['profile']!r} — refusing to train"
                f" on them")
        png_path = root / "sheets" / f"sheet_{number}.png"
        if not png_path.is_file():
            raise ValueError(
                f"Sheet {number} carries no raster: {root} is missing"
                f" {png_path.name} and training reads pixels")
        width, height, pixels = decode_gray_png(png_path.read_bytes())
        normalized, norm = normalize_sheet(
            Sheet(number=number, width=width, height=height, raster=pixels))
        assert normalized.raster is not None
        bits = _ink_bits(normalized.raster)
        for _, stored in sorted(labels["examples"].items()):
            if stored["kind"] != "symbol":
                continue
            truth = supervised_label(number, stored)
            if truth is None:  # a reject leaves no ink to learn from
                continue
            box = _normalized_box(norm, truth["bbox"],
                                  normalized.width, normalized.height)
            if box is None:
                off_canvas += 1
                continue
            direction = _unit(stored["detection"].get("direction"))
            crops.setdefault(truth["symbol_class"], []).append(
                _Crop(direction, *_crop(bits, normalized.width, box)))
            examples += 1
    return {"root": str(root), "sheets": numbers, "examples": examples,
            "off_canvas": off_canvas, "crops": crops}


# --------------------------------------------------------------------------
# Training: crops -> prototypes + calibrated thresholds -> artifact

def _prototype_bytes(resampled: Sequence[bytes], size: int) -> bytes:
    """Per-pixel mean ink probability, quantized to bytes — the exact
    values inference correlates against, so calibration and detection
    can never disagree on the prototype."""
    count = len(resampled)
    return bytes(
        round(255 * sum(crop[i] for crop in resampled) / count)
        for i in range(size))


def _prototype_values(quantized: bytes) -> list[float]:
    return [value / 255 for value in quantized]


def _variant_entry(symbol_class: str, sector: int | None,
                   crops: Sequence[_Crop]) -> tuple[dict, bytes]:
    """One prototype variant of a class — all its crops when no
    direction is labeled, or one compass sector's worth when it is."""
    target_width = max(_MIN_SIDE, _median([c.width for c in crops]))
    target_height = max(_MIN_SIDE, _median([c.height for c in crops]))
    resampled = [_resampled(c.bits, c.width, c.height,
                            target_width, target_height)
                 for c in crops]
    quantized = _prototype_bytes(resampled, target_width * target_height)
    prototype = _prototype_values(quantized)
    prototype_sum = sum(prototype)
    prototype_sq_sum = sum(p * p for p in prototype)
    scores = [_pearson(bits, prototype, prototype_sum, prototype_sq_sum)
              for bits in resampled]
    threshold = max(_THRESHOLD_FLOOR, _THRESHOLD_MARGIN * min(scores))
    stem = quote(symbol_class, safe="")
    if sector is not None:
        stem += f"@{sector}"
    directions = [c.direction for c in crops if c.direction is not None]
    entry = {
        "prototype": f"prototypes/{stem}.png",
        "width": target_width,
        "height": target_height,
        "direction": (_mean_direction(directions) if directions
                      else None),
        "examples": len(crops),
        "threshold": round(threshold, 6),
        "score_min": round(min(scores), 6),
        "score_mean": round(sum(scores) / len(scores), 6),
    }
    return entry, quantized


def _class_variants(symbol_class: str,
                    crops: Sequence[_Crop]) -> tuple[list[dict],
                                                     dict[str, bytes]]:
    by_sector: dict[int | None, list[_Crop]] = {}
    for crop in crops:
        key = (None if crop.direction is None
               else _sector(crop.direction))
        by_sector.setdefault(key, []).append(crop)
    entries = []
    prototypes: dict[str, bytes] = {}
    for key in sorted(by_sector, key=lambda k: (k is not None, k or 0)):
        entry, quantized = _variant_entry(symbol_class, key,
                                          by_sector[key])
        entries.append(entry)
        prototypes[entry["prototype"]] = quantized
    return entries, prototypes


def _behavior_payload(manifest: dict) -> dict:
    """The manifest fields that determine what the detector does — what
    the version hash covers. Provenance fields (sources, counts) ride
    outside it."""
    return {key: manifest[key]
            for key in ("format", "profile", "normalization", "classes")}


def _artifact_version(manifest: dict,
                      prototypes: Mapping[str, bytes]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(_behavior_payload(manifest), sort_keys=True,
                             ensure_ascii=False).encode("utf-8"))
    for name in sorted(prototypes):
        digest.update(name.encode("utf-8"))
        digest.update(prototypes[name])
    return digest.hexdigest()[:12]


def _write_json(path: Path, payload: dict) -> None:
    """Write-then-replace, like every other persisted artifact here."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    os.replace(tmp, path)


def train_detector(dataset_roots: Sequence[Path | str],
                   profile: Mapping[str, str],
                   out_dir: Path | str) -> dict:
    """Train a symbol detector for one Convention Profile from
    label-factory datasets — pass the clean set root and any degraded
    variant roots together, so the calibrated thresholds absorb what
    degradation does to the symbols. Writes the artifact (manifest plus
    one prototype PNG per class) under out_dir and returns the
    manifest."""
    if not dataset_roots:
        raise ValueError("training needs at least one dataset root"
                         " (the label factory's output directory)")
    profile = dict(profile)
    collected = [_collect_crops(Path(root), profile)
                 for root in dataset_roots]

    crops: dict[str, list[_Crop]] = {}
    for source in collected:
        for symbol_class, class_crops in source.pop("crops").items():
            crops.setdefault(symbol_class, []).extend(class_crops)
    if not crops:
        raise ValueError(
            f"the dataset(s) hold no symbol examples for profile"
            f" {profile_key(profile)} — nothing to train on")

    classes = {}
    prototypes: dict[str, bytes] = {}
    sizes: dict[str, tuple[int, int]] = {}
    for symbol_class in sorted(crops):
        entries, class_prototypes = _class_variants(symbol_class,
                                                    crops[symbol_class])
        classes[symbol_class] = entries
        prototypes.update(class_prototypes)
        for entry in entries:
            sizes[entry["prototype"]] = (entry["width"], entry["height"])

    manifest = {
        "format": DETECTOR_FORMAT,
        "profile": profile,
        "normalization": {"target_long_side": TARGET_LONG_SIDE},
        "classes": classes,
        "sources": collected,
    }
    manifest["version"] = _artifact_version(manifest, prototypes)

    out_dir = Path(out_dir)
    for name, quantized in prototypes.items():
        png_path = out_dir / name
        png_path.parent.mkdir(parents=True, exist_ok=True)
        width, height = sizes[name]
        # stored like a Sheet: ink dark on paper white, inspectable
        png_path.write_bytes(encode_gray_png(
            width, height, bytes(255 - value for value in quantized)))
    _write_json(out_dir / "manifest.json", manifest)
    return manifest


# --------------------------------------------------------------------------
# The detector behind the seam

def _integral_image(bits: bytes, width: int, height: int) -> list[int]:
    """Summed-area table of the ink bits, one extra zero row/column, so
    any window's ink count is four lookups."""
    stride = width + 1
    integral = [0] * (stride * (height + 1))
    for y in range(height):
        row_sum = 0
        base = y * width
        out = (y + 1) * stride
        above = y * stride
        for x in range(width):
            row_sum += bits[base + x]
            integral[out + x + 1] = integral[above + x + 1] + row_sum
    return integral


def _suppresses(kept: Sequence[tuple], candidate: tuple) -> bool:
    cx0, cy0, cx1, cy1 = candidate
    c_area = (cx1 - cx0) * (cy1 - cy0)
    for x0, y0, x1, y1 in kept:
        inter_w = min(cx1, x1) - max(cx0, x0)
        inter_h = min(cy1, y1) - max(cy0, y0)
        if inter_w <= 0 or inter_h <= 0:
            continue
        intersection = inter_w * inter_h
        area = (x1 - x0) * (y1 - y0)
        if intersection / (c_area + area - intersection) >= _NMS_IOU:
            return True
        if intersection / min(c_area, area) >= _NMS_CONTAINMENT:
            return True
    return False


class _ClassMatcher:
    """One prototype variant prepared for scanning: its ink cells, the
    precomputed Pearson terms, and the direction it stands as evidence
    for (oriented variants only)."""

    def __init__(self, symbol_class: str, entry: Mapping,
                 quantized: bytes):
        self.symbol_class = symbol_class
        self.prototype = entry["prototype"]  # the crop file; provenance
                                             # names it as what matched
        self.width = entry["width"]
        self.height = entry["height"]
        self.threshold = entry["threshold"]
        self.direction = (None if entry["direction"] is None
                          else (entry["direction"][0],
                                entry["direction"][1]))
        self.n = self.width * self.height
        prototype = _prototype_values(quantized)
        self.prototype_sum = sum(prototype)
        self.prototype_sq_sum = sum(p * p for p in prototype)
        self.variance_term = (self.n * self.prototype_sq_sum
                              - self.prototype_sum * self.prototype_sum)
        self.cells = [(index // self.width, index % self.width, p)
                      for index, p in enumerate(prototype) if p > 0.0]


class _Candidate(NamedTuple):
    """One window a matcher accepted, before suppression."""
    score: float
    matcher: _ClassMatcher
    x: int
    y: int


def _scan(matcher: _ClassMatcher, bits: bytes, width: int,
          height: int, integral: list[int]) -> list[_Candidate]:
    """Every window of one matcher's size whose ink can match (the
    integral-image prefilter) and whose Pearson score clears the
    matcher's threshold — the one candidate proposer the trained
    detector and the legend-nn classifier share."""
    w, h, n = matcher.width, matcher.height, matcher.n
    if w > width or h > height:
        return []
    min_ink = max(1.0, _MIN_INK_RATIO * matcher.prototype_sum)
    max_ink = _MAX_INK_RATIO * matcher.prototype_sum
    pairs = [(dy * width + dx, p) for dy, dx, p in matcher.cells]
    prototype_sum = matcher.prototype_sum
    variance_term = matcher.variance_term
    threshold = matcher.threshold
    stride = width + 1
    candidates = []
    for y in range(height - h + 1):
        top = y * stride
        bottom = (y + h) * stride
        base = y * width
        for x in range(width - w + 1):
            ink = (integral[bottom + x + w] - integral[bottom + x]
                   - integral[top + x + w] + integral[top + x])
            if ink < min_ink or ink > max_ink or ink == n:
                continue
            anchor = base + x
            cross = sum(p for offset, p in pairs
                        if bits[anchor + offset])
            score = _correlation(n, cross, ink, prototype_sum,
                                 variance_term)
            if score >= threshold:
                candidates.append(_Candidate(score, matcher, x, y))
    return candidates


def _trimmed_bbox(matcher: _ClassMatcher, x: int, y: int) -> Bbox:
    """The matched window with the learned context ring (trained path)
    or the glyph crop's paper ring (legend-nn path) trimmed back off."""
    trim_x = min(_CONTEXT_MARGIN, (matcher.width - _MIN_SIDE) // 2)
    trim_y = min(_CONTEXT_MARGIN, (matcher.height - _MIN_SIDE) // 2)
    return (float(x + trim_x), float(y + trim_y),
            float(x + matcher.width - trim_x),
            float(y + matcher.height - trim_y))


class TrainedSymbolDetector:
    """The trained artifact behind the SymbolDetector seam: slides each
    prototype variant over the (normalized, binarized) raster, scores
    candidate windows by Pearson correlation against the learned ink
    probabilities, and suppresses overlapping reports. Detections carry
    the match score as confidence and the artifact's content-hash
    version in their provenance. Fully local. Direction is asserted only
    when an oriented variant matched — the evidence is the labeled
    orientation its training crops carried, never a guess."""

    def __init__(self, artifact_dir: "Path | str | None" = None):
        # the None default exists for the seam registry: a bare
        # "trained" selection constructs with no arguments and must fail
        # with the syntax hint, not a TypeError
        if not artifact_dir:
            raise ValueError(
                "the trained detector is selected with its artifact"
                " directory: 'trained:<artifact dir>'")
        self.artifact_dir = Path(artifact_dir)
        manifest_path = self.artifact_dir / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(
                f"{self.artifact_dir} holds no trained detector artifact"
                f" (no manifest.json); train one with"
                f" python -m pidgraph.detector")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != DETECTOR_FORMAT:
            raise ValueError(
                f"{manifest_path} is not a {DETECTOR_FORMAT} artifact"
                f" (format {manifest.get('format')!r})")

        prototypes: dict[str, bytes] = {}
        for symbol_class, entries in manifest["classes"].items():
            for entry in entries:
                width, height, pixels = decode_gray_png(
                    (self.artifact_dir / entry["prototype"]).read_bytes())
                if (width, height) != (entry["width"], entry["height"]):
                    raise ValueError(
                        f"prototype {entry['prototype']} is"
                        f" {width}x{height}, but the manifest declares"
                        f" {entry['width']}x{entry['height']}")
                prototypes[entry["prototype"]] = bytes(
                    255 - value for value in pixels)

        version = _artifact_version(manifest, prototypes)
        if version != manifest.get("version"):
            raise ValueError(
                f"artifact {self.artifact_dir} was modified after"
                f" training: its content hashes to {version}, but the"
                f" manifest is stamped {manifest.get('version')!r} —"
                f" retrain rather than editing weights")

        # scale is the contract: an artifact trained under a different
        # normalized scale would silently detect garbage, so it is
        # refused the way a foreign profile is
        trained_scale = manifest["normalization"]["target_long_side"]
        if trained_scale != TARGET_LONG_SIDE:
            raise ValueError(
                f"artifact {self.artifact_dir} was trained at the"
                f" normalized scale target_long_side={trained_scale},"
                f" but this pipeline normalizes to {TARGET_LONG_SIDE} —"
                f" retrain at the current scale")

        self.version = version
        self.profile = dict(manifest["profile"])
        self.component = f"symbol_detector:trained@{version}"
        self._matchers = [
            _ClassMatcher(symbol_class, entry,
                          prototypes[entry["prototype"]])
            for symbol_class in sorted(manifest["classes"])
            for entry in manifest["classes"][symbol_class]]

    @classmethod
    def from_options(cls, options: str) -> "TrainedSymbolDetector":
        return cls(options)

    def detect(self, sheet: Sheet,
               profile: ConventionProfile) -> list[SymbolDetection]:
        if sheet.raster is None:
            raise ValueError(
                f"{self.component} reads pixels; Sheet {sheet.number}"
                f" carries no raster")
        identity = profile.identity_record()
        if identity != self.profile:
            raise ValueError(
                f"artifact {self.artifact_dir} was trained for profile"
                f" {self.profile!r}; refusing to detect for"
                f" {identity!r} — per-company fine-tuning does not"
                f" transfer")

        width, height = sheet.width, sheet.height
        bits = _ink_bits(sheet.raster)
        integral = _integral_image(bits, width, height)
        candidates = []
        for matcher in self._matchers:
            candidates.extend(_scan(matcher, bits, width, height,
                                    integral))
        # best score wins ink; ties break by class then position, so the
        # detection list is deterministic
        candidates.sort(key=lambda c: (-c.score, c.matcher.symbol_class,
                                       c.y, c.x))

        kept: list[tuple] = []
        detections: list[SymbolDetection] = []
        for score, matcher, x, y in candidates:
            bbox = _trimmed_bbox(matcher, x, y)
            if _suppresses(kept, bbox):
                continue
            kept.append(bbox)
            detections.append(SymbolDetection(
                id=f"p{sheet.number}-sym{len(detections)}",
                sheet=sheet.number,
                symbol_class=matcher.symbol_class,
                bbox=bbox,
                confidence=min(1.0, score),
                provenance=Provenance(
                    component=self.component,
                    evidence=f"prototype {matcher.symbol_class}"
                             f" ({matcher.width}x{matcher.height})"
                             f" matched at ({x}, {y}) with score"
                             f" {score:.3f} (threshold"
                             f" {matcher.threshold})"),
                direction=matcher.direction,
            ))
        return detections


# --------------------------------------------------------------------------
# The Legend Dictionary nearest-neighbor classifier (ticket 17)

GLYPHS_DIR = "glyphs"

# The acceptance band is pinned with the classifier: there are no labeled
# crops to calibrate per-class thresholds from — that missing calibration
# data is exactly the onboarding cost this path avoids, priced here as
# fixed thresholds. A candidate scoring under the floor is not a symbol
# report at all; one landing between floor and acceptance is surfaced as
# UNCLASSIFIED_SYMBOL rather than force-assigned its nearest entry.
_NN_ACCEPT_SCORE = 0.6
_NN_CANDIDATE_FLOOR = 0.35

# A below-acceptance window offset from a real symbol correlates off that
# symbol's own ink; when this fraction of its ink is already inside kept
# detections' windows it is the same ink reported twice, not an unnamed
# symbol worth surfacing.
_NN_CLAIMED_INK = 0.5


def _box_ink(integral: list[int], width: int,
             box: tuple[int, int, int, int]) -> int:
    x0, y0, x1, y1 = box
    stride = width + 1
    return (integral[y1 * stride + x1] - integral[y1 * stride + x0]
            - integral[y0 * stride + x1] + integral[y0 * stride + x0])


def _mostly_claimed(integral: list[int], width: int,
                    window: tuple[int, int, int, int],
                    kept_windows: list[tuple[int, int, int, int]]) -> bool:
    """Whether a window's ink is mostly ink kept detections already
    explain (see _NN_CLAIMED_INK)."""
    wx0, wy0, wx1, wy1 = window
    total = _box_ink(integral, width, window)
    claimed = 0
    for kx0, ky0, kx1, ky1 in kept_windows:
        ix0, iy0 = max(wx0, kx0), max(wy0, ky0)
        ix1, iy1 = min(wx1, kx1), min(wy1, ky1)
        if ix1 > ix0 and iy1 > iy0:
            claimed += _box_ink(integral, width, (ix0, iy0, ix1, iy1))
            if claimed >= _NN_CLAIMED_INK * total:
                return True
    return False


class LegendNNSymbolDetector:
    """The candidate cheap adaptation mechanism behind the SymbolDetector
    seam: candidate windows proposed by the trained path's own sliding
    scan, each assigned its nearest Legend Dictionary entry — the
    Convention Profile bundle glyph with the highest Pearson correlation.
    Selected by configuration as 'legend-nn:<profile bundle dir>'.

    Confidence is the match score (distance = 1 - score), and provenance
    names the Legend Dictionary entry and glyph file matched. Glyph crops
    are taken at the pipeline's operating scale (the normalized frame)
    with the same 2 px paper ring the trained path's context margin
    learns; the emitted bbox trims the ring back off.

    FlowArrow glyphs are refused at load: a nearest-neighbor match
    against a static glyph asserts no orientation, and a direction is
    never guessed into the assembly stage."""

    def __init__(self, bundle_dir: "Path | str | None" = None):
        # the None default exists for the seam registry: a bare
        # "legend-nn" selection constructs with no arguments and must
        # fail with the syntax hint, not a TypeError
        if not bundle_dir:
            raise ValueError(
                "the legend-nn classifier is selected with its Convention"
                " Profile bundle: 'legend-nn:<profile bundle dir>'")
        self.bundle = Path(bundle_dir)
        profile = load_profile(self.bundle)
        self.profile = profile.identity_record()

        glyph_paths = sorted((self.bundle / GLYPHS_DIR).glob("*.png"))
        if not glyph_paths:
            raise ValueError(
                f"{self.bundle} holds no Legend Dictionary glyphs"
                f" ({GLYPHS_DIR}/*.png) — the legend-nn classifier"
                " matches candidates against the Legend Sheet's symbol"
                " crops")

        digest = hashlib.sha256()
        digest.update(json.dumps(self.profile, sort_keys=True,
                                 ensure_ascii=False).encode("utf-8"))
        self._matchers = []
        for path in glyph_paths:
            symbol_class = unquote(path.stem)
            entry = profile.legend.get(symbol_class)
            if entry is None:
                raise ValueError(
                    f"glyph {path.name} names {symbol_class!r}, which is"
                    f" not in Convention Profile {profile.name!r}'s"
                    " Legend Dictionary")
            if entry.role == "FlowArrow":
                raise ValueError(
                    f"glyph {path.name}: {symbol_class!r} is a FlowArrow"
                    " — a nearest-neighbor glyph match asserts no"
                    " orientation, and a direction is never guessed;"
                    " flow arrows need the trained path's oriented"
                    " prototypes")
            width, height, pixels = decode_gray_png(path.read_bytes())
            if width < _MIN_SIDE or height < _MIN_SIDE:
                raise ValueError(
                    f"glyph {path.name} is {width}x{height} — a glyph"
                    f" crop needs at least {_MIN_SIDE} px on each side")
            bits = _ink_bits(pixels)
            ink = sum(bits)
            if ink == 0 or ink == width * height:
                raise ValueError(
                    f"glyph {path.name} is uniform"
                    f" ({'all ink' if ink else 'all paper'}) — it"
                    " correlates with nothing and can never match")
            digest.update(f"{path.name}:{width}x{height}".encode("utf-8"))
            digest.update(pixels)
            self._matchers.append(_ClassMatcher(
                symbol_class,
                {"prototype": path.name, "width": width, "height": height,
                 "threshold": _NN_CANDIDATE_FLOOR, "direction": None},
                bytes(255 * bit for bit in bits)))

        self.version = digest.hexdigest()[:12]
        self.component = f"symbol_detector:legend-nn@{self.version}"

    @classmethod
    def from_options(cls, options: str) -> "LegendNNSymbolDetector":
        return cls(options)

    def detect(self, sheet: Sheet,
               profile: ConventionProfile) -> list[SymbolDetection]:
        if sheet.raster is None:
            raise ValueError(
                f"{self.component} reads pixels; Sheet {sheet.number}"
                f" carries no raster")
        identity = profile.identity_record()
        if identity != self.profile:
            raise ValueError(
                f"the Legend Dictionary glyphs under {self.bundle} belong"
                f" to profile {self.profile!r}; refusing to classify for"
                f" {identity!r} — a Convention Profile's glyphs do not"
                " transfer")

        width, height = sheet.width, sheet.height
        bits = _ink_bits(sheet.raster)
        integral = _integral_image(bits, width, height)
        candidates = []
        for matcher in self._matchers:
            candidates.extend(_scan(matcher, bits, width, height,
                                    integral))
        # the nearest neighbor wins ink: a window's best-correlating
        # glyph claims it before any farther entry; ties break by class
        # then position, so the detection list is deterministic
        candidates.sort(key=lambda c: (-c.score, c.matcher.symbol_class,
                                       c.y, c.x))

        kept: list[tuple] = []
        kept_windows: list[tuple[int, int, int, int]] = []
        detections: list[SymbolDetection] = []
        for score, matcher, x, y in candidates:
            bbox = _trimmed_bbox(matcher, x, y)
            if _suppresses(kept, bbox):
                continue
            window = (x, y, x + matcher.width, y + matcher.height)
            if (score < _NN_ACCEPT_SCORE
                    and _mostly_claimed(integral, width, window,
                                        kept_windows)):
                continue
            kept.append(bbox)
            kept_windows.append(window)
            nearest = (f"nearest Legend Dictionary entry"
                       f" {matcher.symbol_class!r} (glyph"
                       f" {matcher.prototype}) at ({x}, {y}): score"
                       f" {score:.3f}, distance {1 - score:.3f}")
            if score >= _NN_ACCEPT_SCORE:
                symbol_class = matcher.symbol_class
                evidence = f"{nearest} (accept >= {_NN_ACCEPT_SCORE})"
            else:
                symbol_class = UNCLASSIFIED_SYMBOL
                evidence = (f"{nearest}, below acceptance"
                            f" {_NN_ACCEPT_SCORE} — surfaced unclassified"
                            " rather than force-assigned")
            detections.append(SymbolDetection(
                id=f"p{sheet.number}-sym{len(detections)}",
                sheet=sheet.number,
                symbol_class=symbol_class,
                bbox=bbox,
                confidence=min(1.0, score),
                provenance=Provenance(component=self.component,
                                      evidence=evidence),
            ))
        return detections


# --------------------------------------------------------------------------
# CLI — training on demand, fully local; artifacts stay outside git

def main(argv: "list[str] | None" = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m pidgraph.detector",
        description="Train a symbol detector for one Convention Profile"
                    " from label-factory datasets — pass the clean set"
                    " and any degraded variant roots together. Select"
                    " the result behind the SymbolDetector seam as"
                    " 'trained:<artifact dir>'.")
    parser.add_argument("dataset_roots", type=Path, nargs="+",
                        help="label-factory dataset directories (each"
                             " holding labels/ and sheets/), e.g. the"
                             " factory out_dir and out_dir/variants/"
                             "<name>")
    parser.add_argument("--name", required=True,
                        help="Convention Profile identity")
    parser.add_argument("--version", required=True,
                        help="Convention Profile version")
    parser.add_argument("--out", type=Path, default=None,
                        help=f"artifact directory (default:"
                             f" {DEFAULT_OUT_DIR}/<name>@<version>,"
                             f" which git ignores)")
    args = parser.parse_args(argv)
    profile = {"name": args.name, "version": args.version}
    out = (args.out if args.out is not None
           else DEFAULT_OUT_DIR / profile_key(profile))
    manifest = train_detector(args.dataset_roots, profile, out)
    for symbol_class, variants in manifest["classes"].items():
        for entry in variants:
            oriented = ("" if entry["direction"] is None
                        else f" direction {entry['direction']}")
            print(f"  {symbol_class}{oriented}:"
                  f" {entry['examples']} examples,"
                  f" {entry['width']}x{entry['height']},"
                  f" threshold {entry['threshold']}")
    examples = sum(source["examples"] for source in manifest["sources"])
    print(f"trained detector {manifest['version']}"
          f" ({len(manifest['classes'])} classes, {examples} examples)"
          f" -> {out}")


if __name__ == "__main__":
    main()
