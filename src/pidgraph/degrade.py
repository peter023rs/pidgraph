"""Synthetic degradation of rendered Sheets (ticket 14): blur, skew,
noise, and compression over clean label-factory renders, so models
trained on clean renders survive real scans.

Transforms operate on the Sheet raster form (row-major 8-bit grayscale
bytes) with the stdlib alone, like pngio — degradation is dev-time
tooling and must not push an imaging dependency into the product. Each
transform is configurable in severity, and a sequence composes into one
degradation. All transforms preserve the canvas size, a deliberate
invariant: a degraded variant stays pixel-for-pixel comparable with its
clean Sheet.

Geometry is the contract: the one geometric transform (skew) exposes
the exact forward point map it rasterizes with, and degrade_bbox /
degrade_polyline push label geometry through the same map — so boxes
and text spans stay aligned with the degraded raster. Photometric
transforms (blur, noise, compression) leave geometry untouched.

Reproducibility: every stochastic draw comes from a random.Random
seeded with (seed, stage index, transform kind), so the same input,
transform configuration, and seed reproduce a dataset exactly —
insertion of a stage never silently reuses another stage's stream.
random.Random's string seeding and gauss are stable across CPython
versions; float math makes bit-exactness a same-platform guarantee.

"Compression" is the JPEG artifact model without the codec: 8x8 block
DCT, quantized by the standard luminance table at an IJG-scaled
quality, then inverted — blocking and ringing without entropy coding,
since only the artifacts matter for training. Pure-Python DCT is slow
on full Sheets (seconds per Sheet) but this never runs in CI; tests
use tiny rasters under the offline invariant.
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, ClassVar, Mapping, Sequence


@dataclass(frozen=True)
class Raster:
    """One grayscale raster: the Sheet.raster / PageRender.pixels form."""
    width: int
    height: int
    pixels: bytes

    def __post_init__(self) -> None:
        if len(self.pixels) != self.width * self.height:
            raise ValueError(
                f"raster of {len(self.pixels)} bytes does not fill"
                f" {self.width}x{self.height}")


def _check_number(value: object, what: str) -> None:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value)):
        raise ValueError(f"{what}, got {value!r}")


# --------------------------------------------------------------------------
# Blur: separable box blur, clamp-at-edge, integer rounding

def _blurred_line(line: bytes, radius: int) -> bytearray:
    size = 2 * radius + 1
    length = len(line)
    out = bytearray(length)
    total = line[0] * (radius + 1)
    for i in range(1, radius + 1):
        total += line[min(i, length - 1)]
    for x in range(length):
        out[x] = (total + radius) // size
        total += line[min(x + radius + 1, length - 1)]
        total -= line[max(x - radius, 0)]
    return out


def _blur_rows(pixels: bytes, width: int, height: int,
               radius: int) -> bytes:
    out = bytearray(width * height)
    for y in range(height):
        out[y * width:(y + 1) * width] = _blurred_line(
            pixels[y * width:(y + 1) * width], radius)
    return bytes(out)


def _transposed(pixels: bytes, width: int, height: int) -> bytes:
    return bytes(pixels[y * width + x]
                 for x in range(width) for y in range(height))


@dataclass(frozen=True)
class Blur:
    """Box blur; severity is the radius in pixels (window 2r+1)."""
    radius: int = 1
    kind: ClassVar[str] = "blur"

    def __post_init__(self) -> None:
        if (isinstance(self.radius, bool)
                or not isinstance(self.radius, int) or self.radius < 1):
            raise ValueError(f"blur radius is a positive integer of"
                             f" pixels, got {self.radius!r}")

    def map_point(self, x: float, y: float, width: int,
                  height: int) -> tuple[float, float]:
        return (x, y)

    def apply(self, raster: Raster, rng: random.Random) -> Raster:
        rows = _blur_rows(raster.pixels, raster.width, raster.height,
                          self.radius)
        columns = _blur_rows(_transposed(rows, raster.width, raster.height),
                             raster.height, raster.width, self.radius)
        return Raster(raster.width, raster.height,
                      _transposed(columns, raster.height, raster.width))


# --------------------------------------------------------------------------
# Skew: rotation about the raster center — the scanner misfeed model.
# The raster is resampled through the inverse map (bilinear, paper-white
# outside the page); labels go through the identical forward map.

@dataclass(frozen=True)
class Skew:
    """Rotation about the raster center; severity is the angle in
    degrees (positive turns the drawing clockwise on the page)."""
    angle_deg: float = 1.0
    kind: ClassVar[str] = "skew"

    def __post_init__(self) -> None:
        _check_number(self.angle_deg,
                      "skew angle_deg is a finite number of degrees")

    def map_point(self, x: float, y: float, width: int,
                  height: int) -> tuple[float, float]:
        theta = math.radians(self.angle_deg)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        cx, cy = width / 2, height / 2
        dx, dy = x - cx, y - cy
        return (cx + dx * cos_t - dy * sin_t,
                cy + dx * sin_t + dy * cos_t)

    def apply(self, raster: Raster, rng: random.Random) -> Raster:
        theta = math.radians(self.angle_deg)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        width, height = raster.width, raster.height
        cx, cy = width / 2, height / 2
        pixels = raster.pixels
        out = bytearray(width * height)
        floor = math.floor
        for oy in range(height):
            dy = oy + 0.5 - cy
            # the inverse rotation of this output pixel center; the
            # source point is affine in ox, so step instead of remapping
            sx = cx + (0.5 - cx) * cos_t + dy * sin_t
            sy = cy - (0.5 - cx) * sin_t + dy * cos_t
            base = oy * width
            for ox in range(width):
                u, v = sx - 0.5, sy - 0.5
                x0, y0 = floor(u), floor(v)
                fx, fy = u - x0, v - y0
                value = 0.0
                for yi, wy in ((y0, 1.0 - fy), (y0 + 1, fy)):
                    if wy == 0.0:
                        continue
                    row = yi * width
                    for xi, wx in ((x0, 1.0 - fx), (x0 + 1, fx)):
                        if wx == 0.0:
                            continue
                        inside = 0 <= xi < width and 0 <= yi < height
                        value += wy * wx * (pixels[row + xi] if inside
                                            else 255)
                out[base + ox] = min(255, int(value + 0.5))
                sx += cos_t
                sy -= sin_t
        return Raster(width, height, bytes(out))


# --------------------------------------------------------------------------
# Noise: additive gaussian per pixel, clamped to the byte range

@dataclass(frozen=True)
class Noise:
    """Additive gaussian noise; severity is sigma in gray levels."""
    sigma: float = 8.0
    kind: ClassVar[str] = "noise"

    def __post_init__(self) -> None:
        _check_number(self.sigma,
                      "noise sigma is a positive number of gray levels")
        if self.sigma <= 0:
            raise ValueError(f"noise sigma is a positive number of gray"
                             f" levels, got {self.sigma!r}")

    def map_point(self, x: float, y: float, width: int,
                  height: int) -> tuple[float, float]:
        return (x, y)

    def apply(self, raster: Raster, rng: random.Random) -> Raster:
        gauss, sigma = rng.gauss, self.sigma
        out = bytearray(len(raster.pixels))
        for i, value in enumerate(raster.pixels):
            noisy = round(value + gauss(0.0, sigma))
            out[i] = 0 if noisy < 0 else 255 if noisy > 255 else noisy
        return Raster(raster.width, raster.height, bytes(out))


# --------------------------------------------------------------------------
# Compression: JPEG-style 8x8 DCT quantization round trip

_BLOCK = 8

# the JPEG Annex K luminance quantization table
_BASE_QUANT = (
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99)

_DCT = [[math.sqrt((1 if u == 0 else 2) / _BLOCK)
         * math.cos((2 * x + 1) * u * math.pi / (2 * _BLOCK))
         for x in range(_BLOCK)] for u in range(_BLOCK)]
_DCT_T = [list(column) for column in zip(*_DCT)]


def _matmul(a: list[list[float]],
            b: list[list[float]]) -> list[list[float]]:
    return [[sum(row[k] * b[k][j] for k in range(_BLOCK))
             for j in range(_BLOCK)] for row in a]


def _quant_table(quality: int) -> list[int]:
    # the IJG quality scaling: 50 is the table as published
    scale = 5000 // quality if quality < 50 else 200 - 2 * quality
    return [max(1, min(255, (base * scale + 50) // 100))
            for base in _BASE_QUANT]


@dataclass(frozen=True)
class Compression:
    """JPEG-style quantization artifacts; severity is the IJG quality
    (1 crushes, 100 is near-lossless). Edge blocks pad by replication."""
    quality: int = 30
    kind: ClassVar[str] = "compression"

    def __post_init__(self) -> None:
        if (isinstance(self.quality, bool)
                or not isinstance(self.quality, int)
                or not 1 <= self.quality <= 100):
            raise ValueError(f"compression quality is an integer in"
                             f" 1..100, got {self.quality!r}")

    def map_point(self, x: float, y: float, width: int,
                  height: int) -> tuple[float, float]:
        return (x, y)

    def apply(self, raster: Raster, rng: random.Random) -> Raster:
        width, height = raster.width, raster.height
        quant = _quant_table(self.quality)
        pixels = raster.pixels
        out = bytearray(width * height)
        for by in range(0, height, _BLOCK):
            for bx in range(0, width, _BLOCK):
                block = [[float(pixels[min(by + y, height - 1) * width
                                       + min(bx + x, width - 1)] - 128)
                          for x in range(_BLOCK)] for y in range(_BLOCK)]
                coeffs = _matmul(_matmul(_DCT, block), _DCT_T)
                for u in range(_BLOCK):
                    row = coeffs[u]
                    for v in range(_BLOCK):
                        q = quant[u * _BLOCK + v]
                        row[v] = round(row[v] / q) * q
                restored = _matmul(_matmul(_DCT_T, coeffs), _DCT)
                for y in range(min(_BLOCK, height - by)):
                    base = (by + y) * width + bx
                    row = restored[y]
                    for x in range(min(_BLOCK, width - bx)):
                        value = int(row[x] + 128.5)
                        out[base + x] = (0 if value < 0
                                         else 255 if value > 255
                                         else value)
        return Raster(width, height, bytes(out))


Transform = Blur | Skew | Noise | Compression

_TRANSFORM_KINDS: dict[str, type[Transform]] = {
    cls.kind: cls for cls in (Blur, Skew, Noise, Compression)}


# --------------------------------------------------------------------------
# Composition: one degradation is a transform sequence plus a seed

def degrade_raster(raster: Raster, transforms: Sequence[Transform],
                   seed: int | str) -> Raster:
    """Apply the transforms in order. Each stage draws from its own
    stream keyed by (seed, stage index, kind), so a dataset is exactly
    reproducible from its configuration and seed."""
    for index, transform in enumerate(transforms):
        rng = random.Random(f"{seed}/{index}/{transform.kind}")
        raster = transform.apply(raster, rng)
    return raster


def map_point(transforms: Sequence[Transform], x: float, y: float,
              width: int, height: int) -> tuple[float, float]:
    """The composed forward point map of the sequence — exactly the
    geometry the degraded raster was resampled through."""
    for transform in transforms:
        x, y = transform.map_point(x, y, width, height)
    return (x, y)


def degrade_bbox(transforms: Sequence[Transform],
                 bbox: Sequence[float], width: int,
                 height: int) -> list[float]:
    """A label box under the composed map: the axis-aligned box
    enclosing the mapped corners, so it still surrounds its ink."""
    x0, y0, x1, y1 = bbox
    corners = [map_point(transforms, x, y, width, height)
               for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    return [min(x for x, _ in corners), min(y for _, y in corners),
            max(x for x, _ in corners), max(y for _, y in corners)]


def degrade_polyline(transforms: Sequence[Transform],
                     polyline: Sequence[Sequence[float]], width: int,
                     height: int) -> list[list[float]]:
    return [list(map_point(transforms, x, y, width, height))
            for x, y in polyline]


def clip_bbox(bbox: Sequence[float], width: int,
              height: int) -> list[float] | None:
    """The box intersected with the canvas — or None when the canvas-
    preserving transforms pushed it entirely off the page: its ink was
    resampled away, and a label must never assert invisible ink."""
    x0, y0, x1, y1 = bbox
    clipped = [max(x0, 0.0), max(y0, 0.0),
               min(x1, float(width)), min(y1, float(height))]
    if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
        return None
    return clipped


def polyline_visible(polyline: Sequence[Sequence[float]], width: int,
                     height: int) -> bool:
    """Whether any of the polyline's extent overlaps the canvas — the
    coarse box test; a run whose whole extent left the page has no ink
    left to label."""
    xs = [x for x, _ in polyline]
    ys = [y for _, y in polyline]
    return (min(xs) < width and max(xs) > 0
            and min(ys) < height and max(ys) > 0)


# --------------------------------------------------------------------------
# Configuration: transforms and dataset variants as JSON-shaped dicts

def transform_config(transform: Transform) -> dict:
    """The transform as a plain dict, echoed into run manifests so a
    dataset states the exact configuration that degraded it."""
    return {"kind": transform.kind,
            **{f.name: getattr(transform, f.name)
               for f in fields(transform)}}


def transform_from_config(config: Mapping[str, Any]) -> Transform:
    params = dict(config)
    kind = params.pop("kind", None)
    cls = _TRANSFORM_KINDS.get(kind)  # type: ignore[arg-type]
    if cls is None:
        raise ValueError(f"unknown degradation kind {kind!r}; kinds are"
                         f" {sorted(_TRANSFORM_KINDS)}")
    allowed = {f.name for f in fields(cls)}
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise ValueError(f"{kind} severity is configured by"
                         f" {sorted(allowed)}, not {unknown}")
    return cls(**params)


_VARIANT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


@dataclass(frozen=True)
class Variant:
    """One degraded dataset the factory emits beside the clean one:
    a name (its directory under the out_dir), a transform sequence,
    and the seed the whole variant reproduces from."""
    name: str
    transforms: tuple[Transform, ...]
    seed: int = 0

    def __post_init__(self) -> None:
        if (not isinstance(self.name, str)
                or not _VARIANT_NAME.fullmatch(self.name)):
            raise ValueError(
                f"a variant name is a path-safe token: an ASCII letter"
                f" or digit, then letters, digits, '.', '_' or '-';"
                f" got {self.name!r}")
        if not self.transforms:
            raise ValueError(f"variant {self.name!r} has no transforms —"
                             f" an undegraded variant is the clean set")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError(f"a variant seed is an integer,"
                             f" got {self.seed!r}")


def variant_from_config(config: Mapping[str, Any]) -> Variant:
    unknown = sorted(set(config) - {"name", "seed", "transforms"})
    if unknown:
        raise ValueError(f"a variant is configured by ['name', 'seed',"
                         f" 'transforms'], not {unknown}")
    transforms = tuple(transform_from_config(entry)
                       for entry in config.get("transforms") or ())
    return Variant(name=config.get("name"),  # type: ignore[arg-type]
                   transforms=transforms, seed=config.get("seed", 0))


def load_variants(path: Path | str) -> list[Variant]:
    """Variants from a JSON file: a non-empty list of variant configs.
    An empty or non-list file is refused by name — a typo'd path must
    not silently yield an undegraded run."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path} does not hold a non-empty JSON list"
                         f" of degradation variants")
    return [variant_from_config(entry) for entry in data]
