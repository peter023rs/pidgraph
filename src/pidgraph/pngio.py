"""Grayscale PNG encoding with the stdlib alone, so persisting Sheet
rasters as run artifacts (for the Review Workbench overlay, ticket 10)
adds no imaging dependency."""

from __future__ import annotations

import struct
import zlib

_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _chunk(tag: bytes, payload: bytes) -> bytes:
    body = tag + payload
    return (struct.pack(">I", len(payload)) + body
            + struct.pack(">I", zlib.crc32(body)))


def encode_gray_png(width: int, height: int, pixels: bytes) -> bytes:
    """Encode a row-major 8-bit grayscale raster (the Sheet.raster form)
    as a PNG: color type 0, bit depth 8, filter None on every row."""
    if len(pixels) != width * height:
        raise ValueError(
            f"raster of {len(pixels)} bytes does not fill"
            f" {width}x{height}")
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    rows = b"".join(b"\x00" + pixels[y * width:(y + 1) * width]
                    for y in range(height))
    return (_SIGNATURE + _chunk(b"IHDR", ihdr)
            + _chunk(b"IDAT", zlib.compress(rows)) + _chunk(b"IEND", b""))
