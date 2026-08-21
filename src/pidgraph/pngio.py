"""Grayscale PNG encoding and decoding with the stdlib alone, so
persisting Sheet rasters as run artifacts (for the Review Workbench
overlay, ticket 10) and reading them back (for the eval harness,
ticket 15) add no imaging dependency."""

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


def _chunks(data: bytes):
    offset = len(_SIGNATURE)
    while offset < len(data):
        if offset + 12 > len(data):
            raise ValueError("PNG data ends in the middle of a chunk")
        (length,) = struct.unpack_from(">I", data, offset)
        if offset + 12 + length > len(data):
            raise ValueError("PNG data ends in the middle of a chunk")
        tag = data[offset + 4:offset + 8]
        payload = data[offset + 8:offset + 8 + length]
        (crc,) = struct.unpack_from(">I", data, offset + 8 + length)
        if crc != zlib.crc32(tag + payload):
            raise ValueError(f"PNG chunk {tag!r} fails its CRC check")
        yield tag, payload
        offset += 12 + length


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def decode_gray_png(data: bytes) -> tuple[int, int, bytes]:
    """Decode a Sheet-raster PNG — 8-bit grayscale, non-interlaced, any
    row filter — back to (width, height, row-major pixels), the inverse
    of encode_gray_png."""
    if not data.startswith(_SIGNATURE):
        raise ValueError("not a PNG: missing the PNG signature")
    ihdr = None
    idat = bytearray()
    for tag, payload in _chunks(data):
        if tag == b"IHDR":
            ihdr = payload
        elif tag == b"IDAT":
            idat.extend(payload)
    if ihdr is None or not idat:
        raise ValueError("not a PNG: no IHDR or no IDAT chunk")
    width, height, depth, color, _, _, interlace = struct.unpack(
        ">IIBBBBB", ihdr)
    if (depth, color) != (8, 0):
        raise ValueError(
            f"Sheet rasters are 8-bit grayscale PNGs; got bit depth"
            f" {depth}, color type {color}")
    if interlace != 0:
        raise ValueError("Sheet rasters are non-interlaced PNGs")

    try:
        raw = zlib.decompress(bytes(idat))
    except zlib.error as error:
        raise ValueError(
            f"PNG pixel data does not decompress: {error}") from None
    if len(raw) != (width + 1) * height:
        raise ValueError(
            f"PNG pixel data of {len(raw)} bytes does not fill"
            f" {width}x{height} plus one filter byte per row")
    pixels = bytearray(width * height)
    for y in range(height):
        row = raw[y * (width + 1):(y + 1) * (width + 1)]
        filter_type, filtered = row[0], row[1:]
        out = y * width
        for x, value in enumerate(filtered):
            a = pixels[out + x - 1] if x else 0                 # left
            b = pixels[out + x - width] if y else 0             # above
            c = pixels[out + x - width - 1] if x and y else 0   # above-left
            if filter_type == 1:
                value += a
            elif filter_type == 2:
                value += b
            elif filter_type == 3:
                value += (a + b) // 2
            elif filter_type == 4:
                value += _paeth(a, b, c)
            elif filter_type != 0:
                raise ValueError(f"PNG row {y} carries unknown filter"
                                 f" type {filter_type}")
            pixels[out + x] = value & 0xFF
    return width, height, bytes(pixels)
