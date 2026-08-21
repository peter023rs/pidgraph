"""PNG round trip (ticket 15): the eval harness reads the Sheet rasters
the label factory persisted with encode_gray_png, so pngio gains the
matching decoder — 8-bit grayscale, non-interlaced, any row filter."""

import struct
import zlib

import pytest

from pidgraph.pngio import decode_gray_png, encode_gray_png


def _chunk(tag: bytes, payload: bytes) -> bytes:
    body = tag + payload
    return (struct.pack(">I", len(payload)) + body
            + struct.pack(">I", zlib.crc32(body)))


def _png(width: int, height: int, raw_rows: bytes,
         color_type: int = 0) -> bytes:
    ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr)
            + _chunk(b"IDAT", zlib.compress(raw_rows))
            + _chunk(b"IEND", b""))


def test_round_trip_recovers_the_raster():
    width, height = 5, 3
    pixels = bytes(range(width * height))
    encoded = encode_gray_png(width, height, pixels)
    assert decode_gray_png(encoded) == (width, height, pixels)


def test_round_trip_of_a_rendered_sheet_raster():
    from conftest import build_synthetic_sheet

    sheet = build_synthetic_sheet()
    encoded = encode_gray_png(sheet.width, sheet.height, sheet.raster)
    assert decode_gray_png(encoded) == (sheet.width, sheet.height,
                                        sheet.raster)


@pytest.mark.parametrize("filter_type,filtered,expected", [
    # width 3, height 2, values chosen so each filter's reconstruction
    # is hand-checkable against the PNG spec
    (1, b"\x01\x0a\x05\x05" b"\x01\x14\x05\x05",
        bytes([10, 15, 20, 20, 25, 30])),        # Sub: adds left
    (2, b"\x00\x0a\x0f\x14" b"\x02\x05\x05\x05",
        bytes([10, 15, 20, 15, 20, 25])),        # Up: adds above
    (3, b"\x03\x0a\x0f\x14" b"\x03\x0a\x0f\x14",
        bytes([10, 20, 30, 15, 32, 51])),        # Average of left/above
    (4, b"\x00\x0a\x0f\x14" b"\x04\x05\x05\x05",
        bytes([10, 15, 20, 15, 20, 25])),        # Paeth: above, then left
])
def test_filtered_rows_decode(filter_type, filtered, expected):
    assert decode_gray_png(_png(3, 2, filtered)) == (3, 2, expected)


def test_multiple_idat_chunks_concatenate():
    width, height = 4, 2
    pixels = bytes(range(8))
    rows = zlib.compress(
        b"".join(b"\x00" + pixels[y * width:(y + 1) * width]
                 for y in range(height)))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    split = (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr)
             + _chunk(b"IDAT", rows[:5]) + _chunk(b"IDAT", rows[5:])
             + _chunk(b"IEND", b""))
    assert decode_gray_png(split) == (width, height, pixels)


def test_refuses_non_grayscale():
    rows = b"\x00" + bytes(9)  # one RGB row, filter 0
    with pytest.raises(ValueError, match="grayscale"):
        decode_gray_png(_png(1, 1, rows, color_type=2))


def test_refuses_a_non_png():
    with pytest.raises(ValueError, match="PNG"):
        decode_gray_png(b"not a png at all")


def test_refuses_a_corrupted_chunk():
    encoded = bytearray(encode_gray_png(2, 2, bytes(4)))
    encoded[-5] ^= 0xFF  # flip a bit inside IEND's CRC
    with pytest.raises(ValueError, match="CRC"):
        decode_gray_png(bytes(encoded))


def test_refuses_a_truncated_file():
    encoded = encode_gray_png(4, 4, bytes(16))
    with pytest.raises(ValueError, match="middle of a chunk"):
        decode_gray_png(encoded[:-7])


def test_refuses_pixel_data_that_does_not_decompress():
    ihdr = struct.pack(">IIBBBBB", 2, 1, 8, 0, 0, 0, 0)
    broken = (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr)
              + _chunk(b"IDAT", b"not a zlib stream")
              + _chunk(b"IEND", b""))
    with pytest.raises(ValueError, match="decompress"):
        decode_gray_png(broken)
