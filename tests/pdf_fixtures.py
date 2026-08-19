"""Programmatically generated PDF fixtures for intake tests.

A minimal stdlib-only PDF writer: enough structure for pypdf to parse.
Two page styles, mirroring what intake must tell apart:

  scanned — the page's only mark is one full-page Flate-compressed
            8-bit DeviceGray image XObject (what a scanner emits)
  vector  — the page strokes paths and shows text (what a CAD export
            emits)

No real drawings enter git: fixtures are written into pytest tmp dirs.
"""

from __future__ import annotations

import zlib
from pathlib import Path


def _stream(dict_entries: str, data: bytes) -> bytes:
    return (f"<<{dict_entries}/Length {len(data)}>>\nstream\n".encode()
            + data + b"\nendstream")


def _assemble(objects: list[bytes]) -> bytes:
    """objects[i] is the body of object i+1; returns the full PDF file."""
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (f"trailer\n<</Size {len(objects) + 1}/Root 1 0 R>>\n"
            f"startxref\n{xref_at}\n%%EOF\n").encode()
    return bytes(out)


def _draw_image_contents(width: int, height: int) -> bytes:
    return _stream("", f"q {width} 0 0 {height} 0 0 cm /Im0 Do Q".encode())


def _image_xobject(width: int, height: int, encoded: bytes,
                   filter_name: str) -> bytes:
    return _stream(
        f"/Type/XObject/Subtype/Image/Width {width}/Height {height}"
        f"/ColorSpace/DeviceGray/BitsPerComponent 8/Filter{filter_name}",
        encoded)


def _image_page(width: int, height: int, encoded: bytes, filter_name: str,
                page_obj: int) -> tuple[bytes, list[bytes]]:
    contents_obj, image_obj = page_obj + 1, page_obj + 2
    page = (f"<</Type/Page/Parent 2 0 R/MediaBox[0 0 {width} {height}]"
            f"/Resources<</XObject<</Im0 {image_obj} 0 R>>>>"
            f"/Contents {contents_obj} 0 R>>").encode()
    return page, [_draw_image_contents(width, height),
                  _image_xobject(width, height, encoded, filter_name)]


def _blank_page(page_obj: int) -> tuple[bytes, list[bytes]]:
    contents_obj = page_obj + 1
    page = (f"<</Type/Page/Parent 2 0 R/MediaBox[0 0 400 200]"
            f"/Contents {contents_obj} 0 R>>").encode()
    return page, [_stream("", b"")]


def _vector_page(page_obj: int) -> tuple[bytes, list[bytes]]:
    contents_obj = page_obj + 1
    page = (f"<</Type/Page/Parent 2 0 R/MediaBox[0 0 400 200]"
            f"/Resources<</Font<</F1 3 0 R>>>>"
            f"/Contents {contents_obj} 0 R>>").encode()
    contents = _stream("", b"1 w 20 20 m 200 20 l S 30 40 100 50 re S "
                           b"BT /F1 12 Tf 50 150 Td (T-101) Tj ET")
    return page, [contents]


def _vector_ops_page(width: int, height: int, ops: bytes,
                     page_obj: int) -> tuple[bytes, list[bytes]]:
    """A vector page drawing caller-chosen content-stream operators —
    for tests that need exactly known geometry (the label factory's
    projection fidelity, ticket 13)."""
    contents_obj = page_obj + 1
    page = (f"<</Type/Page/Parent 2 0 R/MediaBox[0 0 {width} {height}]"
            f"/Resources<</Font<</F1 3 0 R>>>>"
            f"/Contents {contents_obj} 0 R>>").encode()
    return page, [_stream("", ops)]


def build_pdf(pages: list[tuple]) -> bytes:
    """Page specs:
      ("scanned", width, height, raster)                 gray Flate image
      ("raw_image", width, height, encoded, "/Filter")   undecodable image
      ("vector",)                                        paths + text
      ("vector_ops", width, height, ops)                 caller-drawn page
      ("blank",)                                         no marks at all
    """
    # 1 catalog, 2 pages (Kids filled in below), 3 font, then per page
    objects: list[bytes] = [
        b"",
        b"",
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    ]
    kids = []
    for spec in pages:
        page_obj = len(objects) + 1
        kids.append(f"{page_obj} 0 R")
        if spec[0] == "scanned":
            _, width, height, raster = spec
            page, extra = _image_page(width, height, zlib.compress(raster),
                                      "/FlateDecode", page_obj)
        elif spec[0] == "raw_image":
            _, width, height, encoded, filter_name = spec
            page, extra = _image_page(width, height, encoded, filter_name,
                                      page_obj)
        elif spec[0] == "blank":
            page, extra = _blank_page(page_obj)
        elif spec[0] == "vector_ops":
            _, width, height, ops = spec
            page, extra = _vector_ops_page(width, height, ops, page_obj)
        else:
            page, extra = _vector_page(page_obj)
        objects.append(page)
        objects.extend(extra)
    objects[0] = b"<</Type/Catalog/Pages 2 0 R>>"
    objects[1] = (f"<</Type/Pages/Kids[{' '.join(kids)}]"
                  f"/Count {len(pages)}>>").encode()
    return _assemble(objects)


def write_pdf(path: Path, pages: list[tuple]) -> Path:
    path.write_bytes(build_pdf(pages))
    return path


def write_inherited_resources_pdf(path: Path, width: int, height: int,
                                  raster: bytes) -> Path:
    """One scanned page whose /Resources live on the /Pages node — legal
    PDF inheritance that real scanners produce."""
    objects = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1"
        b"/Resources<</XObject<</Im0 5 0 R>>>>>>",
        f"<</Type/Page/Parent 2 0 R/MediaBox[0 0 {width} {height}]"
        f"/Contents 4 0 R>>".encode(),
        _draw_image_contents(width, height),
        _image_xobject(width, height, zlib.compress(raster),
                       "/FlateDecode"),
    ]
    path.write_bytes(_assemble(objects))
    return path
