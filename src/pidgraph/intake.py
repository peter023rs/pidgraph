"""PDF Document intake: enumerate Sheets, refuse vector PDFs, rasterize.

v1 intake is raster-only (spec decision). Every page is classified from
its content stream before any output exists: a page whose only marks are
image XObjects is a scan; any path-painting or text-showing operator is
vector line work, and one vector Sheet refuses the whole Document — v1
never silently produces raster-quality output for a Document that
deserves the deterministic path.

"Sheet" everywhere; "page" only for PDF mechanics.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from pypdf import PdfReader
from pypdf.generic import ContentStream, StreamObject

from .model import Document, Sheet

# Operators that mark the page with vector content: path painting (stroke
# and fill; "n" ends a path without marking, so clipping stays neutral)
# and text showing.
_VECTOR_OPERATORS = {b"S", b"s", b"f", b"F", b"f*", b"B", b"B*", b"b",
                     b"b*", b"sh", b"Tj", b"TJ", b"'", b'"'}


class IntakeError(Exception):
    """A Document pidgraph cannot take in, with the reason spelled out."""


class VectorPdfRefusal(IntakeError):
    """The Document is a vector PDF, which v1 deliberately refuses."""


def _xobjects(resources) -> dict:
    if resources is None:
        return {}
    resources = resources.get_object()
    xobjects = resources.get("/XObject")
    return {} if xobjects is None else dict(xobjects.get_object())


def _inspect_stream(stream, resources, reader,
                    depth: int = 0) -> tuple[bool, list[StreamObject]]:
    """Walk one content stream: does it paint vector marks, and which
    image XObjects does it draw? Form XObjects are walked recursively."""
    has_vector = False
    images: list[StreamObject] = []
    xobjects = _xobjects(resources)
    for operands, operator in ContentStream(stream, reader).operations:
        if operator in _VECTOR_OPERATORS:
            has_vector = True
        elif operator == b"Do" and operands:
            drawn = xobjects.get(str(operands[0]))
            if drawn is None:
                continue
            drawn = drawn.get_object()
            subtype = drawn.get("/Subtype")
            if subtype == "/Image":
                images.append(drawn)
            elif subtype == "/Form" and depth < 8:
                vector, nested = _inspect_stream(
                    drawn, drawn.get("/Resources"), reader, depth + 1)
                has_vector = has_vector or vector
                images.extend(nested)
        # BI..EI inline images (pypdf operator b"INLINE IMAGE") are raster
        # but not extractable in v1; a page relying on one fails the
        # one-scanned-image check rather than being mislabeled vector.
    return has_vector, images


def _image_refusal(name: str, number: int, why: str) -> IntakeError:
    return IntakeError(f"Document '{name}' refused at intake: the scanned "
                       f"image on Sheet {number} {why}")


def _extract_raster(image: StreamObject, number: int,
                    name: str) -> tuple[int, int, bytes]:
    filters = image.get("/Filter", [])
    if not isinstance(filters, list):
        filters = [filters]
    filters = [str(f) for f in filters]
    if any(f not in ("/FlateDecode",) for f in filters):
        raise _image_refusal(
            name, number,
            f"uses image encoding {', '.join(filters)}, which v1 intake "
            f"cannot decode yet (supported: uncompressed or FlateDecode "
            f"grayscale)")
    colorspace = str(image.get("/ColorSpace"))
    bits = int(image.get("/BitsPerComponent", 8))
    if colorspace != "/DeviceGray" or bits != 8:
        raise _image_refusal(
            name, number,
            f"is {colorspace} at {bits} bit(s) per component; v1 intake "
            f"reads 8-bit DeviceGray scans")
    width = cast(int, image["/Width"])
    height = cast(int, image["/Height"])
    raster = image.get_data()
    if len(raster) != width * height:
        raise _image_refusal(
            name, number,
            f"decodes to {len(raster)} bytes, expected {width}x{height} "
            f"= {width * height}")
    return width, height, raster


def load_document(pdf_path: Path | str) -> Document:
    """Read a PDF into a Document of rasterized Sheets, or refuse it.

    Classification of every Sheet happens before any raster is extracted,
    so a refused run produces no partial output.
    """
    path = Path(pdf_path)
    reader = PdfReader(path)

    vector_sheets: list[int] = []
    images_per_sheet: list[list[StreamObject]] = []
    for number, page in enumerate(reader.pages, start=1):
        has_vector = False
        images: list[StreamObject] = []
        contents = page.get_contents()
        if contents is not None:
            has_vector, images = _inspect_stream(
                contents, page.get("/Resources"), reader)
        if has_vector:
            vector_sheets.append(number)
        images_per_sheet.append(images)

    if vector_sheets:
        listed = ", ".join(str(n) for n in vector_sheets)
        noun, verb = (("Sheets", "contain") if len(vector_sheets) > 1
                      else ("Sheet", "contains"))
        raise VectorPdfRefusal(
            f"Document '{path.name}' refused at intake: {noun} {listed} "
            f"{verb} vector line work or text, so this is a vector PDF, "
            f"not a scan. pidgraph v1 digitizes scanned Documents only — "
            f"extract this Document with a deterministic vector path "
            f"(hazop-ai) instead of rasterizing it")

    sheets = []
    for number, images in enumerate(images_per_sheet, start=1):
        if len(images) != 1:
            raise IntakeError(
                f"Document '{path.name}' refused at intake: expected one "
                f"scanned image on Sheet {number}, found {len(images)}")
        width, height, raster = _extract_raster(images[0], number,
                                                path.name)
        sheets.append(Sheet(number=number, width=width, height=height,
                            raster=raster))
    return Document(name=path.name, sheets=tuple(sheets))
