"""Synthetic fixture Sheet: programmatically drawn, exactly known symbols,
lines, and tag strings.

Process topology (all coordinates fixed):

    T-101 (tank) --run1--> V-101 (gate valve) --run2-- T-102 (tank)
        run1 carries a flow arrow (direction evidence) and line number
        150-GA-001; run2 has no direction evidence.
    T-102 --run3-- off-page connector DW02-0003 (an L with one corner).
    PI-100 (instrument bubble) sits above the valve, unconnected.
"""

from pathlib import Path

import pytest

from pidgraph.model import (
    ConventionProfile,
    Document,
    LineAnnotation,
    Sheet,
    SheetAnnotations,
    SymbolAnnotation,
    TextAnnotation,
)
from pidgraph.profile import load_profile

SHEET_W, SHEET_H = 400, 200

SYMBOLS = (
    SymbolAnnotation("tank", (20.0, 80.0, 60.0, 140.0)),
    SymbolAnnotation("nozzle", (58.0, 106.0, 66.0, 114.0)),
    SymbolAnnotation("gate_valve", (180.0, 100.0, 200.0, 120.0)),
    SymbolAnnotation("tank", (320.0, 80.0, 360.0, 140.0)),
    SymbolAnnotation("instrument_bubble", (180.0, 30.0, 200.0, 50.0)),
    SymbolAnnotation("opc", (325.0, 165.0, 355.0, 185.0)),
    SymbolAnnotation("flow_arrow", (115.0, 105.0, 125.0, 115.0),
                     direction=(1.0, 0.0)),
)

LINES = (
    LineAnnotation(((66.0, 110.0), (180.0, 110.0))),                  # run1
    LineAnnotation(((200.0, 110.0), (320.0, 110.0))),                 # run2
    LineAnnotation(((340.0, 140.0), (340.0, 152.0), (340.0, 165.0))), # run3
)

TEXTS = (
    TextAnnotation("T-101", "equipment_tag", (20.0, 60.0, 60.0, 72.0)),
    TextAnnotation("V-101", "equipment_tag", (175.0, 125.0, 205.0, 137.0)),
    TextAnnotation("T-102", "equipment_tag", (320.0, 60.0, 360.0, 72.0)),
    TextAnnotation("PI-100", "instrument_tag", (178.0, 50.0, 202.0, 62.0)),
    TextAnnotation("150-GA-001", "line_number", (90.0, 90.0, 150.0, 100.0)),
    TextAnnotation("DW02-0003", "opc_label", (327.0, 168.0, 353.0, 180.0)),
)


def _render(width: int, height: int, annotations: SheetAnnotations) -> bytes:
    """Draw the annotations into a grayscale raster (0 = ink, 255 = paper),
    so the synthetic Sheet genuinely has pixels."""
    grid = bytearray(b"\xff" * (width * height))

    def ink(x: float, y: float) -> None:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < width and 0 <= yi < height:
            grid[yi * width + xi] = 0

    def stroke(p: tuple, q: tuple) -> None:
        (x0, y0), (x1, y1) = p, q
        steps = max(int(abs(x1 - x0)), int(abs(y1 - y0)), 1)
        for i in range(steps + 1):
            t = i / steps
            ink(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)

    for sym in annotations.symbols:
        x0, y0, x1, y1 = sym.bbox
        for p, q in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
                     ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
            stroke(p, q)
    for line in annotations.lines:
        for p, q in zip(line.polyline, line.polyline[1:]):
            stroke(p, q)
    return bytes(grid)


def build_synthetic_sheet(number: int = 1) -> Sheet:
    annotations = SheetAnnotations(symbols=SYMBOLS, lines=LINES, texts=TEXTS)
    return Sheet(number=number, width=SHEET_W, height=SHEET_H,
                 raster=_render(SHEET_W, SHEET_H, annotations),
                 annotations=annotations)


@pytest.fixture
def synthetic_document() -> Document:
    return Document(name="synthetic-fixture.pdf",
                    sheets=(build_synthetic_sheet(1),))


@pytest.fixture
def synthetic_profile() -> ConventionProfile:
    """Loaded from the checked-in fixture bundle (ticket 04), so the whole
    suite exercises the on-disk profile loader."""
    return load_profile(Path(__file__).parent / "fixtures" / "profiles"
                        / "synthetic-test")
