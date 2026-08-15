"""Line network extraction — classical CV, deterministic, no seam and no ML
(spec decision).

Walking-skeleton placeholder: traces the line annotations a synthetic Sheet
carries. Ticket 05 replaces the internals with the real raster path
(binarize -> thin -> vectorize -> junction analysis -> split runs at symbol
boxes); the signature and the LineDetection contract stay put.
"""

from __future__ import annotations

from .model import LineDetection, Provenance, Sheet

COMPONENT = "line_extractor:skeleton"


def extract_line_network(sheet: Sheet) -> list[LineDetection]:
    if sheet.annotations is None:
        raise ValueError(
            f"{COMPONENT} traces synthetic annotations; Sheet "
            f"{sheet.number} carries none")
    return [
        LineDetection(
            id=f"p{sheet.number}-line{i}",
            sheet=sheet.number,
            polyline=line.polyline,
            line_class=line.line_class,
            confidence=1.0,
            provenance=Provenance(
                component=COMPONENT,
                evidence=f"synthetic annotation {line.line_class} polyline "
                         f"{line.polyline[0]} .. {line.polyline[-1]}"),
        )
        for i, line in enumerate(sheet.annotations.lines)
    ]
