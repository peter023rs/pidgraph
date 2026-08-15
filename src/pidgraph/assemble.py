"""Graph assembly: per-Sheet detections -> assembled topology -> the
s2_pml equipment-level graph.

Direction honesty: a run gets a flow direction only from evidence readable
off the Sheet (here: flow-arrow symbols sitting on the run). Everything
else stays undirected — "unknown" never silently becomes "known".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .model import (
    Bbox,
    ConventionProfile,
    LegendEntry,
    LineDetection,
    Point,
    Sheet,
    SymbolDetection,
    TextDetection,
)

_SNAP_TOL = 2.0    # px: line endpoint counts as touching a symbol bbox
_ARROW_TOL = 3.0   # px: arrow center counts as sitting on a run

_TERMINAL_ROLES = {"Equipment", "PipingComponent",
                   "ProcessInstrumentationFunction", "PipeOffPageConnector"}
# which terminal roles a text class may tag
_TAG_TARGETS = {
    "equipment_tag": {"Equipment", "PipingComponent"},
    "instrument_tag": {"ProcessInstrumentationFunction"},
    "opc_label": {"PipeOffPageConnector"},
}


@dataclass
class Terminal:
    detection: SymbolDetection
    entry: LegendEntry
    tag: TextDetection | None = None
    labels: list[str] = field(default_factory=list)
    nozzles: list[SymbolDetection] = field(default_factory=list)

    def node_tag(self) -> str:
        """The unique s2_pml node tag. OPCs get the OPC-<label> form
        hazop-ai's adapter uses (ADR-0001)."""
        if self.entry.role == "PipeOffPageConnector":
            base = self.labels[0] if self.labels else self.detection.id
            return f"OPC-{base}"
        if self.tag is not None:
            return self.tag.string
        return f"UNTAGGED-{self.detection.id}"

    def node_name(self) -> str:
        if self.entry.role == "PipeOffPageConnector":
            return " / ".join(self.labels) or "off-page connector"
        return self.node_tag()

    def combined_confidence(self) -> float:
        if self.tag is None:
            return self.detection.confidence
        return min(self.detection.confidence, self.tag.confidence)

    def combined_evidence(self) -> str:
        evidence = self.detection.provenance.evidence
        if self.tag is not None:
            evidence += f"; tagged by text {self.tag.id}"
        return evidence


@dataclass(frozen=True)
class FlowEvidence:
    orientation: str   # "from_to" | "to_from", relative to the polyline
    source: str        # evidence kind, e.g. "arrow"
    evidence_id: str   # the detection that supplied it


@dataclass
class Run:
    detection: LineDetection
    attachments: tuple[Terminal | None, Terminal | None]
    attached_via: tuple[SymbolDetection | None, SymbolDetection | None]
    line_number_texts: list[TextDetection] = field(default_factory=list)
    flow: FlowEvidence | None = None


@dataclass
class SheetAssembly:
    sheet: Sheet
    terminals: list[Terminal]
    arrows: list[SymbolDetection]
    runs: list[Run]


def _center(bbox: Bbox) -> Point:
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) / 2, (y0 + y1) / 2)


def _dist(p: Point, q: Point) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def _near_bbox(point: Point, bbox: Bbox, tol: float) -> bool:
    x0, y0, x1, y1 = bbox
    return (x0 - tol <= point[0] <= x1 + tol
            and y0 - tol <= point[1] <= y1 + tol)


def _point_to_segment(p: Point, a: Point, b: Point) -> float:
    (px, py), (ax, ay), (bx, by) = p, a, b
    dx, dy = bx - ax, by - ay
    if dx == dy == 0:
        return _dist(p, a)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy)
                     / (dx * dx + dy * dy)))
    return _dist(p, (ax + t * dx, ay + t * dy))


def _point_to_polyline(p: Point, polyline: tuple[Point, ...]) -> float:
    return min(_point_to_segment(p, a, b)
               for a, b in zip(polyline, polyline[1:]))


def _legend_entry(profile: ConventionProfile,
                  detection: SymbolDetection) -> LegendEntry:
    entry = profile.legend.get(detection.symbol_class)
    if entry is None:
        raise ValueError(
            f"symbol class {detection.symbol_class!r} ({detection.id}) is "
            f"not in Convention Profile {profile.name!r}'s Legend Dictionary"
            " — refusing to guess its meaning")
    return entry


def _nearest(candidates: list[Terminal], point: Point) -> Terminal | None:
    if not candidates:
        return None
    return min(candidates,
               key=lambda t: _dist(_center(t.detection.bbox), point))


def _attach_endpoint(point: Point, terminals: list[Terminal]
                     ) -> tuple[Terminal | None, SymbolDetection | None]:
    """A run endpoint attaches to a nozzle (folded into its equipment) or a
    terminal whose bbox it touches; otherwise it stays loose."""
    for terminal in terminals:
        for nozzle in terminal.nozzles:
            if _near_bbox(point, nozzle.bbox, _SNAP_TOL):
                return terminal, nozzle
    for terminal in terminals:
        if _near_bbox(point, terminal.detection.bbox, _SNAP_TOL):
            return terminal, None
    return None, None


def _seed_flow(run: Run, arrow: SymbolDetection) -> None:
    if arrow.direction is None:
        raise ValueError(
            f"flow arrow {arrow.id} carries no direction vector — it "
            "cannot serve as direction evidence")
    polyline = run.detection.polyline
    center = _center(arrow.bbox)
    a, b = min(zip(polyline, polyline[1:]),
               key=lambda leg: _point_to_segment(center, *leg))
    leg = (b[0] - a[0], b[1] - a[1])
    dot = leg[0] * arrow.direction[0] + leg[1] * arrow.direction[1]
    run.flow = FlowEvidence(
        orientation="from_to" if dot >= 0 else "to_from",
        source="arrow", evidence_id=arrow.id)


def assemble_sheet(sheet: Sheet,
                   symbols: list[SymbolDetection],
                   lines: list[LineDetection],
                   texts: list[TextDetection],
                   profile: ConventionProfile) -> SheetAssembly:
    terminals: list[Terminal] = []
    nozzles: list[SymbolDetection] = []
    arrows: list[SymbolDetection] = []
    for detection in symbols:
        entry = _legend_entry(profile, detection)
        if entry.role in _TERMINAL_ROLES:
            terminals.append(Terminal(detection=detection, entry=entry))
        elif entry.role == "Nozzle":
            nozzles.append(detection)
        elif entry.role == "FlowArrow":
            arrows.append(detection)
        else:
            raise ValueError(f"legend role {entry.role!r} "
                             f"({detection.symbol_class}) is not assemblable")

    equipment = [t for t in terminals if t.entry.role == "Equipment"]
    for nozzle in nozzles:
        parent = _nearest(equipment, _center(nozzle.bbox))
        if parent is None:
            raise ValueError(f"nozzle {nozzle.id} has no Equipment to "
                             "attach to")
        parent.nozzles.append(nozzle)

    runs = []
    for line in lines:
        start = _attach_endpoint(line.polyline[0], terminals)
        end = _attach_endpoint(line.polyline[-1], terminals)
        runs.append(Run(detection=line,
                        attachments=(start[0], end[0]),
                        attached_via=(start[1], end[1])))

    for text in texts:
        if text.text_class == "line_number":
            if runs:
                run = min(runs, key=lambda r: _point_to_polyline(
                    _center(text.bbox), r.detection.polyline))
                run.line_number_texts.append(text)
            continue
        roles = _TAG_TARGETS.get(text.text_class)
        if roles is None:
            continue  # not a tagging class (free text stays in detections)
        target = _nearest([t for t in terminals if t.entry.role in roles],
                          _center(text.bbox))
        if target is None:
            continue
        if text.text_class == "opc_label":
            target.labels.append(text.string)
            if target.tag is None:
                target.tag = text
        elif target.tag is None:
            target.tag = text
        else:
            raise ValueError(
                f"terminal {target.detection.id} already tagged "
                f"{target.tag.string!r}; refusing to overwrite with "
                f"{text.string!r}")

    for arrow in arrows:
        center = _center(arrow.bbox)
        on_run = [r for r in runs if _point_to_polyline(
            center, r.detection.polyline) <= _ARROW_TOL]
        if len(on_run) != 1:
            raise ValueError(
                f"flow arrow {arrow.id} sits on {len(on_run)} runs — "
                "direction evidence must be unambiguous")
        _seed_flow(on_run[0], arrow)

    return SheetAssembly(sheet=sheet, terminals=terminals, arrows=arrows,
                         runs=runs)


def build_plant_graph(assemblies: list[SheetAssembly]) -> dict:
    """The s2_pml equipment-level graph (ADR-0001): terminals become nodes,
    runs between terminals become edges split by direction evidence.
    equipment_type values stay inside hazop-ai's EquipmentType vocabulary
    ("vessel" is its fallback too)."""
    nodes = []
    for assembly in assemblies:
        for terminal in assembly.terminals:
            attributes: dict = {"sheets": [assembly.sheet.number]}
            if terminal.labels:
                attributes["labels"] = list(terminal.labels)
            nodes.append({
                "tag": terminal.node_tag(),
                "name": terminal.node_name(),
                "equipment_type": terminal.entry.equipment_type or "vessel",
                "detection_confidence": terminal.combined_confidence(),
                "attributes": attributes,
            })

    edges = []
    for assembly in assemblies:
        for run in assembly.runs:
            start, end = run.attachments
            if start is None or end is None:
                continue  # dead ends stay geometry; no equipment edge
            source, target = start.node_tag(), end.node_tag()
            edge_attrs: dict = {"direction": "unknown"}
            if run.flow is not None:
                if run.flow.orientation == "to_from":
                    source, target = target, source
                edge_attrs = {
                    "direction": "known",
                    "direction_sources": [
                        f"{run.flow.source}:{run.flow.evidence_id}"]}
            if run.line_number_texts:
                edge_attrs["line_numbers"] = [
                    t.string for t in run.line_number_texts]
            edges.append({"source": source, "target": target,
                          "attributes": edge_attrs})

    return {"nodes": nodes, "edges": edges}
