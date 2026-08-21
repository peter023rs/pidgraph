"""Graph assembly: per-Sheet detections -> assembled topology -> the
s2_pml equipment-level graph.

Direction honesty: a run gets a flow direction only from evidence readable
off the Sheet (here: flow-arrow symbols sitting on the run). Everything
else stays undirected — "unknown" never silently becomes "known".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations

from .model import (
    UNCLASSIFIED_SYMBOL,
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
    "opc_direction": {"PipeOffPageConnector"},
}


@dataclass
class Terminal:
    detection: SymbolDetection
    entry: LegendEntry
    tag: TextDetection | None = None
    labels: list[str] = field(default_factory=list)
    nozzles: list[SymbolDetection] = field(default_factory=list)
    direction_texts: list[TextDetection] = field(default_factory=list)
                                          # OPC direction text ("TO ..." /
                                          # "FROM ..."), evidence for flow

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
    source: str        # evidence kind: "arrow" | "connector"
    evidence_id: str   # the detection that supplied it
    propagated: bool = False  # True: carried along the line network from a
                              # neighboring run, not read off this run

    def ref(self) -> str:
        """The evidence reference emitted in direction_sources /
        direction_conflicts; propagation provenance stays visible."""
        ref = f"{self.source}:{self.evidence_id}"
        return f"propagated({ref})" if self.propagated else ref


@dataclass
class Junction:
    """A branch point where several runs meet without a symbol between
    them — loose run endpoints that coincide on the Sheet."""
    point: Point
    runs: list[Run] = field(default_factory=list)


@dataclass
class Run:
    detection: LineDetection
    attachments: tuple[Terminal | None, Terminal | None]
    attached_via: tuple[SymbolDetection | None, SymbolDetection | None]
    junctions: tuple[Junction | None, Junction | None] = (None, None)
    line_number_texts: list[TextDetection] = field(default_factory=list)
    flow_evidence: list[FlowEvidence] = field(default_factory=list)

    @property
    def flow_conflict(self) -> bool:
        """Evidence disagrees on this run's direction — surfaced as a
        conflict; neither orientation is ever asserted."""
        return len({e.orientation for e in self.flow_evidence}) > 1

    @property
    def flow(self) -> FlowEvidence | None:
        """The run's agreed direction: seeded evidence first, else the
        first propagated. None without evidence, and None on conflict."""
        if not self.flow_evidence or self.flow_conflict:
            return None
        seeded = [e for e in self.flow_evidence if not e.propagated]
        return (seeded or self.flow_evidence)[0]


@dataclass
class SheetAssembly:
    sheet: Sheet
    terminals: list[Terminal]
    arrows: list[SymbolDetection]
    runs: list[Run]
    junctions: list[Junction] = field(default_factory=list)


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


def _group_junctions(runs: list[Run]) -> list[Junction]:
    """Loose run endpoints that coincide (within snap tolerance) are one
    branch point: the runs meet there. A lone loose endpoint stays a dead
    end."""
    groups: list[tuple[Point, list[tuple[Run, int]]]] = []
    for run in runs:
        for end in (0, 1):
            if run.attachments[end] is not None:
                continue
            point = run.detection.polyline[0 if end == 0 else -1]
            for anchor, members in groups:
                if _dist(anchor, point) <= _SNAP_TOL:
                    members.append((run, end))
                    break
            else:
                groups.append((point, [(run, end)]))

    junctions = []
    for anchor, members in groups:
        if len(members) < 2:
            continue
        junction = Junction(point=anchor)
        for run, end in members:
            junction.runs.append(run)
            ends = list(run.junctions)
            ends[end] = junction
            run.junctions = (ends[0], ends[1])
        junctions.append(junction)
    return junctions


def _seed_arrow_flow(run: Run, arrow: SymbolDetection) -> None:
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
    run.flow_evidence.append(FlowEvidence(
        orientation="from_to" if dot >= 0 else "to_from",
        source="arrow", evidence_id=arrow.id))


def _seed_connector_flow(terminal: Terminal, text: TextDetection,
                         runs: list[Run]) -> None:
    """Off-page-connector direction text: "TO ..." means flow leaves the
    sheet through this OPC, "FROM ..." means it enters — every run attached
    to the OPC is seeded accordingly."""
    tokens = text.string.split()
    if not tokens or tokens[0].upper() not in ("TO", "FROM"):
        raise ValueError(
            f"opc_direction text {text.id} reads {text.string!r} — "
            "direction text must start with TO or FROM to serve as "
            "direction evidence")
    toward_opc = tokens[0].upper() == "TO"
    for run in runs:
        for end in (0, 1):
            if run.attachments[end] is not terminal:
                continue
            at_polyline_end = end == 1
            orientation = ("from_to" if at_polyline_end == toward_opc
                           else "to_from")
            run.flow_evidence.append(FlowEvidence(
                orientation=orientation, source="connector",
                evidence_id=text.id))


def _propagate_flow(runs: list[Run]) -> None:
    """Conservative propagation: direction crosses a junction only where
    exactly two runs meet — a plain continuation of the same line. A
    branch, a terminal, or a dead end stops it, and conflicting evidence
    stops it where the conflict arises. Propagated evidence keeps its
    seed's identity, so provenance distinguishes seeded from propagated."""
    queue = [run for run in runs if run.flow is not None]
    while queue:
        run = queue.pop(0)
        evidence = run.flow
        if evidence is None:  # became conflicted after it was queued
            continue
        for end in (0, 1):
            junction = run.junctions[end]
            if junction is None or len(junction.runs) != 2:
                continue
            other = (junction.runs[0] if junction.runs[1] is run
                     else junction.runs[1])
            if other is run:
                continue  # a run looping back to its own junction
            if any(e.source == evidence.source
                   and e.evidence_id == evidence.evidence_id
                   for e in other.flow_evidence):
                continue  # this seed already reached that run
            outbound = (evidence.orientation == "from_to") == (end == 1)
            other_end = 0 if other.junctions[0] is junction else 1
            away = "from_to" if other_end == 0 else "to_from"
            toward = "to_from" if other_end == 0 else "from_to"
            other.flow_evidence.append(FlowEvidence(
                orientation=away if outbound else toward,
                source=evidence.source, evidence_id=evidence.evidence_id,
                propagated=True))
            if other.flow is not None:
                queue.append(other)


def assemble_sheet(sheet: Sheet,
                   symbols: list[SymbolDetection],
                   lines: list[LineDetection],
                   texts: list[TextDetection],
                   profile: ConventionProfile) -> SheetAssembly:
    terminals: list[Terminal] = []
    nozzles: list[SymbolDetection] = []
    arrows: list[SymbolDetection] = []
    for detection in symbols:
        if detection.symbol_class == UNCLASSIFIED_SYMBOL:
            continue  # fail-closed (ticket 17): an unclassified symbol
                      # stays in the detection record for review but
                      # never becomes a plant item
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
    junctions = _group_junctions(runs)

    for text in texts:
        if not text.resolved:
            continue  # fail-closed (ticket 06): an unresolved tag stays in
                      # the detection record but never names a plant item
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
        if text.text_class == "opc_direction":
            target.direction_texts.append(text)
        elif text.text_class == "opc_label":
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
        _seed_arrow_flow(on_run[0], arrow)

    for terminal in terminals:
        for text in terminal.direction_texts:
            _seed_connector_flow(terminal, text, runs)
    _propagate_flow(runs)

    return SheetAssembly(sheet=sheet, terminals=terminals, arrows=arrows,
                         runs=runs, junctions=junctions)


def build_plant_graph(assemblies: list[SheetAssembly]) -> dict:
    """The s2_pml equipment-level graph (ADR-0001): terminals become nodes,
    runs between terminals become edges split by direction evidence, and
    terminals meeting at junctions become undirected edges. Every edge
    carries confidence and provenance. equipment_type values stay inside
    hazop-ai's EquipmentType vocabulary ("vessel" is its fallback too)."""
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
                continue  # loose ends handled below (junction) or dropped
            source, target = start.node_tag(), end.node_tag()
            refs = sorted({e.ref() for e in run.flow_evidence})
            edge_attrs: dict = {"direction": "unknown"}
            if run.flow_conflict:
                edge_attrs = {"direction": "conflict",
                              "direction_conflicts": refs}
            elif run.flow is not None:
                if run.flow.orientation == "to_from":
                    source, target = target, source
                edge_attrs = {"direction": "known",
                              "direction_sources": refs}
            if run.line_number_texts:
                edge_attrs["line_numbers"] = [
                    t.string for t in run.line_number_texts]
            detection = run.detection
            edge_attrs["confidence"] = detection.confidence
            edge_attrs["provenance"] = (
                f"{detection.provenance.component}: "
                f"{detection.provenance.evidence}")
            edges.append({"source": source, "target": target,
                          "attributes": edge_attrs})
        edges.extend(_junction_edges(assembly))

    return {"nodes": nodes, "edges": edges}


def _chain_direction(runs: list[Run], terminals: list[Terminal]
                     ) -> tuple[Terminal, Terminal] | None:
    """For a two-terminal junction chain whose every run carries an agreed
    direction: the terminal the flow leaves and the one it reaches, or
    None when the runs' directions do not line up head to tail."""
    outward: list[Terminal] = []
    inward: list[Terminal] = []
    for terminal in terminals:
        for run in runs:
            flow = run.flow
            if flow is None:
                return None
            for end in (0, 1):
                if run.attachments[end] is terminal:
                    leaves = (flow.orientation == "from_to") == (end == 0)
                    (outward if leaves else inward).append(terminal)
    if (len(outward) == 1 and len(inward) == 1
            and outward[0] is not inward[0]):
        return outward[0], inward[0]
    return None


def _junction_edges(assembly: SheetAssembly) -> list[dict]:
    """Terminals joined through a branch point (or a chain of them) are
    connected: every terminal pair in a junction-connected group of runs
    gets an undirected edge — except a two-terminal chain whose runs all
    carry one propagated-consistent direction, which becomes a single
    directed edge, and a chain holding conflicting evidence, which is
    surfaced as a conflict (ticket 07)."""
    roots: dict[int, int] = {id(j): id(j) for j in assembly.junctions}

    def find(a: int) -> int:
        while roots[a] != a:
            roots[a] = roots[roots[a]]
            a = roots[a]
        return a

    for run in assembly.runs:
        j0, j1 = run.junctions
        if j0 is not None and j1 is not None:
            roots[find(id(j0))] = find(id(j1))

    components: dict[int, list[Run]] = {}
    for run in assembly.runs:
        for root in {find(id(j)) for j in run.junctions if j is not None}:
            components.setdefault(root, []).append(run)

    edges = []
    for root in sorted(components):
        runs = components[root]
        terminals: list[Terminal] = []
        for run in runs:
            for terminal in run.attachments:
                if terminal is not None and terminal not in terminals:
                    terminals.append(terminal)
        if len(terminals) < 2:
            continue  # a branch that reaches at most one plant item
        terminals.sort(key=Terminal.node_tag)
        run_ids = sorted(run.detection.id for run in runs)
        attrs: dict = {
            "direction": "unknown",
            "confidence": min(run.detection.confidence for run in runs),
            "provenance": (
                f"{runs[0].detection.provenance.component}: runs "
                f"{', '.join(run_ids)} meeting at a junction"),
        }
        line_numbers = sorted({text.string for run in runs
                               for text in run.line_number_texts})
        if line_numbers:
            attrs["line_numbers"] = line_numbers

        refs = sorted({e.ref() for run in runs for e in run.flow_evidence})
        if any(run.flow_conflict for run in runs):
            attrs["direction"] = "conflict"
            attrs["direction_conflicts"] = refs
        elif len(terminals) == 2:
            oriented = _chain_direction(runs, terminals)
            if oriented is not None:
                upstream, downstream = oriented
                edges.append({
                    "source": upstream.node_tag(),
                    "target": downstream.node_tag(),
                    "attributes": attrs | {"direction": "known",
                                           "direction_sources": refs}})
                continue
        for a, b in combinations(terminals, 2):
            edges.append({"source": a.node_tag(), "target": b.node_tag(),
                          "attributes": dict(attrs)})
    return edges
