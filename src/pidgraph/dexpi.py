"""DEXPI JSON emission — pidgraph's canonical output.

The shape mirrors hazop-ai's plant_model_dexpi.json exactly (ADR-0001);
the schema string names that shared contract. pidgraph additionally puts
confidence and provenance on every record — additive keys the contract
fixture explicitly allows.
"""

from __future__ import annotations

import math
from typing import Mapping

from .assemble import Run, SheetAssembly, Terminal
from .model import ConventionProfile, Document, Point, Provenance

SCHEMA = "hazop-l1/dexpi-aligned/v1"

# The contract's segmentClass vocabulary — profile line semantics must map
# into it, validated at profile load so a bad mapping never surfaces mid-run.
SEGMENT_CLASSES = frozenset({"PipingNetworkSegment", "SignalLine"})


def _xy(point: Point) -> list[float]:
    return [round(point[0], 1), round(point[1], 1)]


def _center(bbox) -> list[float]:
    x0, y0, x1, y1 = bbox
    return _xy(((x0 + x1) / 2, (y0 + y1) / 2))


def _prov(provenance: Provenance) -> dict:
    return {"component": provenance.component,
            "evidence": provenance.evidence}


class _SheetEmitter:
    def __init__(self, assembly: SheetAssembly):
        self.assembly = assembly
        self.sheet = assembly.sheet.number
        self.records: dict[str, list] = {
            "Equipment": [], "Nozzle": [], "PipingComponent": [],
            "ProcessInstrumentationFunction": [], "PipingNetworkSegment": [],
            "PipeOffPageConnector": [], "PipingNode": [],
        }
        self._n = 0
        self._s = 0
        self._terminal_ids: dict[int, str] = {}
        self._nozzle_ids: dict[int, str] = {}
        self._junction_ids: dict[int, str] = {}

    def _next_node_id(self) -> str:
        node_id = f"p{self.sheet}n{self._n}"  # id scheme mirrors hazop-ai
        self._n += 1
        return node_id

    def _next_segment_id(self) -> str:
        segment_id = f"p{self.sheet}s{self._s}"
        self._s += 1
        return segment_id

    def _emit_terminal(self, terminal: Terminal) -> None:
        node_id = self._next_node_id()
        self._terminal_ids[id(terminal)] = node_id
        detection = terminal.detection
        base = {
            "id": node_id,
            "sheet": self.sheet,
            "position": _center(detection.bbox),
            "confidence": terminal.combined_confidence(),
            "provenance": {
                "component": detection.provenance.component,
                "evidence": terminal.combined_evidence()},
        }
        tag = terminal.tag.string if terminal.tag else None
        role = terminal.entry.role
        if role == "Equipment":
            self.records["Equipment"].append(base | {
                "tagName": tag,
                "shape": terminal.entry.shape or "unspecified",
                "bbox": [round(c, 1) for c in detection.bbox]})
            for nozzle in terminal.nozzles:
                nozzle_id = self._next_node_id()
                self._nozzle_ids[id(nozzle)] = nozzle_id
                self.records["Nozzle"].append({
                    "id": nozzle_id,
                    "sheet": self.sheet,
                    "position": _center(nozzle.bbox),
                    "equipment": node_id,
                    "subTagName": None,
                    "confidence": nozzle.confidence,
                    "provenance": _prov(nozzle.provenance)})
        elif role == "PipingComponent":
            self.records["PipingComponent"].append(base | {
                "componentClass": terminal.entry.component_class
                                  or "PipingComponent"})
        elif role == "ProcessInstrumentationFunction":
            self.records["ProcessInstrumentationFunction"].append(base | {
                "tagName": tag or f"UNTAGGED-{detection.id}",
                "componentClass": "ProcessInstrumentationFunction"})
        elif role == "PipeOffPageConnector":
            self.records["PipeOffPageConnector"].append(base | {
                "labels": list(terminal.labels)})

    def _endpoint_id(self, run: Run, end: int, point: Point) -> str:
        terminal, via_nozzle = run.attachments[end], run.attached_via[end]
        if via_nozzle is not None:
            return self._nozzle_ids[id(via_nozzle)]
        if terminal is not None:
            return self._terminal_ids[id(terminal)]
        junction = run.junctions[end]
        if junction is not None:
            return self._emit_junction_node(junction)
        return self._emit_piping_node(point, "dead-end", run)

    def _emit_junction_node(self, junction) -> str:
        """One shared PipingNode per branch point, however many runs meet
        there."""
        if id(junction) not in self._junction_ids:
            node_id = self._next_node_id()
            runs = junction.runs
            self.records["PipingNode"].append({
                "id": node_id,
                "sheet": self.sheet,
                "position": _xy(junction.point),
                "nodeType": "junction",
                "confidence": min(r.detection.confidence for r in runs),
                "provenance": {
                    "component": runs[0].detection.provenance.component,
                    "evidence": "junction of lines " + ", ".join(
                        sorted(r.detection.id for r in runs))}})
            self._junction_ids[id(junction)] = node_id
        return self._junction_ids[id(junction)]

    def _emit_piping_node(self, point: Point, node_type: str,
                          run: Run) -> str:
        node_id = self._next_node_id()
        self.records["PipingNode"].append({
            "id": node_id,
            "sheet": self.sheet,
            "position": _xy(point),
            "nodeType": node_type,
            "confidence": run.detection.confidence,
            "provenance": {
                "component": run.detection.provenance.component,
                "evidence": f"vertex of line {run.detection.id}"}})
        return node_id

    def _emit_run(self, run: Run, segment_class: str) -> None:
        polyline = run.detection.polyline
        chain = [self._endpoint_id(run, 0, polyline[0])]
        chain += [self._emit_piping_node(p, "corner", run)
                  for p in polyline[1:-1]]
        chain.append(self._endpoint_id(run, 1, polyline[-1]))

        for leg_index, (a, b) in enumerate(zip(polyline, polyline[1:])):
            segment = {
                "id": self._next_segment_id(),
                "sheet": self.sheet,
                "from": chain[leg_index],
                "to": chain[leg_index + 1],
                "segmentClass": segment_class,
                "length": round(math.dist(a, b), 1),
                "polyline": [_xy(a), _xy(b)],
                "confidence": run.detection.confidence,
                "provenance": _prov(run.detection.provenance),
            }
            if run.line_number_texts:
                segment["lineNumber"] = [t.string
                                         for t in run.line_number_texts]
            flow = run.flow  # None without evidence, and None on conflict
            if flow is not None:
                segment["flowDirection"] = flow.orientation
                # propagation provenance: "propagated" (contract vocab)
                # distinguishes carried direction from on-run evidence
                segment["flowDirectionSource"] = (
                    "propagated" if flow.propagated else flow.source)
            self.records["PipingNetworkSegment"].append(segment)

    def emit(self, line_semantics: Mapping[str, str]) -> dict[str, list]:
        for terminal in self.assembly.terminals:
            self._emit_terminal(terminal)
        for run in self.assembly.runs:
            self._emit_run(run, line_semantics.get(
                run.detection.line_class, "PipingNetworkSegment"))
        return self.records


def emit_plant_model(assemblies: list[SheetAssembly], document: Document,
                     profile: ConventionProfile) -> dict:
    conceptual: dict[str, list] = {
        "Equipment": [], "Nozzle": [], "PipingComponent": [],
        "ProcessInstrumentationFunction": [], "PipingNetworkSegment": [],
        "PipeOffPageConnector": [], "PipingNode": [],
    }
    line_sheets: dict[str, set[int]] = {}
    for assembly in assemblies:
        records = _SheetEmitter(assembly).emit(profile.line_semantics)
        for type_name, items in records.items():
            conceptual[type_name].extend(items)
        for run in assembly.runs:
            for text in run.line_number_texts:
                line_sheets.setdefault(text.string, set()).add(
                    assembly.sheet.number)

    cross_sheet = [
        {"lineNumber": line, "sheets": sorted(sheets)}
        for line, sheets in sorted(line_sheets.items())
        if len(sheets) > 1
    ]
    return {
        "schema": SCHEMA,
        "sourceDrawing": document.name,
        "conventionProfile": profile.identity_record(),
        "conceptualModel": conceptual,
        "crossSheetLinks": cross_sheet,
    }
