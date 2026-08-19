"""Dev-time label factory (ticket 13): render hazop-ai's 2401 vector
artifacts to raster Sheets and project the deterministic detections, text
spans, and connectivity down as pixel-level ground truth — so detector and
OCR training/eval data exists before the corpus arrives.

This is a dependency on hazop-ai's artifacts (topology_page<N>.json), not
its code, and it dissolves as Review Workbench corrections accumulate. It
is not part of the shipped Raster Path: nothing in the pipeline imports
it, and the factory itself never runs in CI — only its tests do, against
small synthetic vector fixtures. Everything it writes is drawing-derived
content and belongs outside git (ADR-0001).

The artifacts carry detections and connectivity in PDF points (top-left
origin, y down) but no drawing primitives and no text geometry, so the
Sheet raster comes from rasterizing the source PDF and the text ground
truth from the PDF's text layer — both behind the read_page seam, the
factory's one rendering dependency (PyMuPDF, the labelfactory extra).
Projection to pixels is the uniform scale dpi/72; hazop-ai's own overlay
renderer draws in PDF point space and rasterizes the same way.

Ground truth leaves the factory in the shapes the rest of the product
already consumes: symbol/text/line examples as pass verdicts in the
labels.LabelStore schema (exported per profile as a training set,
ticket 12), and terminal-level connectivity per Sheet for the eval
harness's connectivity gate (ticket 15).
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

from .labels import LabelStore, make_example, profile_key
from .pngio import encode_gray_png
from .training import export_training_set

POINTS_PER_INCH = 72.0
DEFAULT_DPI = 150  # hazop-ai's own overlay render resolution

# derived drawing content stays outside git: data/ is ignored (ADR-0001)
DEFAULT_OUT_DIR = Path("data") / "labelfactory"

# the labels' Convention Profile stamp: the 2401 package's convention,
# versioned by the artifact generation that produced the ground truth
DEFAULT_PROFILE = {"name": "hazop-ai-2401", "version": "l1"}

COMPONENT = "label_factory:projection"

_REQUIRED_KEYS = ("page", "nodes", "edges", "direction_stats")

# artifact node kinds that are drawn symbols vs. line-network geometry
_TERMINAL_KINDS = ("equipment", "valve", "instrument", "off-page", "nozzle")
_PASS_THROUGH_KINDS = ("junction", "corner", "dead-end")


@dataclass(frozen=True)
class BoxSynthesis:
    """Half-extents in PDF points, (x, y), for the symbol kinds whose
    artifact record carries only a center point. Equipment comes with its
    detected bbox; everything else is synthesized from the drawing
    convention: the ISA bubble radius hazop-ai pins (16.4 pt), valve and
    off-page-connector extents measured off the 2401 renders."""
    instrument_half_pt: tuple[float, float] = (16.4, 16.4)
    valve_half_pt: tuple[float, float] = (8.0, 8.0)
    nozzle_half_pt: tuple[float, float] = (4.0, 4.0)
    opc_half_pt: tuple[float, float] = (16.0, 8.0)

    def half(self, kind: str) -> tuple[float, float]:
        return {
            "instrument": self.instrument_half_pt,
            "valve": self.valve_half_pt,
            "nozzle": self.nozzle_half_pt,
            "off-page": self.opc_half_pt,
        }[kind]


@dataclass(frozen=True)
class TextSpan:
    """One span of the page's text layer, in PDF points — the lossless
    OCR ground truth the artifacts themselves never persisted."""
    string: str
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class SheetLabels:
    """One artifact's ground truth, projected to pixels: detection-shaped
    label dicts for the three example kinds, plus terminal-level
    connectivity links between the symbol label ids."""
    sheet: int
    symbols: list[dict] = field(default_factory=list)
    texts: list[dict] = field(default_factory=list)
    lines: list[dict] = field(default_factory=list)
    links: list[dict] = field(default_factory=list)


def validate_artifact(artifact: dict, source: str) -> None:
    """Refuse an artifact that is not one of hazop-ai's current
    topology_page emissions, naming the file and what is missing. The
    stale legend-sheet runs shipped beside the real ones predate
    direction_stats, so that key doubles as the schema discriminator."""
    missing = [key for key in _REQUIRED_KEYS if key not in artifact]
    if missing:
        detail = ""
        if "direction_stats" in missing:
            detail = (" — an artifact without direction_stats predates"
                      " hazop-ai's current schema (the stale legend-sheet"
                      " runs); re-run hazop-ai's Stage 1 or drop it")
        raise ValueError(
            f"{source} is not a current hazop-ai topology artifact:"
            f" missing {', '.join(missing)}{detail}")


def _node_label(sheet: int, node_id: object) -> str:
    """The label id an artifact node projects to — one scheme, shared by
    the symbol labels and the connectivity links that reference them."""
    return f"p{sheet}-node{node_id}"


def _symbol_class(node: dict) -> str:
    kind = node["kind"]
    if kind == "equipment":
        shape = node.get("shape")
        return f"equipment_{shape}" if shape else "equipment"
    if kind == "valve":
        subclass = node.get("subclass")
        return f"{subclass}_valve" if subclass else "valve"
    if kind == "instrument":
        return "instrument_bubble"
    if kind == "off-page":
        return "opc"
    return "nozzle"


def _node_bbox_pt(node: dict, synthesis: BoxSynthesis) -> list[float]:
    """The symbol's box in PDF points: a node carrying its detected bbox
    (equipment, and the equipment-like off-page items) keeps it;
    center-only kinds get a synthesized box around xy."""
    if node.get("bbox") is not None:
        return [float(v) for v in node["bbox"]]
    x, y = node["xy"]
    half_x, half_y = synthesis.half(node["kind"])
    return [x - half_x, y - half_y, x + half_x, y + half_y]


def _scaled_bbox(bbox: list[float], scale: float) -> list[float]:
    return [v * scale for v in bbox]


def _symbol_labels(artifact: dict, source: str, scale: float,
                   synthesis: BoxSynthesis) -> list[dict]:
    sheet = artifact["page"]
    labels = []
    for node in artifact["nodes"]:
        kind = node["kind"]
        if kind in _PASS_THROUGH_KINDS:
            continue
        if kind not in _TERMINAL_KINDS:
            raise ValueError(
                f"{source} node {node.get('id')!r} has kind {kind!r},"
                f" which the label factory does not know; symbol kinds"
                f" are {list(_TERMINAL_KINDS)}")
        labels.append({
            "id": _node_label(sheet, node["id"]),
            "sheet": sheet,
            "symbol_class": _symbol_class(node),
            "bbox": _scaled_bbox(_node_bbox_pt(node, synthesis), scale),
            "confidence": 1.0,
            "provenance": {
                "component": COMPONENT,
                "evidence": f"hazop-ai {source} node {node['id']}"
                            f" ({kind})",
            },
        })
    return labels


def _tag_tokens(node: dict) -> frozenset[str]:
    """The pieces an instrument tag is drawn as: bubbles letter the
    function on one line and the number below, so 'PT 00204' appears as
    the spans 'PT' and '00204'."""
    tokens = set()
    for key in ("tag", "full_tag"):
        for piece in str(node.get(key) or "").replace("-", " ").split():
            tokens.add(piece)
    return frozenset(tokens)


def _candidate_strings(artifact: dict, source: str) -> dict[str, tuple]:
    """Ground-truth strings the artifact knows, mapped to their text
    class — matched exactly against the page's spans. Built in one
    deterministic order; the first class to claim a string keeps it."""
    candidates: dict[str, tuple[str, str]] = {}

    def claim(string: object, text_class: str, detail: str) -> None:
        if isinstance(string, str) and string.strip():
            candidates.setdefault(string.strip(), (text_class, detail))

    for node in artifact["nodes"]:
        kind = node["kind"]
        if kind == "equipment":
            claim(node.get("tag"), "equipment_tag",
                  f"equipment node {node['id']}")
        elif kind == "instrument":
            for key in ("tag", "full_tag"):
                claim(node.get(key), "instrument_tag",
                      f"instrument node {node['id']}")
        elif kind == "off-page":
            for label in node.get("labels") or ():
                claim(label, "opc_label", f"off-page node {node['id']}")
    for index, edge in enumerate(artifact["edges"]):
        for number in edge.get("line_numbers") or ():
            claim(number, "line_number", f"edge {index}")
    return candidates


def _inside(bbox: list[float], point: tuple[float, float]) -> bool:
    x0, y0, x1, y1 = bbox
    return x0 <= point[0] <= x1 and y0 <= point[1] <= y1


def _text_labels(artifact: dict, spans: list[TextSpan], source: str,
                 scale: float, synthesis: BoxSynthesis) -> list[dict]:
    sheet = artifact["page"]
    candidates = _candidate_strings(artifact, source)
    bubbles = [(node, _node_bbox_pt(node, synthesis), _tag_tokens(node))
               for node in artifact["nodes"]
               if node["kind"] == "instrument"]

    labels = []
    for index, span in enumerate(spans):
        string = span.string.strip()
        if not string:
            continue
        evidence = f"PDF text layer span {index} of page {sheet}"
        if string in candidates:
            text_class, detail = candidates[string]
            evidence += (f"; matches {text_class} {string!r}"
                         f" from hazop-ai {source} {detail}")
        else:
            center = ((span.bbox[0] + span.bbox[2]) / 2,
                      (span.bbox[1] + span.bbox[3]) / 2)
            bubble = next(
                (b for b in bubbles
                 if string in b[2] and _inside(b[1], center)), None)
            if bubble is not None:
                node = bubble[0]
                text_class = "instrument_tag"
                evidence += (f"; token of instrument tag"
                             f" {node.get('tag')!r} (hazop-ai {source}"
                             f" node {node['id']})")
            else:
                text_class = "free_text"
                evidence += "; matches no artifact string"
        labels.append({
            "id": f"p{sheet}-span{index}",
            "sheet": sheet,
            "string": string,
            "text_class": text_class,
            "bbox": _scaled_bbox(list(span.bbox), scale),
            "confidence": 1.0,
            "provenance": {"component": COMPONENT, "evidence": evidence},
        })
    return labels


# drawn stroke styles the artifacts use, mapped to ground-truth line
# classes; the nozzle-to-shell "internal" edges are bookkeeping, not ink
_LINE_CLASSES = {"solid": "pipe", "dashed": "signal"}


def _line_labels(artifact: dict, source: str, scale: float) -> list[dict]:
    sheet = artifact["page"]
    labels = []
    for index, edge in enumerate(artifact["edges"]):
        style = edge.get("style", "solid")
        if style == "internal":
            continue
        labels.append({
            "id": f"p{sheet}-edge{index}",
            "sheet": sheet,
            "line_class": _LINE_CLASSES.get(style, style),
            "polyline": [[x * scale, y * scale]
                         for x, y in edge["points"]],
            "confidence": 1.0,
            "provenance": {
                "component": COMPONENT,
                "evidence": f"hazop-ai {source} edge {index} ({style})",
            },
        })
    return labels


def _terminal_ids(artifact: dict, source: str) -> dict[int, str | None]:
    """Each node's terminal identity: symbols stand for themselves,
    nozzles fold into their parent item, and pass-through geometry has
    none. The artifact references a nozzle's parent by index into the
    sheet's bbox-carrying nodes in order of appearance — equipment plus
    the equipment-like off-page items (verified against hazop-ai's DEXPI
    export on every 2401 nozzle)."""
    sheet = artifact["page"]
    items = [n for n in artifact["nodes"] if n.get("bbox") is not None]
    terminals: dict[int, str | None] = {}
    for node in artifact["nodes"]:
        kind, node_id = node["kind"], node["id"]
        if kind in _PASS_THROUGH_KINDS:
            terminals[node_id] = None
        elif kind == "nozzle":
            index = node.get("equipment")
            if (not isinstance(index, int) or isinstance(index, bool)
                    or not 0 <= index < len(items)):
                raise ValueError(
                    f"{source} nozzle node {node_id} references"
                    f" equipment-like item {index!r}, but the artifact"
                    f" draws {len(items)} bbox-carrying nodes")
            parent = items[index]
            if parent["kind"] in _PASS_THROUGH_KINDS:
                raise ValueError(
                    f"{source} nozzle node {node_id} folds into node"
                    f" {parent['id']} ({parent['kind']}), which is not a"
                    f" plant item")
            terminals[node_id] = _node_label(sheet, parent["id"])
        else:
            terminals[node_id] = _node_label(sheet, node_id)
    return terminals


@dataclass
class _Attachment:
    """One terminal's contact with a pass-through component: through
    which edge, and whether directed evidence says flow leaves the
    terminal there."""
    terminal: str
    edge_index: int
    outbound: bool | None      # None: the edge carries no direction
    source: str | None         # the edge's direction_source
    line_numbers: tuple[str, ...]


def _edge_direction(edge: dict, incident: int) -> tuple[bool | None,
                                                        str | None]:
    direction = edge.get("direction")
    if direction not in ("ab", "ba"):
        return None, None
    at_a = edge["a"] == incident
    outbound = (direction == "ab") == at_a
    return outbound, edge.get("direction_source")


def _terminal_flow(attachments: list[_Attachment]) -> bool | None:
    """What a terminal's attachments to one component agree flow does
    there: True = leaves it, False = enters it, None = no direction is
    asserted (undirected or disagreeing evidence — never guessed)."""
    directions = {a.outbound for a in attachments}
    if directions == {True} or directions == {False}:
        return next(iter(directions))
    return None


def _pair_link(a: list[_Attachment], b: list[_Attachment],
               interior_edges: list[int], interior_numbers: set[str],
               artifact_source: str) -> dict:
    """One ground-truth connection, in the s2_pml edge shape the
    pipeline's build_plant_graph emits (ADR-0001) — so the eval
    harness's connectivity gate compares like for like."""
    flow_a, flow_b = _terminal_flow(a), _terminal_flow(b)
    source, target = a[0].terminal, b[0].terminal
    if flow_a is not None and flow_b is not None and flow_a != flow_b:
        direction = "known"
        if flow_b:  # flow leaves b: it is the upstream side
            source, target = target, source
    else:
        direction = "unknown"
        source, target = sorted((source, target))
    attributes: dict = {"direction": direction}
    if direction == "known":
        attributes["direction_sources"] = sorted(
            {att.source for att in a + b if att.source is not None})
    numbers = interior_numbers.union(
        *(att.line_numbers for att in a + b))
    if numbers:
        attributes["line_numbers"] = sorted(numbers)
    edges = sorted({att.edge_index for att in a + b} | set(interior_edges))
    attributes["confidence"] = 1.0
    attributes["provenance"] = (
        f"{COMPONENT}: hazop-ai {artifact_source} edges"
        f" {', '.join(str(index) for index in edges)}")
    return {"source": source, "target": target, "attributes": attributes}


def _connectivity_links(artifact: dict, source: str) -> list[dict]:
    """Terminal-level ground-truth connectivity: direct terminal-to-
    terminal runs, plus terminals joined through contracted pass-through
    geometry (corners, junctions, dead ends). Direction is carried only
    where the terminal-incident edges assert it consistently."""
    terminals = _terminal_ids(artifact, source)

    # union-find over pass-through nodes
    roots: dict[int, int] = {node_id: node_id
                             for node_id, t in terminals.items()
                             if t is None}

    def find(node_id: int) -> int:
        while roots[node_id] != node_id:
            roots[node_id] = roots[roots[node_id]]
            node_id = roots[node_id]
        return node_id

    def resolve(edge: dict, index: int, end: str) -> int:
        node_id = edge[end]
        if node_id not in terminals:
            raise ValueError(f"{source} edge {index} references unknown"
                             f" node {node_id}")
        return node_id

    drawn = [(index, edge) for index, edge in enumerate(artifact["edges"])
             if edge.get("style") != "internal"]
    for index, edge in drawn:
        a, b = resolve(edge, index, "a"), resolve(edge, index, "b")
        if terminals[a] is None and terminals[b] is None:
            roots[find(a)] = find(b)

    links = []
    attachments: dict[int, list[_Attachment]] = {}
    interior_edges: dict[int, list[int]] = {}
    interior_numbers: dict[int, set[str]] = {}
    for index, edge in drawn:
        a, b = edge["a"], edge["b"]
        term_a, term_b = terminals[a], terminals[b]
        numbers = tuple(edge.get("line_numbers") or ())
        if term_a is not None and term_b is not None:
            if term_a == term_b:
                continue  # a run looping back to one plant item
            flow_a, flow_source = _edge_direction(edge, a)
            att_a = _Attachment(term_a, index, flow_a, flow_source, numbers)
            flow_b, _ = _edge_direction(edge, b)
            att_b = _Attachment(term_b, index, flow_b, flow_source, numbers)
            links.append(_pair_link([att_a], [att_b], [], set(), source))
            continue
        if term_a is None and term_b is None:
            root = find(a)
            interior_edges.setdefault(root, []).append(index)
            interior_numbers.setdefault(root, set()).update(numbers)
            continue
        terminal_id, pass_id = (a, b) if term_a is not None else (b, a)
        terminal_name = term_a if term_a is not None else term_b
        assert terminal_name is not None  # the both-None case continued
        outbound, flow_source = _edge_direction(edge, terminal_id)
        attachments.setdefault(find(pass_id), []).append(_Attachment(
            terminal_name, index, outbound, flow_source, numbers))

    for root in attachments:
        by_terminal: dict[str, list[_Attachment]] = {}
        for attachment in attachments[root]:
            by_terminal.setdefault(attachment.terminal, []).append(
                attachment)
        names = sorted(by_terminal)
        for i, first in enumerate(names):
            for second in names[i + 1:]:
                links.append(_pair_link(
                    by_terminal[first], by_terminal[second],
                    interior_edges.get(root, []),
                    interior_numbers.get(root, set()), source))

    return sorted(links, key=lambda l: (l["source"], l["target"],
                                        l["attributes"]["provenance"]))


def build_sheet_labels(artifact: dict, spans: list[TextSpan],
                       scale: float,
                       synthesis: BoxSynthesis | None = None,
                       source: str = "artifact") -> SheetLabels:
    """Project one validated artifact (plus the page's text spans, in PDF
    points) down to pixel-level ground truth at the given scale."""
    synthesis = synthesis or BoxSynthesis()
    return SheetLabels(
        sheet=artifact["page"],
        symbols=_symbol_labels(artifact, source, scale, synthesis),
        texts=_text_labels(artifact, spans, source, scale, synthesis),
        lines=_line_labels(artifact, source, scale),
        links=_connectivity_links(artifact, source),
    )


# --------------------------------------------------------------------------
# The rendering edge: one page of the source PDF as raster plus text layer

@dataclass(frozen=True)
class PageRender:
    """One rendered PDF page: the raster (row-major 8-bit grayscale, the
    Sheet.raster form) and the page's text spans in PDF points."""
    width: int
    height: int
    pixels: bytes
    spans: tuple[TextSpan, ...]
    page_width_pt: float
    page_height_pt: float


PageReader = Callable[[Path, int, int], PageRender]


def read_page(pdf_path: Path, page_number: int, dpi: int) -> PageRender:
    """The PyMuPDF-backed reader. Imported lazily so the factory's one
    rendering dependency stays optional — install the labelfactory extra
    to use it. Rendering is fully local (ADR-0001)."""
    try:
        import pymupdf  # type: ignore[import-not-found]
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "the label factory renders Sheets with PyMuPDF; install the "
            "labelfactory extra (pip install pidgraph[labelfactory])"
        ) from error
    with pymupdf.open(pdf_path) as document:
        if not 1 <= page_number <= document.page_count:
            raise ValueError(
                f"the artifact points at page {page_number}, but "
                f"{Path(pdf_path).name} has {document.page_count} pages")
        page = document[page_number - 1]
        zoom = dpi / POINTS_PER_INCH
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom),
                                 colorspace=pymupdf.csGRAY)
        spans = tuple(
            TextSpan(span["text"], tuple(span["bbox"]))
            for block in page.get_text("dict")["blocks"]
            for line in block.get("lines", [])
            for span in line["spans"])
        return PageRender(width=pixmap.width, height=pixmap.height,
                          pixels=bytes(pixmap.samples), spans=spans,
                          page_width_pt=page.rect.width,
                          page_height_pt=page.rect.height)


# --------------------------------------------------------------------------
# The batch: an artifact directory in, ground truth out, every artifact
# accounted for

def _write_json(path: Path, payload: dict) -> None:
    """Write-then-replace, like every other persisted artifact here."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    os.replace(tmp, path)


def _print_progress(outcome: dict) -> None:
    line = f"{outcome['artifact']}: {outcome['status']}"
    if outcome["status"] == "failed":
        line += f" — {outcome['reason']}"
    else:
        line += f" (Sheet {outcome['sheet']})"
    print(line, file=sys.stderr, flush=True)


def _discovered(artifact_dir: Path) -> list[Path]:
    """The directory's topology artifacts, in page order. An empty
    directory is refused by name — a typo'd path must not yield an
    empty, plausible-looking dataset."""
    def page_number(path: Path) -> int:
        match = re.match(r"topology_page(\d+)", path.name)
        return int(match.group(1)) if match else sys.maxsize

    paths = sorted(artifact_dir.glob("topology_page*.json"),
                   key=lambda p: (page_number(p), p.name))
    if not paths:
        raise ValueError(
            f"{artifact_dir} holds no topology_page*.json artifacts —"
            f" point the factory at hazop-ai's l1_output directory")
    return paths


def _examples(labels: SheetLabels) -> list[dict]:
    """Every projected label as a pass verdict — ground truth in exactly
    the labeled-example schema reviewing produces (ticket 10) and the
    training export consumes (ticket 12)."""
    return ([make_example("symbol", detection, "pass")
             for detection in labels.symbols]
            + [make_example("line", detection, "pass")
               for detection in labels.lines]
            + [make_example("text", detection, "pass")
               for detection in labels.texts])


def _process_artifact(path: Path, pdf_path: Path, out_dir: Path,
                      profile: Mapping[str, str], dpi: int,
                      synthesis: BoxSynthesis | None,
                      reader: PageReader, store: LabelStore,
                      seen: dict[int, str]) -> dict:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    validate_artifact(artifact, path.name)
    sheet = artifact["page"]
    if sheet in seen:
        raise ValueError(f"{path.name} claims Sheet {sheet}, already"
                         f" produced by {seen[sheet]}")
    render = reader(pdf_path, sheet, dpi)
    scale = dpi / POINTS_PER_INCH
    labels = build_sheet_labels(artifact, list(render.spans), scale,
                                synthesis, source=path.name)

    png_path = out_dir / "sheets" / f"sheet_{sheet}.png"
    png_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.write_bytes(encode_gray_png(render.width, render.height,
                                         render.pixels))
    store.record_many(profile, sheet, _examples(labels), replace=True)
    connectivity_path = out_dir / "connectivity" / f"sheet_{sheet}.json"
    _write_json(connectivity_path, {"profile": dict(profile),
                                    "sheet": sheet,
                                    "links": labels.links})
    seen[sheet] = path.name
    return {
        "artifact": path.name,
        "sheet": sheet,
        "status": "succeeded",
        "render": {"dpi": dpi, "scale": scale,
                   "width_px": render.width, "height_px": render.height,
                   "page_width_pt": render.page_width_pt,
                   "page_height_pt": render.page_height_pt},
        "examples": {"symbols": len(labels.symbols),
                     "texts": len(labels.texts),
                     "lines": len(labels.lines)},
        "links": len(labels.links),
        "paths": {
            "sheet_png": str(png_path),
            "labels": str(out_dir / "labels" / profile_key(profile)
                          / f"sheet_{sheet}.json"),
            "connectivity": str(connectivity_path),
        },
    }


def run_label_factory(artifact_dir: Path | str, pdf_path: Path | str,
                      out_dir: Path | str,
                      profile: Mapping[str, str] | None = None,
                      dpi: int = DEFAULT_DPI,
                      synthesis: BoxSynthesis | None = None,
                      reader: PageReader | None = None,
                      progress: Callable[[dict], None] | None = None,
                      ) -> dict:
    """Batch-process an artifact directory: render each Sheet, project
    its ground truth, and export the training set — reporting per-
    artifact success or failure without stopping the batch, so coverage
    is stated, never implied. Re-running against the same out_dir
    regenerates every Sheet it reproduces in place."""
    artifact_dir, pdf_path = Path(artifact_dir), Path(pdf_path)
    out_dir = Path(out_dir)
    profile = dict(profile) if profile is not None else dict(DEFAULT_PROFILE)
    reader = reader if reader is not None else read_page
    report_progress = progress if progress is not None else _print_progress

    store = LabelStore(out_dir / "labels")
    seen: dict[int, str] = {}
    outcomes = []
    for path in _discovered(artifact_dir):
        try:
            outcome = _process_artifact(path, pdf_path, out_dir, profile,
                                        dpi, synthesis, reader, store,
                                        seen)
        except Exception as error:
            # one bad artifact never costs the batch; an interrupt
            # (BaseException) still stops it
            outcome = {"artifact": path.name, "status": "failed",
                       "reason": f"{type(error).__name__}: {error}"}
        outcomes.append(outcome)
        report_progress(outcome)

    succeeded = sum(1 for o in outcomes if o["status"] == "succeeded")
    training = None
    paths = {"manifest": str(out_dir / "manifest.json"),
             "labels_root": str(out_dir / "labels")}
    if succeeded:
        training_path = (out_dir / "training"
                         / f"{profile_key(profile)}.jsonl")
        training = export_training_set(out_dir / "labels", profile,
                                       training_path,
                                       source=pdf_path.name)
        paths["training"] = str(training_path)

    report = {
        "document": pdf_path.name,
        "profile": profile,
        "dpi": dpi,
        "coverage": {"total_artifacts": len(outcomes),
                     "succeeded": succeeded,
                     "failed": len(outcomes) - succeeded},
        "artifacts": outcomes,
        "training": training,
        "paths": paths,
    }
    _write_json(out_dir / "manifest.json", report)
    return report


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m pidgraph.label_factory",
        description="Render hazop-ai's vector artifacts to raster Sheets"
                    " with projected pixel-level ground-truth labels"
                    " (dev-time tooling; output stays outside git).")
    parser.add_argument("artifact_dir", type=Path,
                        help="hazop-ai's l1_output directory holding"
                             " topology_page<N>.json artifacts")
    parser.add_argument("pdf", type=Path,
                        help="the source vector PDF the artifacts were"
                             " extracted from")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR,
                        help=f"output directory (default:"
                             f" {DEFAULT_OUT_DIR}, which git ignores)")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI,
                        help=f"render resolution (default: {DEFAULT_DPI},"
                             f" hazop-ai's own overlay scale)")
    parser.add_argument("--name", default=DEFAULT_PROFILE["name"],
                        help="Convention Profile identity to stamp on"
                             " the labels")
    parser.add_argument("--version", default=DEFAULT_PROFILE["version"],
                        help="Convention Profile version to stamp on"
                             " the labels")
    args = parser.parse_args(argv)
    report = run_label_factory(args.artifact_dir, args.pdf, args.out,
                               profile={"name": args.name,
                                        "version": args.version},
                               dpi=args.dpi)
    coverage = report["coverage"]
    print(f"{coverage['succeeded']}/{coverage['total_artifacts']}"
          f" artifacts -> {args.out}")
    if report["training"] is not None:
        print(f"training set: {report['training']['count']} examples"
              f" -> {report['paths']['training']}")


if __name__ == "__main__":
    main()
