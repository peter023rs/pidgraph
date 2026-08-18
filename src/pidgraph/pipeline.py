"""digitize(Document, Convention Profile) — the one top seam.

Raster Path per Sheet: normalization -> symbol detection -> line network
extraction -> OCR -> lexicon-constrained decoding -> graph assembly ->
DEXPI JSON emission -> graph store. Each later ticket deepens one stage;
the path itself is fixed here.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from .assemble import SheetAssembly, assemble_sheet, build_plant_graph
from .dexpi import emit_plant_model
from .lexicon import decode_tags
from .lines import extract_line_network
from .model import (
    ConventionProfile,
    Document,
    LineDetection,
    Provenance,
    RunArtifacts,
    Sheet,
    SymbolDetection,
    TextDetection,
)
from .normalize import (
    Normalization,
    map_detections_to_original,
    normalize_sheet,
)
from .seams import (
    PipelineConfig,
    SymbolDetector,
    TextRecognizer,
    build_components,
)


def _detection_record(sheet: Sheet,
                      profile: ConventionProfile,
                      normalization: Normalization,
                      symbols: list[SymbolDetection],
                      lines: list[LineDetection],
                      texts: list[TextDetection]) -> dict:
    return {
        "sheet": sheet.number,
        "profile": profile.identity_record(),
        "normalization": normalization.as_record(),
        "symbols": [asdict(s) for s in symbols],
        "lines": [asdict(l) for l in lines],
        "texts": [asdict(t) for t in texts],
    }


def digitize_sheet(sheet: Sheet,
                   profile: ConventionProfile,
                   detector: SymbolDetector,
                   recognizer: TextRecognizer,
                   ) -> tuple[dict, SheetAssembly]:
    """One Sheet through the Raster Path: its detection record (in
    original Sheet coordinates) and its assembled topology."""
    normalized, normalization = normalize_sheet(sheet)
    symbols = detector.detect(normalized, profile)
    lines = extract_line_network(normalized, symbols, profile)
    texts = decode_tags(recognizer.recognize(normalized, profile),
                        profile.tag_grammar)
    # extraction ran in the normalized frame; everything recorded or
    # assembled from here on is in original Sheet coordinates
    symbols, lines, texts = map_detections_to_original(
        normalization, symbols, lines, texts)
    record = _detection_record(sheet, profile, normalization,
                               symbols, lines, texts)
    return record, assemble_sheet(sheet, symbols, lines, texts, profile)


def detections_from_record(record: dict) -> tuple[
        list[SymbolDetection], list[LineDetection], list[TextDetection]]:
    """Rebuild the detection dataclasses from a detection record — the
    inverse of _detection_record's detection lists, so a resumed batch run
    (ticket 08) reassembles a completed Sheet without re-running
    extraction. Values pass through verbatim (a JSON round trip preserves
    them); only the list-vs-tuple shape is restored."""
    symbols = [
        SymbolDetection(
            id=s["id"], sheet=s["sheet"], symbol_class=s["symbol_class"],
            bbox=tuple(s["bbox"]), confidence=s["confidence"],
            provenance=Provenance(**s["provenance"]),
            direction=(None if s["direction"] is None
                       else tuple(s["direction"])))
        for s in record["symbols"]]
    lines = [
        LineDetection(
            id=l["id"], sheet=l["sheet"],
            polyline=tuple(tuple(p) for p in l["polyline"]),
            line_class=l["line_class"], confidence=l["confidence"],
            provenance=Provenance(**l["provenance"]))
        for l in record["lines"]]
    texts = [
        TextDetection(
            id=t["id"], sheet=t["sheet"], string=t["string"],
            text_class=t["text_class"], bbox=tuple(t["bbox"]),
            confidence=t["confidence"],
            provenance=Provenance(**t["provenance"]),
            raw_string=t["raw_string"], correction=t["correction"],
            resolved=t["resolved"], candidates=tuple(t["candidates"]))
        for t in record["texts"]]
    return symbols, lines, texts


def write_run_outputs(out_dir: Path, plant_model: dict,
                      records: Sequence[dict],
                      store_summary: dict) -> dict[str, str]:
    """Write the file outputs a run leaves behind — the DEXPI plant model
    and the per-Sheet detection records — returning their paths (plus the
    store's, when it wrote one)."""
    paths: dict[str, str] = {}
    model_path = out_dir / "plant_model_dexpi.json"
    model_path.write_text(
        json.dumps(plant_model, ensure_ascii=False, indent=1),
        encoding="utf-8")
    paths["plant_model"] = str(model_path)
    detections_dir = out_dir / "detections"
    detections_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        record_path = detections_dir / f"sheet_{record['sheet']}.json"
        record_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=1),
            encoding="utf-8")
        paths[f"detections/sheet_{record['sheet']}"] = str(record_path)
    if store_summary.get("path"):
        paths["plant_graph"] = store_summary["path"]
    return paths


def digitize(document: Document,
             profile: ConventionProfile,
             config: PipelineConfig | None = None,
             out_dir: Path | str | None = None) -> RunArtifacts:
    config = config or PipelineConfig()
    detector, recognizer, store = build_components(config)

    records = []
    assemblies = []
    for sheet in document.sheets:
        record, assembly = digitize_sheet(sheet, profile, detector,
                                          recognizer)
        records.append(record)
        assemblies.append(assembly)

    plant_graph = build_plant_graph(assemblies)
    plant_model = emit_plant_model(assemblies, document, profile)

    out_dir = Path(out_dir) if out_dir is not None else None
    store_summary = store.store(plant_graph, out_dir)

    paths: dict[str, str] = {}
    if out_dir is not None:
        paths = write_run_outputs(out_dir, plant_model, records,
                                  store_summary)

    return RunArtifacts(
        detection_records=tuple(records),
        plant_model=plant_model,
        plant_graph=plant_graph,
        store_summary=store_summary,
        paths=paths,
    )
