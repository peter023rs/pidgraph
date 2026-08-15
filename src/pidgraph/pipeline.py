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

from .assemble import assemble_sheet, build_plant_graph
from .dexpi import emit_plant_model
from .lexicon import decode_tags
from .lines import extract_line_network
from .model import (
    ConventionProfile,
    Document,
    LineDetection,
    RunArtifacts,
    Sheet,
    SymbolDetection,
    TextDetection,
)
from .seams import PipelineConfig, build_components


def _normalize(sheet: Sheet) -> Sheet:
    """Raster normalization (deskew, resolution, binarize) — identity until
    ticket 03."""
    return sheet


def _detection_record(sheet: Sheet,
                      symbols: list[SymbolDetection],
                      lines: list[LineDetection],
                      texts: list[TextDetection]) -> dict:
    return {
        "sheet": sheet.number,
        "symbols": [asdict(s) for s in symbols],
        "lines": [asdict(l) for l in lines],
        "texts": [asdict(t) for t in texts],
    }


def digitize(document: Document,
             profile: ConventionProfile,
             config: PipelineConfig | None = None,
             out_dir: Path | str | None = None) -> RunArtifacts:
    config = config or PipelineConfig()
    detector, recognizer, store = build_components(config)

    records = []
    assemblies = []
    for sheet in document.sheets:
        sheet = _normalize(sheet)
        symbols = detector.detect(sheet, profile)
        lines = extract_line_network(sheet)
        texts = decode_tags(recognizer.recognize(sheet, profile),
                            profile.tag_grammar)
        records.append(_detection_record(sheet, symbols, lines, texts))
        assemblies.append(assemble_sheet(sheet, symbols, lines, texts,
                                         profile))

    plant_graph = build_plant_graph(assemblies)
    plant_model = emit_plant_model(assemblies, document, profile)

    out_dir = Path(out_dir) if out_dir is not None else None
    store_summary = store.store(plant_graph, out_dir)

    paths: dict[str, str] = {}
    if out_dir is not None:
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

    return RunArtifacts(
        detection_records=tuple(records),
        plant_model=plant_model,
        plant_graph=plant_graph,
        store_summary=store_summary,
        paths=paths,
    )
