"""Batch runs over a multi-Sheet Document (ticket 08).

run_batch() is digitize()'s operator-scale sibling: the same Raster Path,
but Sheet by Sheet — progress is reported per Sheet while the run moves,
each completed Sheet's detection record is persisted under
<run_dir>/state/ the moment it exists, and the run report accounts for
every Sheet of the Document, so coverage is stated, never implied.

A Sheet that fails is recorded with its reason and the batch continues;
an interrupt (KeyboardInterrupt) stops the run, and starting it again on
the same run directory resumes: completed Sheets are skipped and their
persisted records feed the final assembly, so a resumed run converges to
the artifacts an uninterrupted one writes. Assembly always rebuilds from
the persisted record form — fresh and resumed Sheets share one path.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal

from .assemble import assemble_sheet, build_plant_graph
from .dexpi import emit_plant_model
from .model import ConventionProfile, Document, RunArtifacts
from .pipeline import (
    detections_from_record,
    digitize_sheet,
    write_run_outputs,
)
from .seams import PipelineConfig, build_components


# Default confidence floor: an extraction below it is reported as a gap
# for the operator to check first.
LOW_CONFIDENCE_THRESHOLD = 0.7


class BatchStateError(Exception):
    """Persisted batch state that cannot be resumed against, with the
    reason spelled out."""


SheetStatus = Literal["started", "succeeded", "failed", "skipped-by-resume"]


@dataclass(frozen=True)
class SheetProgress:
    """One per-Sheet progress event, emitted while the run moves."""
    sheet: int
    index: int   # 1-based position in the Document
    total: int
    status: SheetStatus
    reason: str | None = None  # failed only


ProgressCallback = Callable[[SheetProgress], None]


@dataclass(frozen=True)
class BatchRunResult:
    """What a completed batch run hands the operator: the run report plus
    the run artifacts."""
    report: dict
    artifacts: RunArtifacts


def _print_progress(event: SheetProgress) -> None:
    line = (f"Sheet {event.sheet} ({event.index}/{event.total}): "
            f"{event.status}")
    if event.reason is not None:
        line += f" — {event.reason}"
    print(line, file=sys.stderr, flush=True)


def _write_json(path: Path, payload: dict) -> None:
    """Write-then-rename, so an interrupt never leaves a truncated file
    that a resumed run would mistake for state."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    os.replace(tmp, path)


def _profile_fingerprint(profile: ConventionProfile) -> str:
    """Content hash of the Convention Profile's parts. The manifest pins
    it alongside name and version, so a profile edited under an unchanged
    version cannot silently mix into a resumed run."""
    payload = {
        "legend": {name: asdict(entry)
                   for name, entry in profile.legend.items()},
        "tag_grammar": dict(profile.tag_grammar),
        "line_semantics": dict(profile.line_semantics),
        "line_styles": dict(profile.line_styles),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False,
                   sort_keys=True).encode("utf-8")).hexdigest()


def _manifest(document: Document, profile: ConventionProfile,
              config: PipelineConfig) -> dict:
    return {
        "document": document.name,
        "sheets": [sheet.number for sheet in document.sheets],
        "profile": profile.identity_record()
                   | {"content_fingerprint": _profile_fingerprint(profile)},
        "config": asdict(config),
    }


def _check_manifest(state_dir: Path, manifest: dict) -> None:
    """A run directory belongs to one (Document, Convention Profile,
    configuration) triple. Resuming against state from a different run
    is refused — mixed state could not converge to an uninterrupted
    run's artifacts."""
    path = state_dir / "manifest.json"
    if not path.exists():
        _write_json(path, manifest)
        return
    existing = json.loads(path.read_text(encoding="utf-8"))
    if existing == manifest:
        return
    differing = ", ".join(sorted(
        key for key in manifest if existing.get(key) != manifest[key]))
    raise BatchStateError(
        f"run directory {state_dir.parent} holds state for a different "
        f"run ({differing} differ) — it was started as "
        f"{json.dumps(existing, ensure_ascii=False)}. Resume with the "
        f"same Document, Convention Profile, and configuration, or start "
        f"a fresh run directory")


def _gap(detection: dict, kind: str, **fields) -> dict:
    return {"sheet": detection["sheet"], "kind": kind,
            "id": detection["id"], "confidence": detection["confidence"],
            "detail": detection["provenance"]["evidence"], **fields}


def _collect_gaps(records: list[dict], threshold: float) -> list[dict]:
    """Low-confidence extractions and fail-closed (unresolved) tag reads,
    per Sheet record — the report's gaps, so coverage is honest."""
    gaps = []
    for record in records:
        for text in record["texts"]:
            if not text["resolved"]:
                gaps.append(_gap(text, "unresolved-text",
                                 text_class=text["text_class"],
                                 string=text["string"]))
            elif text["confidence"] < threshold:
                gaps.append(_gap(text, "low-confidence",
                                 text_class=text["text_class"],
                                 string=text["string"]))
        for symbol in record["symbols"]:
            if symbol["confidence"] < threshold:
                gaps.append(_gap(symbol, "low-confidence",
                                 symbol_class=symbol["symbol_class"]))
        for line in record["lines"]:
            if line["confidence"] < threshold:
                gaps.append(_gap(line, "low-confidence",
                                 line_class=line["line_class"]))
    return gaps


def _sheet_state_path(state_dir: Path, number: int) -> Path:
    return state_dir / f"sheet_{number}.json"


def _load_sheet_state(state_dir: Path, number: int) -> dict | None:
    path = _sheet_state_path(state_dir, number)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise BatchStateError(
            f"state file {path} is unreadable ({error}); delete it to "
            f"re-process Sheet {number}") from None


def run_batch(document: Document,
              profile: ConventionProfile,
              run_dir: Path | str,
              config: PipelineConfig | None = None,
              progress: ProgressCallback | None = None,
              low_confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD,
              ) -> BatchRunResult:
    """Run a Document as a resumable batch with per-Sheet progress and an
    honest run report (failures with reasons, low-confidence gaps, every
    Sheet accounted for)."""
    config = config or PipelineConfig()
    report_progress = progress if progress is not None else _print_progress
    run_dir = Path(run_dir)
    state_dir = run_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    _check_manifest(state_dir, _manifest(document, profile, config))

    detector, recognizer, store = build_components(config)

    total = len(document.sheets)
    outcomes: list[dict] = []
    records: dict[int, dict] = {}
    for index, sheet in enumerate(document.sheets, start=1):
        state = _load_sheet_state(state_dir, sheet.number)
        if state is not None and state["status"] == "succeeded":
            records[sheet.number] = state["record"]
            outcomes.append({"sheet": sheet.number,
                             "status": "skipped-by-resume"})
            report_progress(SheetProgress(sheet.number, index, total,
                                          "skipped-by-resume"))
            continue
        report_progress(SheetProgress(sheet.number, index, total, "started"))
        try:
            # digitize_sheet's assembly is dropped, but not wasted: it
            # proves the record assembles inside this Sheet's failure
            # isolation, so the end-of-run reassembly below cannot fail
            record, _ = digitize_sheet(sheet, profile, detector, recognizer)
        except Exception as error:
            # one bad Sheet never costs the run; an interrupt
            # (BaseException) still stops it
            reason = f"{type(error).__name__}: {error}"
            _write_json(_sheet_state_path(state_dir, sheet.number),
                        {"sheet": sheet.number, "status": "failed",
                         "reason": reason})
            outcomes.append({"sheet": sheet.number, "status": "failed",
                             "reason": reason})
            report_progress(SheetProgress(sheet.number, index, total,
                                          "failed", reason))
            continue
        _write_json(_sheet_state_path(state_dir, sheet.number),
                    {"sheet": sheet.number, "status": "succeeded",
                     "record": record})
        records[sheet.number] = record
        outcomes.append({"sheet": sheet.number, "status": "succeeded"})
        report_progress(SheetProgress(sheet.number, index, total,
                                      "succeeded"))

    # Assembly rebuilds from the persisted record form for fresh and
    # resumed Sheets alike — one path, so a resumed run converges on the
    # artifacts an uninterrupted one writes.
    ordered_records = [records[sheet.number] for sheet in document.sheets
                       if sheet.number in records]
    assemblies = [
        assemble_sheet(sheet, *detections_from_record(records[sheet.number]),
                       profile)
        for sheet in document.sheets if sheet.number in records]

    plant_graph = build_plant_graph(assemblies)
    plant_model = emit_plant_model(assemblies, document, profile)
    store_summary = store.store(plant_graph, run_dir)
    paths = write_run_outputs(run_dir, plant_model, ordered_records,
                              store_summary,
                              [sheet for sheet in document.sheets
                               if sheet.number in records])

    counts = {"succeeded": 0, "failed": 0, "skipped-by-resume": 0}
    for outcome in outcomes:
        counts[outcome["status"]] += 1
    report = {
        "document": document.name,
        "profile": profile.identity_record(),
        "low_confidence_threshold": low_confidence_threshold,
        "coverage": {
            "total_sheets": total,
            "succeeded": counts["succeeded"],
            "failed": counts["failed"],
            "skipped_by_resume": counts["skipped-by-resume"],
            "digitized": counts["succeeded"] + counts["skipped-by-resume"],
        },
        "sheets": outcomes,
        "failures": [{"sheet": o["sheet"], "reason": o["reason"]}
                     for o in outcomes if o["status"] == "failed"],
        "gaps": _collect_gaps(ordered_records, low_confidence_threshold),
    }
    report_path = run_dir / "run_report.json"
    _write_json(report_path, report)
    paths["run_report"] = str(report_path)

    artifacts = RunArtifacts(
        detection_records=tuple(ordered_records),
        plant_model=plant_model,
        plant_graph=plant_graph,
        store_summary=store_summary,
        paths=paths,
    )
    return BatchRunResult(report=report, artifacts=artifacts)
