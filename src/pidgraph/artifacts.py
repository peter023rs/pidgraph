"""Run-artifact readers for the review side — the Review Workbench
(ticket 10/11) and the product KPI (ticket 19). What a run leaves on disk
under <run_dir>/ — detections/sheet_N.json per Sheet, sheets/sheet_N.png,
the DEXPI plant model, a batch run's state — is read back here by Sheet
number, with nothing from the extraction engine imported: the review side
has no path to invoke extraction.

A Sheet's identity is its artifact filename (the number the Workbench
serves it under), so every review-side tool keys verdicts, queues,
progress and KPI one way even if a record's internal "sheet" field
disagrees with its filename.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

# Detection kinds as the record fields that hold them — the distinction
# the Workbench's queues scope by and the KPI reports per kind.
KINDS = (("symbol", "symbols"), ("line", "lines"), ("text", "texts"))

_SHEET_FILE = re.compile(r"sheet_(\d+)\.json")


class RecordSummary(NamedTuple):
    """All the review side needs of a detection record to count verdict
    coverage: whose Convention Profile it was digitized under, and which
    detections it contains."""
    profile: dict
    ids: frozenset[str]


def review_state(decided: int, total: int) -> str:
    """One Sheet's review state from verdict coverage (ticket 11) — the
    one rule the Workbench's progress view and the KPI's acceptance both
    derive from. A Sheet with nothing detected needs no reviewer minutes,
    so it counts as reviewed."""
    if decided == total:
        return "reviewed"
    if decided == 0:
        return "untouched"
    return "in progress"


def iter_detections(record: dict) -> Iterator[tuple[str, dict]]:
    for kind, field in KINDS:
        for detection in record[field]:
            yield kind, detection


def record_path(run_dir: Path | str, number: int) -> Path:
    return Path(run_dir) / "detections" / f"sheet_{number}.json"


def sheet_numbers(run_dir: Path | str) -> list[int]:
    """The Sheets a run left detection records for, by artifact filename.
    Stray files beside the records (editor leftovers, interrupted .tmp
    writes) are not Sheets."""
    return sorted(
        int(match.group(1))
        for path in (Path(run_dir) / "detections").glob("sheet_*.json")
        if (match := _SHEET_FILE.fullmatch(path.name)))


def load_record(run_dir: Path | str, number: int) -> dict | None:
    path = record_path(run_dir, number)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


class RecordSummaries:
    """Each record's Convention Profile and detection ids, cached against
    the record file's stat: detection records never change while a run is
    under review, and re-parsing full geometry for every request of a
    400-Sheet index would be wasteful. Verdicts are deliberately never
    cached anywhere on the review side — review state and KPI are
    recomputed from the persisted store on every request."""

    def __init__(self, run_dir: Path | str):
        self.run_dir = Path(run_dir)
        self._cache: dict[int, tuple[tuple[int, int], RecordSummary]] = {}

    def get(self, number: int) -> RecordSummary | None:
        """None if the record is absent. A record that cannot be read
        raises (ValueError for truncated JSON — a run killed mid-write —
        KeyError/TypeError for a record missing its fields)."""
        path = record_path(self.run_dir, number)
        try:
            stat = path.stat()
        except OSError:
            return None
        key = (stat.st_mtime_ns, stat.st_size)
        cached = self._cache.get(number)
        if cached is not None and cached[0] == key:
            return cached[1]
        record = json.loads(path.read_text(encoding="utf-8"))
        summary = RecordSummary(
            record["profile"],
            frozenset(d["id"] for _, d in iter_detections(record)))
        self._cache[number] = (key, summary)
        return summary


def load_sheets(run_dir: Path | str,
                summaries: RecordSummaries | None = None,
                ) -> list[tuple[int, RecordSummary | None]]:
    """Every Sheet the run left a record for, in number order, each with
    its summary — or None where the record is unreadable, so one bad
    Sheet degrades to one row instead of taking down a whole Document's
    view. Pass a long-lived RecordSummaries to reuse its cache."""
    if summaries is None:
        summaries = RecordSummaries(run_dir)
    sheets: list[tuple[int, RecordSummary | None]] = []
    for number in sheet_numbers(run_dir):
        try:
            summary = summaries.get(number)
        except (ValueError, KeyError, TypeError):
            sheets.append((number, None))
            continue
        if summary is None:  # vanished between listing and reading
            continue
        sheets.append((number, summary))
    return sheets


def document_identifier(run_dir: Path | str) -> str:
    """The Document a run digitized, by identifier: the batch manifest's
    document name, else the plant model's sourceDrawing, else the run
    directory's own name. An identifier only — never drawing content."""
    run_dir = Path(run_dir)
    for path, key in ((run_dir / "state" / "manifest.json", "document"),
                      (run_dir / "plant_model_dexpi.json", "sourceDrawing")):
        try:
            value = json.loads(path.read_text(encoding="utf-8")).get(key)
        except (OSError, ValueError, AttributeError):
            continue
        if isinstance(value, str) and value:
            return value
    return run_dir.name
