"""Reviewer verdicts as labeled examples (ticket 10): every pass /
reject / edit taken in the Review Workbench persists as a labeled
example keyed to Convention Profile (identity + version) and Sheet —
reviewing is simultaneously training-data creation. Examples carry
drawing-derived content, so they live next to the run artifacts outside
git (ADR-0001); ticket 12 exports them as per-profile training sets."""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Iterable, Mapping
from urllib.parse import quote

VERDICTS = ("pass", "reject", "edit")


def _check_number(value: object) -> bool:
    # JSON parsing lets NaN/Infinity through; a non-finite coordinate
    # would poison the training export (ticket 12)
    return (isinstance(value, (int, float))
            and not isinstance(value, bool) and math.isfinite(value))


def _check_bbox(value: object) -> None:
    if not (isinstance(value, list) and len(value) == 4
            and all(_check_number(c) for c in value)):
        raise ValueError(f"a corrected bbox is [x0, y0, x1, y1],"
                         f" got {value!r}")


def _check_polyline(value: object) -> None:
    if not (isinstance(value, list) and len(value) >= 2
            and all(isinstance(p, list) and len(p) == 2
                    and all(_check_number(c) for c in p)
                    for p in value)):
        raise ValueError(f"a corrected polyline is [[x, y], ...] with at"
                         f" least two points, got {value!r}")


def _check_string(value: object) -> None:
    if not (isinstance(value, str) and value):
        raise ValueError(f"a corrected tag text is a non-empty string,"
                         f" got {value!r}")


# What an edit may correct, per detection kind: geometry for all three,
# tag text only where there is text to correct.
_CORRECTION_CHECKS = {
    "symbol": {"bbox": _check_bbox},
    "line": {"polyline": _check_polyline},
    "text": {"bbox": _check_bbox, "string": _check_string},
}


def make_example(kind: str, detection: dict, verdict: object,
                 correction: object = None) -> dict:
    """One verdict validated into its labeled-example form. Pass and
    reject stand alone; an edit supplies the corrected tag text or
    geometry, and the correction must fit the detection kind."""
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {list(VERDICTS)},"
                         f" got {verdict!r}")
    if verdict != "edit":
        if correction is not None:
            raise ValueError(f"a {verdict} verdict carries no correction")
        return {"verdict": verdict, "kind": kind,
                "detection": detection, "correction": None}
    checks = _CORRECTION_CHECKS[kind]
    if not isinstance(correction, dict) or not correction:
        raise ValueError(f"an edit supplies a correction with"
                         f" {sorted(checks)}, got {correction!r}")
    for key, value in correction.items():
        if key not in checks:
            raise ValueError(f"a {kind} correction allows"
                             f" {sorted(checks)}, not {key!r}")
        checks[key](value)
    return {"verdict": verdict, "kind": kind,
            "detection": detection, "correction": correction}


def profile_key(profile: Mapping[str, str]) -> str:
    """One directory per Convention Profile identity + version; both
    parts percent-quoted so arbitrary identity strings cannot escape the
    labels root or collide across the '@' separator."""
    return (quote(profile["name"], safe="") + "@"
            + quote(profile["version"], safe=""))


class LabelStore:
    """Labeled examples on disk — <root>/<name>@<version>/sheet_<N>.json,
    one file per (Convention Profile, Sheet) holding the latest verdict
    for each detection. Write-then-replace, so an interrupted save never
    leaves a truncated store behind."""

    def __init__(self, root: Path | str):
        self.root = Path(root)

    def _sheet_path(self, profile: Mapping[str, str], sheet: int) -> Path:
        return self.root / profile_key(profile) / f"sheet_{sheet}.json"

    def profiles(self) -> list[str]:
        """The store's partitions — quoted name@version directory keys."""
        if not self.root.is_dir():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def sheets(self, profile: Mapping[str, str]) -> list[int]:
        """Sheet numbers the profile holds labeled examples for. The
        store owns its on-disk layout: stray files beside the sheet
        stores (editor leftovers, interrupted .tmp writes) are not
        sheets."""
        pattern = re.compile(r"sheet_(\d+)\.json")
        directory = self.root / profile_key(profile)
        return sorted(
            int(match.group(1))
            for path in directory.glob("sheet_*.json")
            if (match := pattern.fullmatch(path.name)))

    def sheet_labels(self, profile: Mapping[str, str], sheet: int) -> dict:
        path = self._sheet_path(profile, sheet)
        if not path.is_file():
            return {"profile": dict(profile), "sheet": sheet,
                    "examples": {}}
        return json.loads(path.read_text(encoding="utf-8"))

    def record(self, profile: Mapping[str, str], sheet: int,
               example: dict) -> dict:
        self.record_many(profile, sheet, [example])
        return example

    def record_many(self, profile: Mapping[str, str], sheet: int,
                    examples: Iterable[dict],
                    replace: bool = False) -> int:
        """All of one Sheet's new examples in a single write — the label
        factory (ticket 13) records hundreds per Sheet, where one
        write-then-replace each would rewrite the growing store
        quadratically. replace=True starts the Sheet over instead of
        merging, so a regenerated ground-truth Sheet carries no stale
        examples."""
        labels: dict = ({"profile": dict(profile), "sheet": sheet,
                         "examples": {}} if replace
                        else self.sheet_labels(profile, sheet))
        count = 0
        for example in examples:
            labels["examples"][example["detection"]["id"]] = example
            count += 1
        path = self._sheet_path(profile, sheet)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(labels, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        os.replace(tmp, path)
        return count
