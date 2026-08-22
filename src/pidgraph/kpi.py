"""Product KPI (ticket 19): corrections per 100 symbols and reviewer
minutes per accepted Sheet, computed from Review Workbench activity, so
"the software works" is measurable on real corpus Sheets. Together with
the eval harness's pinned component gates (ticket 15), this is the
definition of "works" that triggers hazop-ai reintegration.

Both numbers are recomputed from what the Workbench persists — the
verdict store (labels.LabelStore) and the append-only activity log kept
beside it — plus the run's detection records; never from session memory.
The measurement basis travels with every report, because a minutes figure
without its rules is not a KPI. Reports name Sheets, Documents and
Convention Profiles by identifier and carry counts and minutes only: no
drawing content leaves the local store (ADR-0001).
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, NamedTuple, Sequence

from .artifacts import (
    KINDS,
    RecordSummaries,
    RecordSummary,
    document_identifier,
    load_sheets,
    review_state,
)
from .labels import VERDICTS, LabelStore

# An interval between two consecutive Workbench events longer than this
# is a break, not review time. Ten minutes: a verdict is a click, and
# studying a Sheet before the first one takes a minute or three — anything
# longer is the reviewer away from the Workbench. Erring long is the
# conservative direction: it can only make the minutes look worse.
DEFAULT_IDLE_MINUTES = 10.0

ACTIVITY_FILE = "activity.jsonl"

Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Event(NamedTuple):
    """One Workbench activity event — a Sheet opened for review ("open")
    or a verdict saved ("verdict") — at the server's timestamp."""
    at: datetime
    event: str
    sheet: int


class ActivityLog:
    """Workbench activity as one JSON line per event, appended to
    <labels root>/activity.jsonl — beside the verdicts, outside git with
    the rest of the run's review data (ADR-0001). The verdict store keeps
    only the latest verdict per detection; the log keeps every action with
    its time, which is what reviewer minutes are measured from."""

    def __init__(self, root: Path | str, clock: Clock | None = None):
        self.path = Path(root) / ACTIVITY_FILE
        self._clock = clock if clock is not None else _utc_now

    def record(self, event: str, sheet: int, **fields: object) -> dict:
        entry: dict = {"at": self._clock().isoformat(), "event": event,
                       "sheet": sheet, **fields}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def events(self) -> tuple[list[Event], int]:
        """Every readable event in time order, plus the count of lines
        that could not be read (a process killed mid-append leaves a
        partial last line) — counted in the report, never silently
        dropped."""
        if not self.path.is_file():
            return [], 0
        events: list[Event] = []
        unreadable = 0
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = _parse_event(line)
            if event is None:
                unreadable += 1
            else:
                events.append(event)
        events.sort(key=lambda e: e.at)
        return events, unreadable


def _parse_event(line: str) -> Event | None:
    try:
        entry = json.loads(line)
        at = datetime.fromisoformat(entry["at"])
        event, sheet = entry["event"], entry["sheet"]
    except (ValueError, KeyError, TypeError):
        return None
    if not (isinstance(event, str) and isinstance(sheet, int)):
        return None
    if at.tzinfo is None:  # the Workbench stamps UTC; a hand-edited
        at = at.replace(tzinfo=timezone.utc)  # naive time is read as such
    return Event(at, event, sheet)


def attribute_minutes(events: Iterable[Event],
                      idle_minutes: float) -> dict[int, float]:
    """Reviewer minutes per Sheet from the activity timeline. The
    timeline splits at every event; each interval of at most idle_minutes
    belongs to the Sheet of the event that opened it — from an action on a
    Sheet until the next action anywhere, the reviewer is on that Sheet.
    A longer interval is a break and counts nothing; the time after the
    last event is unmeasured. One timeline, so interleaved Sheets never
    double-count."""
    ordered = sorted(events, key=lambda e: e.at)
    minutes: dict[int, float] = defaultdict(float)
    for prev, nxt in zip(ordered, ordered[1:]):
        gap = (nxt.at - prev.at).total_seconds() / 60.0
        if gap <= idle_minutes:
            minutes[prev.sheet] += gap
    return dict(minutes)


KIND_NAMES = tuple(kind for kind, _ in KINDS)
CORRECTIONS = ("reject", "edit")


def _counts() -> dict[str, int]:
    return {"reviewed": 0, "pass": 0, "reject": 0, "edit": 0,
            "corrections": 0}


def _by_kind() -> dict[str, dict[str, int]]:
    return {kind: _counts() for kind in KIND_NAMES}


def _identity(profile: Mapping[str, str]) -> tuple[str, str]:
    return profile["name"], profile["version"]


def _sheet_row(number: int, summary: RecordSummary | None,
               examples: Mapping[str, dict], minutes: float,
               timed: bool) -> dict:
    """One Sheet's KPI inputs: verdict counts per detection kind over the
    detections this Sheet's record contains (a stale label from another
    run's artifacts counts nowhere), acceptance, and its reviewer minutes.
    An unreadable record is a row with detections None — one bad Sheet,
    not a broken Document. A stored example whose kind or verdict the
    store's contract does not allow (tampered with outside the Workbench)
    is refused by name, never counted under a guess."""
    if summary is None:
        return {"sheet": number, "profile": None, "detections": None,
                "decided": 0, "accepted": False, "timed": timed,
                "minutes": minutes, "by_kind": _by_kind()}
    by_kind = _by_kind()
    decided = 0
    for detection_id in sorted(summary.ids):
        stored = examples.get(detection_id)
        if stored is None:
            continue
        kind, verdict = stored["kind"], stored["verdict"]
        if kind not in by_kind or verdict not in VERDICTS:
            raise ValueError(
                f"Sheet {number} example {detection_id!r} has kind"
                f" {kind!r} and verdict {verdict!r}; a labeled example is"
                f" one of {list(KIND_NAMES)} with a verdict in"
                f" {list(VERDICTS)}")
        decided += 1
        by_kind[kind]["reviewed"] += 1
        by_kind[kind][verdict] += 1
        if verdict in CORRECTIONS:
            by_kind[kind]["corrections"] += 1
    total = len(summary.ids)
    return {"sheet": number, "profile": summary.profile,
            "detections": total, "decided": decided,
            # accepted: the Workbench's "reviewed" state, on a Sheet with
            # something to review — a blank Sheet was never reviewed, so
            # it is never an accepted Sheet
            "accepted": (total > 0
                         and review_state(decided, total) == "reviewed"),
            "timed": timed, "minutes": minutes, "by_kind": by_kind}


def _metrics(rows: Sequence[dict]) -> dict:
    """The KPI over a set of Sheet rows — one shape for a Document, a
    Convention Profile, and everything together."""
    by_kind = _by_kind()
    sheets = {"total": 0, "accepted": 0, "accepted_timed": 0,
              "accepted_untimed": 0, "unreadable": 0}
    minutes = {"total": 0.0, "on_timed_accepted_sheets": 0.0}
    for row in rows:
        sheets["total"] += 1
        if row["detections"] is None:
            sheets["unreadable"] += 1
        for kind, counts in row["by_kind"].items():
            for key, value in counts.items():
                by_kind[kind][key] += value
        minutes["total"] += row["minutes"]
        if row["accepted"]:
            sheets["accepted"] += 1
            if row["timed"]:
                sheets["accepted_timed"] += 1
                minutes["on_timed_accepted_sheets"] += row["minutes"]
            else:
                sheets["accepted_untimed"] += 1
    symbols = by_kind["symbol"]
    return {
        "corrections_per_100_symbols":
            (100.0 * symbols["corrections"] / symbols["reviewed"]
             if symbols["reviewed"] else None),
        # over the accepted Sheets the activity log timed: an accepted
        # Sheet with no logged verdict is counted, never zeroed in
        "reviewer_minutes_per_accepted_sheet":
            (minutes["on_timed_accepted_sheets"] / sheets["accepted_timed"]
             if sheets["accepted_timed"] else None),
        "by_kind": by_kind,
        "sheets_summary": sheets,
        "minutes": minutes,
    }


def basis(idle_minutes: float) -> dict:
    """The measurement basis, stated alongside the numbers."""
    return {
        "idle_threshold_minutes": idle_minutes,
        "corrections_per_100_symbols":
            "100 × (reject + edit verdicts on symbol detections) / symbol"
            " detections with a verdict — the latest verdict per detection"
            " in the persisted verdict store, counting only detections the"
            " run's artifacts contain. Lines and texts are reported beside"
            " it per kind, never mixed into it.",
        "reviewer_minutes_per_accepted_sheet":
            "Reviewer minutes on timed accepted Sheets / timed accepted"
            " Sheets. An accepted Sheet has at least one detection and a"
            " verdict on every detection; it is timed when at least one of"
            " its verdicts was saved through the Workbench, whose log holds"
            " every Sheet opened and every verdict saved at the server's"
            " clock. The timeline splits at each event, and each interval"
            f" of at most {idle_minutes:g} minutes belongs to the Sheet of"
            " the event that opened it; a longer interval is a break and"
            " counts nothing; time after the last logged event is"
            " unmeasured. Accepted Sheets with no logged verdict are"
            " counted but not timed.",
    }


def run_kpi(run_dir: Path | str, labels_root: Path | str | None = None,
            idle_minutes: float = DEFAULT_IDLE_MINUTES,
            summaries: RecordSummaries | None = None) -> dict:
    """One Document's KPI from its run directory: the detection records,
    the verdict store (labels/ inside the run directory unless the
    Workbench was given another labels_root) and the activity log beside
    the verdicts. Recomputed in full on every call."""
    run_dir = Path(run_dir)
    labels_root = (run_dir / "labels" if labels_root is None
                   else Path(labels_root))
    store = LabelStore(labels_root)
    events, unreadable = ActivityLog(labels_root).events()
    minutes = attribute_minutes(events, idle_minutes)
    # a Sheet is timed by the log when a verdict on it was saved through
    # the Workbench — opening it to look is not reviewing it
    timed = {event.sheet for event in events if event.event == "verdict"}

    rows = []
    for number, summary in load_sheets(run_dir, summaries):
        examples = (store.sheet_labels(summary.profile, number)["examples"]
                    if summary is not None else {})
        rows.append(_sheet_row(number, summary, examples,
                               minutes.get(number, 0.0), number in timed))
    known = {row["sheet"] for row in rows}
    profiles = {_identity(p): p
                for p in (row["profile"] for row in rows) if p}
    return {
        "document": document_identifier(run_dir),
        "run_dir": str(run_dir),
        "labels_root": str(labels_root),
        "profiles": [profiles[key] for key in sorted(profiles)],
        "idle_threshold_minutes": idle_minutes,
        **_metrics(rows),
        "activity_log": {
            "events": len(events),
            "unreadable_lines": unreadable,
            # activity on a Sheet number this run has no record for (a
            # renamed artifact, a log from another run) is real reviewer
            # time, accounted for here rather than dropped
            "minutes_outside_artifacts": sum(
                m for sheet, m in minutes.items() if sheet not in known),
        },
        "sheets": rows,
    }


def kpi_report(documents: Sequence[dict]) -> dict:
    """The KPI per Document, per Convention Profile (aggregated across the
    Documents reviewed under it) and overall, with the basis stated. The
    documents are run_kpi() results; they must share one idle threshold,
    or the stated basis would not be the one the numbers were computed
    under."""
    thresholds = {document["idle_threshold_minutes"]
                  for document in documents}
    if len(thresholds) > 1:
        raise ValueError(
            f"documents were computed at different idle thresholds"
            f" {sorted(thresholds)}; one report states one basis")
    idle_minutes = thresholds.pop() if thresholds else DEFAULT_IDLE_MINUTES

    by_profile: dict[tuple[str, str], dict] = {}
    for document in documents:
        for row in document["sheets"]:
            profile = row["profile"]
            if profile is None:
                continue
            entry = by_profile.setdefault(
                _identity(profile),
                {"profile": profile, "documents": [], "rows": []})
            if document["document"] not in entry["documents"]:
                entry["documents"].append(document["document"])
            entry["rows"].append(row)
    profiles = [{"profile": entry["profile"],
                 "documents": entry["documents"],
                 **_metrics(entry["rows"])}
                for _, entry in sorted(by_profile.items())]
    all_rows = [row for document in documents for row in document["sheets"]]
    return {
        "basis": basis(idle_minutes),
        "overall": _metrics(all_rows),
        "documents": list(documents),
        "profiles": profiles,
    }


def display_kpi(value: float | None) -> str:
    """A KPI number for a person — or the plain statement that it is not
    measurable yet, never a zero standing in for 'no data'."""
    return "not yet measurable" if value is None else f"{value:.1f}"


def _metric_lines(metrics: dict, indent: str) -> list[str]:
    symbols = metrics["by_kind"]["symbol"]
    sheets = metrics["sheets_summary"]
    return [
        f"{indent}corrections per 100 symbols:"
        f" {display_kpi(metrics['corrections_per_100_symbols'])}"
        f" ({symbols['reviewed']} symbols reviewed,"
        f" {symbols['corrections']} corrections)",
        f"{indent}reviewer minutes per accepted Sheet:"
        f" {display_kpi(metrics['reviewer_minutes_per_accepted_sheet'])}"
        f" ({sheets['accepted']} accepted of {sheets['total']} Sheets,"
        f" {sheets['accepted_timed']} timed by the Workbench log)",
    ]


def format_summary(report: dict) -> str:
    lines = [f"Product KPI — {len(report['documents'])} Document(s),"
             f" {len(report['profiles'])} Convention Profile(s)"]
    lines += _metric_lines(report["overall"], "  ")
    lines.append("  basis:")
    for key in ("corrections_per_100_symbols",
                "reviewer_minutes_per_accepted_sheet"):
        lines.append(f"    {key.replace('_', ' ')}: {report['basis'][key]}")
    lines.append("per Document:")
    for document in report["documents"]:
        lines.append(f"  {document['document']} ({document['run_dir']})")
        lines += _metric_lines(document, "    ")
    lines.append("per Convention Profile:")
    for entry in report["profiles"]:
        profile = entry["profile"]
        lines.append(f"  {profile['name']}@{profile['version']} —"
                     f" {', '.join(entry['documents'])}")
        lines += _metric_lines(entry, "    ")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m pidgraph.kpi",
        description="The product KPI — corrections per 100 symbols and"
                    " reviewer minutes per accepted Sheet — from one or"
                    " more runs' persisted verdicts and Workbench"
                    " activity, per Document and per Convention Profile.")
    parser.add_argument("run_dirs", nargs="+", type=Path, metavar="run_dir",
                        help="run directory holding the artifacts (and,"
                             " by default, the verdicts under labels/)")
    parser.add_argument("--labels-dir", type=Path, default=None,
                        help="where the verdicts persist, when the"
                             " Workbench was given --labels-dir; applies"
                             " to exactly one run_dir")
    parser.add_argument("--idle-minutes", type=float,
                        default=DEFAULT_IDLE_MINUTES,
                        help="an interval between Workbench events longer"
                             " than this is a break, not review time"
                             f" (default {DEFAULT_IDLE_MINUTES:g})")
    parser.add_argument("--out", type=Path, default=None,
                        help="write the full JSON report here (outside"
                             " git, beside the run artifacts)")
    args = parser.parse_args(argv)
    if args.labels_dir is not None and len(args.run_dirs) != 1:
        parser.error("--labels-dir applies to exactly one run_dir")
    if args.idle_minutes < 0:
        parser.error("--idle-minutes must not be negative")

    report = kpi_report([run_kpi(run_dir, args.labels_dir, args.idle_minutes)
                         for run_dir in args.run_dirs])
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.out.with_name(args.out.name + ".tmp")
        tmp.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        os.replace(tmp, args.out)
    print(format_summary(report))
    if args.out is not None:
        print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
