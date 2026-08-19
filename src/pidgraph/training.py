"""Training-set export (ticket 12): reviewer verdicts leave the
LabelStore as a supervised training set per Convention Profile, so
per-company fine-tuning draws on exactly the deployed corpus. Pass
verdicts become positive examples, rejects negatives, and edits carry
the human-corrected label beside the original detection.

The export reads the verdict store alone — each labeled example already
snapshots its detection — and is deterministic over it: examples are
sorted by Sheet and detection id, so re-export yields the same bytes
regardless of the order reviewing happened in. Training sets carry
drawing-derived content, so they live outside git next to the labels
(ADR-0001).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping

from .labels import LabelStore, profile_key

# The ground-truth fields per detection kind. A pass confirms them as
# recorded; an edit overrides exactly the corrected ones; a reject has
# no ground truth — the detection itself is the negative.
_LABEL_FIELDS = {
    "symbol": ("symbol_class", "bbox"),
    "line": ("line_class", "polyline"),
    "text": ("text_class", "string", "bbox"),
}

_ROLES = {"pass": "positive", "reject": "negative", "edit": "corrected"}


def _label(sheet: int, stored: dict) -> dict | None:
    """The supervised label of one stored example — or a refusal naming
    the record: the store's write contract validates verdicts and
    corrections, but not kinds or detection contents."""
    kind = stored["kind"]
    detection = stored["detection"]
    fields = _LABEL_FIELDS.get(kind)
    if fields is None:
        raise ValueError(
            f"Sheet {sheet} example {detection.get('id')!r} has kind"
            f" {kind!r}; a training example is one of"
            f" {sorted(_LABEL_FIELDS)}")
    if stored["verdict"] == "reject":
        return None
    try:
        label = {field: detection[field] for field in fields}
    except KeyError as error:
        raise ValueError(
            f"Sheet {sheet} {kind} {detection.get('id')!r} lacks the"
            f" label field {error.args[0]!r}") from None
    if stored["verdict"] == "edit":
        label.update(stored["correction"])
    return label


def _training_example(profile: Mapping[str, str], source: str | None,
                      sheet: int, stored: dict) -> dict:
    return {"profile": dict(profile), "source": source, "sheet": sheet,
            "kind": stored["kind"], "verdict": stored["verdict"],
            "example": _ROLES[stored["verdict"]],
            "detection": stored["detection"],
            "label": _label(sheet, stored)}


def export_training_set(labels_root: Path | str,
                        profile: Mapping[str, str],
                        out_path: Path | str,
                        source: str | None = None) -> dict:
    """Export one Convention Profile's labeled examples as a JSONL
    training set at out_path — one example per line, sorted by Sheet
    then detection id, written whole (write-then-replace, like the
    store it reads). A profile the store holds no verdicts for is
    refused by name, so a typo'd identity or version surfaces instead
    of yielding an empty training set; a store file recorded for a
    different identity (a case-insensitive filesystem can resolve one
    profile's directory for another's name) is refused rather than
    re-stamped. The labels store is per-run: pass source to tell
    Documents apart when merging per-profile exports."""
    store = LabelStore(labels_root)
    examples = []
    counts = dict.fromkeys(_ROLES.values(), 0)
    for sheet in store.sheets(profile):
        labels = store.sheet_labels(profile, sheet)
        if labels["profile"] != dict(profile):
            raise ValueError(
                f"labels under {profile_key(profile)} for Sheet {sheet}"
                f" were recorded for profile {labels['profile']!r} —"
                f" refusing to re-stamp them")
        for _, stored in sorted(labels["examples"].items()):
            examples.append(_training_example(profile, source, sheet,
                                              stored))
            counts[_ROLES[stored["verdict"]]] += 1
    if not examples:
        raise ValueError(
            f"no labeled examples for profile {profile_key(profile)};"
            f" the store holds {store.profiles() or 'no profiles'}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_text(
        "".join(json.dumps(example, ensure_ascii=False) + "\n"
                for example in examples),
        encoding="utf-8")
    os.replace(tmp, out_path)
    return {"path": str(out_path), "profile": dict(profile),
            "source": source, "count": len(examples),
            "examples": counts}


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m pidgraph.training",
        description="Export one Convention Profile's reviewer verdicts"
                    " as a JSONL training set.")
    parser.add_argument("labels_root", type=Path,
                        help="the verdict store, e.g. <run_dir>/labels")
    parser.add_argument("--name", required=True,
                        help="Convention Profile identity")
    parser.add_argument("--version", required=True,
                        help="Convention Profile version")
    parser.add_argument("--source", default=None,
                        help="run/Document identifier stamped on every"
                             " example, for merging per-profile exports"
                             " across Documents")
    parser.add_argument("--out", type=Path, default=None,
                        help="output path (default: a training/"
                             " directory beside the labels, outside git"
                             " with the rest of the run artifacts)")
    args = parser.parse_args(argv)
    profile = {"name": args.name, "version": args.version}
    out = (args.out if args.out is not None
           else args.labels_root.parent / "training"
           / f"{profile_key(profile)}.jsonl")
    summary = export_training_set(args.labels_root, profile, out,
                                  args.source)
    print(f"{summary['count']} examples ({summary['examples']})"
          f" -> {summary['path']}")


if __name__ == "__main__":
    main()
