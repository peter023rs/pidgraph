"""Convention Profile as a versioned on-disk artifact.

A profile bundle is a directory hand-built per company (no automated
Legend Sheet ingest in v1), holding an identity file and the three parts:

    profile.json         {"name": ..., "version": ...}
    legend.json          symbol_class -> LegendEntry fields
    tag_grammar.json     text_class -> fullmatch regex
    line_semantics.json  line_class -> DEXPI segmentClass

Loading validates the whole bundle up front and reports every problem at
once — a bad profile fails here, never mid-run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .dexpi import SEGMENT_CLASSES
from .model import ConventionProfile, LegendEntry

ROLES = frozenset({
    "Equipment", "PipingComponent", "ProcessInstrumentationFunction",
    "PipeOffPageConnector", "Nozzle", "FlowArrow",
})

_IDENTITY_KEYS = ("name", "version")
_LEGEND_OPTIONAL = ("equipment_type", "component_class", "shape")


class ProfileError(ValueError):
    """A Convention Profile bundle failed validation at load."""

    def __init__(self, bundle: Path, problems: list[str]):
        self.problems = tuple(problems)
        listing = "\n".join(f"  - {p}" for p in problems)
        super().__init__(
            f"invalid Convention Profile bundle {bundle}:\n{listing}")


def _read_part(bundle: Path, part: str, problems: list[str]) -> dict | None:
    path = bundle / part
    if not path.is_file():
        problems.append(f"{part}: missing")
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        problems.append(f"{part}: {error}")
        return None
    if not isinstance(data, dict):
        problems.append(f"{part}: must be a JSON object, "
                        f"got {type(data).__name__}")
        return None
    return data


def _identity(raw: dict, problems: list[str]) -> tuple[str, str]:
    for key in sorted(set(raw) - set(_IDENTITY_KEYS)):
        problems.append(f"profile.json: unknown key {key!r}")
    values = []
    for key in _IDENTITY_KEYS:
        value = raw.get(key)
        if not isinstance(value, str) or not value:
            problems.append(f"profile.json: {key!r} must be a "
                            f"non-empty string, got {value!r}")
            value = ""
        values.append(value)
    return values[0], values[1]


def _legend_entry(symbol_class: str, raw, problems: list[str]):
    where = f"legend.json: {symbol_class!r}"
    if not isinstance(raw, dict):
        problems.append(f"{where}: entry must be an object")
        return None
    found_before = len(problems)
    for key in sorted(set(raw) - {"role", *_LEGEND_OPTIONAL}):
        problems.append(f"{where}: unknown key {key!r}")
    role = raw.get("role")
    if role not in ROLES:
        problems.append(f"{where}: role {role!r} is not one of "
                        f"{sorted(ROLES)}")
    fields = {}
    for key in _LEGEND_OPTIONAL:
        value = raw.get(key)
        if value is not None and not isinstance(value, str):
            problems.append(f"{where}: {key} must be a string or null, "
                            f"got {value!r}")
            value = None
        fields[key] = value
    if role is None or len(problems) > found_before:
        return None
    return LegendEntry(role=role, **fields)


def _legend(raw: dict, problems: list[str]) -> dict[str, LegendEntry]:
    if not raw:
        problems.append("legend.json: empty — a Legend Dictionary needs at "
                        "least one symbol class")
    legend = {}
    for symbol_class, entry_raw in raw.items():
        entry = _legend_entry(symbol_class, entry_raw, problems)
        if entry is not None:
            legend[symbol_class] = entry
    return legend


def _tag_grammar(raw: dict, problems: list[str]) -> dict[str, str]:
    if not raw:
        problems.append("tag_grammar.json: empty — a tag grammar needs at "
                        "least one text class")
    grammar = {}
    for text_class, pattern in raw.items():
        if not isinstance(pattern, str):
            problems.append(f"tag_grammar.json: {text_class!r} must map to "
                            f"a regex string, got {pattern!r}")
            continue
        try:
            re.compile(pattern)
        except re.error as error:
            problems.append(f"tag_grammar.json: {text_class!r} regex "
                            f"{pattern!r} does not compile ({error})")
            continue
        grammar[text_class] = pattern
    return grammar


def _line_semantics(raw: dict, problems: list[str]) -> dict[str, str]:
    # empty is complete here: emission falls back to the default
    # segmentClass for unmapped line classes
    semantics = {}
    for line_class, segment_class in raw.items():
        if segment_class not in SEGMENT_CLASSES:
            problems.append(
                f"line_semantics.json: {line_class!r} maps to "
                f"{segment_class!r}, not a DEXPI segmentClass "
                f"{sorted(SEGMENT_CLASSES)}")
            continue
        semantics[line_class] = segment_class
    return semantics


def load_profile(bundle: Path | str) -> ConventionProfile:
    bundle = Path(bundle)
    problems: list[str] = []

    parts = {part: _read_part(bundle, f"{part}.json", problems)
             for part in ("profile", "legend", "tag_grammar",
                          "line_semantics")}

    name = version = ""
    legend: dict[str, LegendEntry] = {}
    grammar: dict[str, str] = {}
    semantics: dict[str, str] = {}
    if parts["profile"] is not None:
        name, version = _identity(parts["profile"], problems)
    if parts["legend"] is not None:
        legend = _legend(parts["legend"], problems)
    if parts["tag_grammar"] is not None:
        grammar = _tag_grammar(parts["tag_grammar"], problems)
    if parts["line_semantics"] is not None:
        semantics = _line_semantics(parts["line_semantics"], problems)

    if problems:
        raise ProfileError(bundle, problems)
    return ConventionProfile(name=name, version=version, legend=legend,
                             tag_grammar=grammar, line_semantics=semantics)
