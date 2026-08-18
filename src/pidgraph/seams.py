"""The three component seams, each with an offline deterministic default
selected by configuration (hazop-ai's proven seam pattern):

  SymbolDetector  — stub today, trained detector later (ticket 16)
  TextRecognizer  — stub today, the chosen OCR engine later (ticket 18)
  GraphStore      — Cypher-script emission today, live Neo4j later (ticket 09)

The stubs read the ground-truth annotations a synthetic Sheet carries; real
implementations read the raster. Both live behind the same interface, so the
pipeline never knows which it got.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .cypher_store import CypherScriptGraphStore
from .lexicon import CONFUSION_PAIRS
from .model import (
    ConventionProfile,
    Provenance,
    Sheet,
    SymbolDetection,
    TextDetection,
)


class SymbolDetector(Protocol):
    def detect(self, sheet: Sheet,
               profile: ConventionProfile) -> list[SymbolDetection]: ...


class TextRecognizer(Protocol):
    def recognize(self, sheet: Sheet,
                  profile: ConventionProfile) -> list[TextDetection]: ...


class GraphStore(Protocol):
    def store(self, graph: dict, out_dir: Path | None) -> dict: ...


def _require_annotations(sheet: Sheet, component: str):
    if sheet.annotations is None:
        raise ValueError(
            f"{component} is an offline stub and needs an annotated "
            f"synthetic Sheet; Sheet {sheet.number} carries no annotations")
    return sheet.annotations


class StubSymbolDetector:
    """Deterministic offline default: reports the symbols a synthetic Sheet
    is annotated with, at full confidence."""

    COMPONENT = "symbol_detector:stub"

    def detect(self, sheet: Sheet,
               profile: ConventionProfile) -> list[SymbolDetection]:
        annotations = _require_annotations(sheet, self.COMPONENT)
        return [
            SymbolDetection(
                id=f"p{sheet.number}-sym{i}",
                sheet=sheet.number,
                symbol_class=sym.symbol_class,
                bbox=sym.bbox,
                confidence=1.0,
                provenance=Provenance(
                    component=self.COMPONENT,
                    evidence=f"synthetic annotation {sym.symbol_class}"
                             f" at bbox {sym.bbox}"),
                direction=sym.direction,
            )
            for i, sym in enumerate(annotations.symbols)
        ]


# The classic letter/digit OCR flips, seeded deterministically into every
# stub read so the lexicon decoder's corrections are proven end to end.
# Derived from the decoder's own confusion set — one source of truth, so
# the stub can never emit noise the decoder no longer repairs.
_OCR_NOISE = str.maketrans({digit: letter
                            for letter, digit in CONFUSION_PAIRS})


class StubTextRecognizer:
    """Deterministic offline default: reads the tag strings a synthetic
    Sheet is annotated with, degraded by seeded OCR noise (0 read as O,
    5 read as S) the way a real engine smudges glyphs."""

    COMPONENT = "text_recognizer:stub"

    def recognize(self, sheet: Sheet,
                  profile: ConventionProfile) -> list[TextDetection]:
        annotations = _require_annotations(sheet, self.COMPONENT)
        detections = []
        for i, text in enumerate(annotations.texts):
            noisy = text.string.translate(_OCR_NOISE)
            evidence = (f"synthetic annotation {text.text_class}"
                        f" at bbox {text.bbox}")
            if noisy != text.string:
                evidence += f", read with seeded OCR noise as {noisy!r}"
            detections.append(TextDetection(
                id=f"p{sheet.number}-text{i}",
                sheet=sheet.number,
                string=noisy,
                text_class=text.text_class,
                bbox=text.bbox,
                confidence=1.0,
                provenance=Provenance(component=self.COMPONENT,
                                      evidence=evidence),
            ))
        return detections


SYMBOL_DETECTORS = {"stub": StubSymbolDetector}
TEXT_RECOGNIZERS = {"stub": StubTextRecognizer}
GRAPH_STORES = {"cypher-script": CypherScriptGraphStore}


@dataclass(frozen=True)
class PipelineConfig:
    """Selects the implementation behind each component seam. The defaults
    are the offline deterministic ones — no GPU, no models, no network."""
    symbol_detector: str = "stub"
    text_recognizer: str = "stub"
    graph_store: str = "cypher-script"


def _resolve(registry: dict, name: str, seam: str):
    try:
        return registry[name]()
    except KeyError:
        raise KeyError(
            f"unknown {seam} {name!r}; available: {sorted(registry)}"
        ) from None


def build_components(
        config: PipelineConfig
) -> tuple[SymbolDetector, TextRecognizer, GraphStore]:
    return (
        _resolve(SYMBOL_DETECTORS, config.symbol_detector, "SymbolDetector"),
        _resolve(TEXT_RECOGNIZERS, config.text_recognizer, "TextRecognizer"),
        _resolve(GRAPH_STORES, config.graph_store, "GraphStore"),
    )
