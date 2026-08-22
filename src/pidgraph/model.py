"""Domain model for the Raster Path (vocabulary: CONTEXT.md).

A Document is one drawing package; a Sheet is one page of it. A synthetic
Sheet may carry ground-truth annotations — the stub components read those
instead of pixels, so the whole pipeline is walkable offline. Detections
are what components emit at the seams; every one carries confidence and
provenance (which component produced it, from what evidence).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

Point = tuple[float, float]
Bbox = tuple[float, float, float, float]  # x0, y0, x1, y1


# --------------------------------------------------------------------------
# Input side: Document, Sheet, ground-truth annotations, Convention Profile

@dataclass(frozen=True)
class SymbolAnnotation:
    symbol_class: str            # key into the profile's Legend Dictionary
    bbox: Bbox
    direction: Point | None = None  # unit vector; flow arrows only


@dataclass(frozen=True)
class LineAnnotation:
    polyline: tuple[Point, ...]
    line_class: str = "pipe"


@dataclass(frozen=True)
class TextAnnotation:
    string: str
    text_class: str              # key into the profile's tag grammar
    bbox: Bbox


@dataclass(frozen=True)
class SheetAnnotations:
    """Exactly known ground truth carried by a synthetic Sheet."""
    symbols: tuple[SymbolAnnotation, ...] = ()
    lines: tuple[LineAnnotation, ...] = ()
    texts: tuple[TextAnnotation, ...] = ()


@dataclass(frozen=True)
class Sheet:
    number: int
    width: int
    height: int
    raster: bytes | None = None  # row-major grayscale, 0 = ink, 255 = paper
    annotations: SheetAnnotations | None = None


@dataclass(frozen=True)
class Document:
    name: str
    sheets: tuple[Sheet, ...]


@dataclass(frozen=True)
class LegendEntry:
    """Semantics of one symbol class in a Legend Dictionary."""
    role: str                          # "Equipment" | "PipingComponent" |
                                       # "ProcessInstrumentationFunction" |
                                       # "PipeOffPageConnector" | "Nozzle" |
                                       # "FlowArrow"
    equipment_type: str | None = None  # s2_pml node type ("tank", "valve", ...)
    component_class: str | None = None # DEXPI componentClass ("GateValve", ...)
    shape: str | None = None           # DEXPI Equipment shape ("capsule", ...)


@dataclass(frozen=True)
class ConventionProfile:
    """Per-company adaptation bundle: Legend Dictionary, tag grammar,
    line-type semantics. In-memory form of the versioned on-disk bundle
    loaded by profile.load_profile()."""
    name: str
    version: str
    legend: Mapping[str, LegendEntry]
    tag_grammar: Mapping[str, str]           # text_class -> fullmatch regex
    line_semantics: Mapping[str, str] = field(default_factory=dict)
                                             # line_class -> DEXPI segmentClass
    line_styles: Mapping[str, str] = field(default_factory=dict)
                                             # stroke style -> line_class

    def identity_record(self) -> dict[str, str]:
        """The stamp run artifacts carry: which profile version produced
        them (reproducibility, ticket 04)."""
        return {"name": self.name, "version": self.version}


# --------------------------------------------------------------------------
# Output side: detections at the component seams, run artifacts at the top

# The reserved symbol class a detector emits for ink it found but could
# not confidently name (ticket 17). It stays in the detection record for
# review, but assembly never turns it into a plant item, and a Legend
# Dictionary may not define it as a real class (load_profile refuses).
UNCLASSIFIED_SYMBOL = "unclassified"

# The reserved text class an OCR engine's adapter emits for a read that
# no tag-grammar class fits (ticket 18) — free text, a label outside the
# Convention Profile's grammar, or a read smudged past classification.
# It has no grammar by construction, so the lexicon decoder fails it
# closed: it stays in the detection record for review and never names a
# plant item, and a tag grammar may not define it (load_profile refuses).
UNCLASSIFIED_TEXT = "unclassified"


@dataclass(frozen=True)
class Provenance:
    component: str  # which component produced the element
    evidence: str   # from what evidence


@dataclass(frozen=True)
class SymbolDetection:
    id: str
    sheet: int
    symbol_class: str
    bbox: Bbox
    confidence: float
    provenance: Provenance
    direction: Point | None = None  # flow arrows only


@dataclass(frozen=True)
class LineDetection:
    id: str
    sheet: int
    polyline: tuple[Point, ...]
    line_class: str
    confidence: float
    provenance: Provenance


@dataclass(frozen=True)
class TextDetection:
    """A recognized string, after lexicon-constrained decoding against the
    Convention Profile's tag grammar. `string` is the corrected tag when a
    correction was applied; an unresolved detection keeps the raw candidate
    verbatim and is flagged, never guessed (fail-closed, ticket 06)."""
    id: str
    sheet: int
    string: str
    text_class: str
    bbox: Bbox
    confidence: float
    provenance: Provenance
    raw_string: str | None = None    # the candidate as read, when decoding
                                     # touched it (corrected or unresolved)
    correction: str | None = None    # swaps applied, e.g. "O->0 at index 3"
    resolved: bool = False           # set True only by the lexicon decoder;
                                     # a detection is born unverified, and an
                                     # unresolved one never names a plant item
    candidates: tuple[str, ...] = () # grammar-valid near-matches when the
                                     # correction was ambiguous


@dataclass(frozen=True)
class RunArtifacts:
    """What digitize() produces: per-Sheet detection records plus the DEXPI
    JSON plant model, the s2_pml equipment-level graph, and the store
    result (the offline default includes the Cypher script)."""
    detection_records: tuple[dict, ...]
    plant_model: dict
    plant_graph: dict
    store_summary: dict
    paths: dict[str, str]

    @property
    def cypher_script(self) -> str | None:
        return self.store_summary.get("script")
