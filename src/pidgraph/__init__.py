"""pidgraph: scanned P&ID Sheets -> plant topology graph -> DEXPI JSON +
Neo4j (s2_pml schema, ADR-0001). The one top seam is digitize()."""

from .intake import IntakeError, VectorPdfRefusal, load_document
from .model import (
    ConventionProfile,
    Document,
    LegendEntry,
    LineAnnotation,
    RunArtifacts,
    Sheet,
    SheetAnnotations,
    SymbolAnnotation,
    TextAnnotation,
)
from .pipeline import digitize
from .seams import PipelineConfig

__all__ = [
    "ConventionProfile",
    "Document",
    "IntakeError",
    "LegendEntry",
    "LineAnnotation",
    "PipelineConfig",
    "RunArtifacts",
    "Sheet",
    "SheetAnnotations",
    "SymbolAnnotation",
    "TextAnnotation",
    "VectorPdfRefusal",
    "digitize",
    "load_document",
]
