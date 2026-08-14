# pidgraph

Standalone P&ID digitization software: scanned P&ID drawings are read by
computer vision and OCR into a plant topology graph, exported as DEXPI JSON
and loaded into Neo4j. Independent of hazop-ai, but deliberately speaks the
same output contracts so it can be reintegrated once proven.

## Language

**Document**:
One P&ID drawing package from a company, arriving as a single PDF
(typically 200–500 sheets).
_Avoid_: file, drawing set

**Sheet**:
A single page of a Document — one drawing.
_Avoid_: page (reserve "page" for PDF mechanics)

**Legend Sheet**:
A Sheet that defines the Document's symbols, line types, tag formats and
service codes rather than depicting the process.

**Raster Path**:
The extraction engine that turns scanned Sheets into a topology graph. In
pidgraph v1 it is the only path; the name preserves hazop-ai's router
vocabulary, where it sits beside the deterministic vector path.
_Avoid_: OCR pipeline, CV pipeline

**OCR**:
Strictly the text-recognition component inside the Raster Path — reading
tags, line numbers and labels from pixels. Never a name for the whole
product or engine.
_Avoid_: using "OCR" for the Raster Path or for pidgraph itself

**Convention Profile**:
The per-company (or per-convention-family) bundle that adapts the engine to
one drawing convention: Legend Dictionary, tag grammar, line-type semantics.
_Avoid_: config, template

**Legend Dictionary**:
The symbol glyphs and their meanings harvested from a Document's Legend
Sheets; the symbol part of a Convention Profile.

**Review Workbench**:
The web UI where a process engineer inspects the digitized P&ID overlaid on
the original Sheet and passes or rejects each detection. Every verdict is
kept as a labeled example.
_Avoid_: viewer (that is hazop-ai's read-only overlay)

**DEXPI JSON**:
The DEXPI-aligned plant-model JSON contract, identical in shape to
hazop-ai's `plant_model_dexpi.json` — pidgraph's canonical output.

**FLOWS_TO**:
Neo4j relationship for a connection whose flow direction is known, always
carrying its evidence source. Mirrors hazop-ai's `s2_pml` schema.

**CONNECTED_TO**:
Neo4j relationship for a connection whose flow direction is unknown. A
direction is never guessed into FLOWS_TO without evidence.
