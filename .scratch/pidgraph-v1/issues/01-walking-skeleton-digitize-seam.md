# 01 — Walking skeleton: digitize() a synthetic Sheet to DEXPI JSON and Cypher

**What to build:** Running `digitize(Document, Convention Profile)` on a tiny programmatically drawn synthetic Sheet — exactly known symbols, lines, and tag strings — produces run artifacts an operator can inspect: per-Sheet detection records and a DEXPI JSON plant model, plus an offline Cypher script in the s2_pml schema. Everything flows through the three component seams on their offline deterministic defaults; no real model, engine, or database exists yet, but the full path is walkable and every later ticket only deepens a stage of it.

**Blocked by:** None — can start immediately.

**Status:** resolved

- [x] `digitize(Document, Convention Profile)` is the one top seam: it accepts a Document and a Convention Profile (a minimal in-memory profile is acceptable for this ticket) and returns run artifacts — per-Sheet detection records plus DEXPI JSON.
- [x] The emitted DEXPI JSON matches hazop-ai's plant-model contract in shape, pinned by a contract fixture mirrored from hazop-ai per ADR-0001 — a test fails if the shape drifts.
- [x] The three component seams exist, each with an offline deterministic default selected by configuration: SymbolDetector (stub), TextRecognizer (stub), GraphStore (Cypher-script emission).
- [x] The default GraphStore emits a Cypher script conforming to the s2_pml schema; connections without direction evidence are emitted as CONNECTED_TO — no FLOWS_TO appears without an evidence source.
- [x] Every extracted element in the detection records and DEXPI JSON carries confidence and provenance (which component produced it, from what evidence).
- [x] Pipeline-seam tests digitize the synthetic fixture Sheet and assert on the emitted DEXPI JSON and detection records — external behavior at the seams, never internals.
- [x] The full test suite runs with no GPU, no trained models, no network, no Neo4j, and no API keys.

## Comments

2026-08-15 — Implemented on main. `digitize(Document, ConventionProfile,
config=None, out_dir=None)` in `src/pidgraph/pipeline.py` returns
RunArtifacts (per-Sheet detection records, DEXPI JSON, s2_pml equipment
graph, store result) and optionally writes them to a run directory. Seams:
`SymbolDetector`/`TextRecognizer` stubs read synthetic-Sheet annotations;
`GraphStore` default emits an s2_pml Cypher script and raises if a "known"
direction lacks direction_sources. DEXPI shape pinned by
`tests/fixtures/dexpi_contract_shape.json` (structure/types/vocab only —
no drawing content in git, per ADR-0001). 14 tests, stdlib-only runtime,
suite runs offline. Code-reviewed (standards + spec axes); fixed OPC
vocabulary drift (OPC-<label> / equipment_type "line", matching hazop-ai's
adapter) and added a two-Sheet test so crossSheetLinks shape is pinned
non-vacuously. Deferred to later tickets as designed: real lexicon
correction (06), raster-based line extraction (05), raster normalization
(03).

