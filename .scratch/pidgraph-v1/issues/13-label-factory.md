# 13 — Label factory: render hazop-ai vector artifacts with projected ground truth

**What to build:** A dev-time label factory renders hazop-ai's 2401 vector artifacts to raster Sheets and projects the deterministic detections, text spans, and connectivity down as pixel-level ground-truth labels — so detector and OCR training/eval data exists before the corpus arrives. This is a dependency on hazop-ai's artifacts, not its code, and dissolves as Workbench corrections accumulate.

**Blocked by:** 01 (Walking skeleton).

**Status:** resolved

- [x] Given a hazop-ai vector artifact, the factory renders a raster Sheet and emits labels: symbol boxes with classes, text spans with ground-truth strings, and connectivity — in the same label schema the eval harness and training consume.
- [x] Projection is geometrically faithful: a projected box surrounds its rendered symbol at the chosen render resolution (verified on a synthetic vector fixture with known geometry).
- [x] The factory batch-processes an artifact directory and reports per-artifact success/failure without stopping the batch.
- [x] Rendered Sheets and labels are written outside git; ignore rules keep all derived drawing content out of the repository per ADR-0001.
- [x] The factory is dev-time tooling — it is not part of the shipped Raster Path and never runs in CI; its tests use small synthetic vector fixtures, keeping the offline test invariant.

## Comments

2026-08-19 — Implemented as `src/pidgraph/label_factory.py`:
`run_label_factory(artifact_dir, pdf_path, out_dir, ...)` plus
`python -m pidgraph.label_factory <l1_output> <pdf> [--out --dpi
--name --version]` (default out: `data/labelfactory`, gitignored; a
test asserts the default root stays ignored via `git check-ignore`).

The artifacts (`topology_page<N>.json`) carry detections and
connectivity in PDF points (top-left origin) but no drawing primitives
and no text geometry, so the Sheet raster comes from rasterizing the
source PDF (PyMuPDF, new `labelfactory` extra, imported lazily like the
neo4j driver; also added to `dev` so the render tests run) and text
ground truth comes from the PDF's own text layer — lossless strings
with exact boxes, strictly better than anything hazop-ai persisted.
Projection is the uniform scale dpi/72 (verified against hazop-ai's own
overlay renderer); default 150 dpi is hazop-ai's overlay resolution.

Labels leave the factory in the shapes existing consumers already read:
symbol/text/line ground truth as pass-verdict examples in the
LabelStore schema (ticket 10), exported per profile through
`export_training_set` (ticket 12) — `LabelStore.record_many` was added
for a single atomic write per Sheet. Connectivity is emitted per Sheet
(`connectivity/sheet_<N>.json`) as terminal-level links in the s2_pml
edge shape `build_plant_graph` emits (ADR-0001), with the same edge
multiplicity, so the ticket-15 connectivity gate compares like for
like: nozzles fold into their parent item, pass-through geometry
(junction/corner/dead-end) is contracted, and direction is asserted
only where terminal-incident edges carry consistent evidence — never
guessed.

Decisions verified empirically against the real 2401 artifacts before
coding: a nozzle's `equipment` field indexes the sheet's bbox-carrying
nodes in order of appearance — equipment plus equipment-like off-page
items — confirmed 188/188 against hazop-ai's DEXPI export (a pure
"equipment nodes" reading breaks on sheet 7); equipment keeps its
detected bbox while valve/instrument/nozzle/OPC boxes are synthesized
from convention constants (ISA bubble radius 16.4 pt pinned by
hazop-ai; valve extents measured off the rendered sheets), all
configurable via `BoxSynthesis`; instrument tags render as separate
token spans inside the bubble, so text classing matches exact strings
first and tag tokens within the bubble box second, with unmatched spans
kept as `free_text` ground truth (string and box still exact). The
stale legend-sheet artifacts (pages 1–3, missing `direction_stats`) are
refused per artifact with a named reason and the batch continues.

Real run: 9/12 artifacts succeed (the 3 stale ones refused), 5,420
examples — 644 symbols, 1,682 text spans, 3,094 lines — plus 732
connectivity links; two runs are byte-identical apart from the paths
the manifest embeds. Tests (28 new): pure projection core and batch
behavior offline with an injected fake reader; render fidelity against
a programmatic vector-PDF fixture (`pdf_fixtures` gained a
`vector_ops` page) proving projected equipment boxes, synthesized valve
boxes, and text-span boxes all surround their rendered ink at the
chosen resolution (`importorskip` where PyMuPDF is absent).
Code-reviewed (standards + spec axes); the connectivity shape was
aligned to s2_pml edges and the synthesized-box fidelity test added in
response. Known conservatisms, dissolving as Workbench corrections
accumulate: direction ground truth is sparse (16 links; mid-chain
arrow evidence is not path-followed), and equipment-tag spans that
don't exact-match stay `free_text`. Synthetic degradation is ticket 14;
the eval harness consuming these sets is ticket 15.
