# 05 — Line network extraction with junctions and symbol-boundary splits

**What to build:** The pipe line network of a Sheet is extracted so connectivity reflects the real process: binarize → thin → vectorize → junction analysis → split runs at symbol boxes. A line passing through a valve becomes two connections meeting at that valve, never a polyline traced through it; junctions become real branch points. The result lands in the DEXPI JSON and Cypher output as connections.

**Blocked by:** 01 (Walking skeleton).

**Status:** done

- [x] The line network stage runs the classical-CV sequence binarize → thin → vectorize → junction analysis → split runs at symbol boxes; deterministic, no seam, no ML.
- [x] A synthetic fixture with a T-junction yields three runs meeting at one junction node.
- [x] A synthetic fixture with a line through a symbol box yields two runs terminating at that symbol — connectivity in DEXPI JSON shows the symbol between them.
- [x] Extracted connections appear in the emitted DEXPI JSON and s2_pml Cypher as CONNECTED_TO (direction is a later ticket's concern) with confidence and provenance per element.
- [x] Line-type semantics from the Convention Profile classify runs where the profile defines them (e.g. process vs instrument line styles).
- [x] Pipeline-seam tests assert connectivity on synthetic fixtures with exactly known topology; offline test invariant holds.

## Comments

Implemented in `src/pidgraph/lines.py` (pure-Python classical CV on the
normalized raster; extractor signature grew to
`extract_line_network(sheet, symbols, profile)` — symbol boxes split runs,
flow-arrow boxes are bridged straight across). Junctions are first-class in
assembly (`Junction`, one shared `junction` PipingNode in DEXPI, terminal
pairs meeting at a junction get undirected CONNECTED_TO edges with
confidence + provenance). Stroke style (solid/dashed) is classified through
the new optional `line_styles.json` Convention Profile part
(style → line_class); line_semantics then maps line_class → segmentClass as
before. Seam tests: `tests/test_line_network.py`. The conftest fixture's
run3 was documented as an L but drawn collinear — it is now a real L (OPC
moved to (365,160,395,180)), so the corner survives real vectorization.
