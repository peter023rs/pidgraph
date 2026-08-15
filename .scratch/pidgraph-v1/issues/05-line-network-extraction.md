# 05 — Line network extraction with junctions and symbol-boundary splits

**What to build:** The pipe line network of a Sheet is extracted so connectivity reflects the real process: binarize → thin → vectorize → junction analysis → split runs at symbol boxes. A line passing through a valve becomes two connections meeting at that valve, never a polyline traced through it; junctions become real branch points. The result lands in the DEXPI JSON and Cypher output as connections.

**Blocked by:** 01 (Walking skeleton).

**Status:** ready-for-agent

- [ ] The line network stage runs the classical-CV sequence binarize → thin → vectorize → junction analysis → split runs at symbol boxes; deterministic, no seam, no ML.
- [ ] A synthetic fixture with a T-junction yields three runs meeting at one junction node.
- [ ] A synthetic fixture with a line through a symbol box yields two runs terminating at that symbol — connectivity in DEXPI JSON shows the symbol between them.
- [ ] Extracted connections appear in the emitted DEXPI JSON and s2_pml Cypher as CONNECTED_TO (direction is a later ticket's concern) with confidence and provenance per element.
- [ ] Line-type semantics from the Convention Profile classify runs where the profile defines them (e.g. process vs instrument line styles).
- [ ] Pipeline-seam tests assert connectivity on synthetic fixtures with exactly known topology; offline test invariant holds.

## Comments
