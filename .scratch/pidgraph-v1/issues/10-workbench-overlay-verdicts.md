# 10 — Review Workbench: Sheet overlay and pass/reject/edit verdicts

**What to build:** A reviewer opens the Review Workbench — a local web application — and sees the digitized P&ID overlaid on the original Sheet, so checking the extraction is visual and fast. Each detection takes one action: pass, reject, or edit (supplying the corrected tag text or geometry). Every verdict is automatically stored as a labeled example keyed to Convention Profile and Sheet — reviewing is simultaneously training-data creation. The Workbench reads run artifacts only; it never re-runs extraction.

**Blocked by:** 01 (Walking skeleton), 04 (Convention Profile as a versioned artifact).

**Status:** ready-for-agent

- [ ] The Workbench serves locally and renders a Sheet with its detections overlaid on the original raster — symbols, line runs, and text each visually distinguishable and positioned by their recorded geometry.
- [ ] Each detection can be given a verdict: pass, reject, or edit; edit accepts a corrected tag text or corrected geometry.
- [ ] Every verdict persists as a labeled example keyed to the Convention Profile (identity + version) and the Sheet identifier, capturing the original detection and, for edits, the correction.
- [ ] Verdicts survive a restart of the Workbench; re-opening a Sheet shows its existing verdicts.
- [ ] The Workbench's data comes only from run artifacts — it has no path to invoke extraction.
- [ ] Tests use the Flask test client against prepared run artifacts, asserting overlays render and verdicts persist as labeled examples (prior art: hazop-ai's s1_dim app tests); offline test invariant holds.
- [ ] No drawing content enters git: fixtures are synthetic, and labeled examples referencing real Sheets live outside the repository per ADR-0001.

## Comments
