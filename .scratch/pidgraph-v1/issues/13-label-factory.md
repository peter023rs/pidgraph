# 13 — Label factory: render hazop-ai vector artifacts with projected ground truth

**What to build:** A dev-time label factory renders hazop-ai's 2401 vector artifacts to raster Sheets and projects the deterministic detections, text spans, and connectivity down as pixel-level ground-truth labels — so detector and OCR training/eval data exists before the corpus arrives. This is a dependency on hazop-ai's artifacts, not its code, and dissolves as Workbench corrections accumulate.

**Blocked by:** 01 (Walking skeleton).

**Status:** ready-for-agent

- [ ] Given a hazop-ai vector artifact, the factory renders a raster Sheet and emits labels: symbol boxes with classes, text spans with ground-truth strings, and connectivity — in the same label schema the eval harness and training consume.
- [ ] Projection is geometrically faithful: a projected box surrounds its rendered symbol at the chosen render resolution (verified on a synthetic vector fixture with known geometry).
- [ ] The factory batch-processes an artifact directory and reports per-artifact success/failure without stopping the batch.
- [ ] Rendered Sheets and labels are written outside git; ignore rules keep all derived drawing content out of the repository per ADR-0001.
- [ ] The factory is dev-time tooling — it is not part of the shipped Raster Path and never runs in CI; its tests use small synthetic vector fixtures, keeping the offline test invariant.

## Comments
