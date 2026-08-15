# 03 — Raster normalization: deskew, resolution-normalize, binarize

**What to build:** Scanned Sheets are normalized before extraction — deskewed, resolution-normalized, and binarized — so detector and OCR accuracy doesn't depend on scanner quirks. An operator digitizing a skewed, oddly-scaled scan gets the same downstream behavior as from a clean one, and can see in provenance exactly what normalization was applied.

**Blocked by:** 01 (Walking skeleton).

**Status:** ready-for-agent

- [ ] Normalization runs as the first Raster Path stage on every Sheet: deskew, resolution-normalize to a target scale, binarize.
- [ ] A programmatically skewed fixture Sheet (known rotation) is recovered to within a stated tolerance, verified deterministically.
- [ ] Fixtures at off-target resolutions are normalized to the target scale, and downstream stages consume the normalized raster.
- [ ] The normalization applied to each Sheet (angle corrected, scale factor, binarization method) is recorded in that Sheet's provenance in the run artifacts.
- [ ] Geometry in detection records maps back to the original Sheet coordinates so overlays remain correct after normalization.
- [ ] Deterministic classical CV throughout — no seam, no ML, offline test invariant holds.

## Comments
