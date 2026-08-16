# 03 — Raster normalization: deskew, resolution-normalize, binarize

**What to build:** Scanned Sheets are normalized before extraction — deskewed, resolution-normalized, and binarized — so detector and OCR accuracy doesn't depend on scanner quirks. An operator digitizing a skewed, oddly-scaled scan gets the same downstream behavior as from a clean one, and can see in provenance exactly what normalization was applied.

**Blocked by:** 01 (Walking skeleton).

**Status:** done

- [x] Normalization runs as the first Raster Path stage on every Sheet: deskew, resolution-normalize to a target scale, binarize.
- [x] A programmatically skewed fixture Sheet (known rotation) is recovered to within a stated tolerance, verified deterministically.
- [x] Fixtures at off-target resolutions are normalized to the target scale, and downstream stages consume the normalized raster.
- [x] The normalization applied to each Sheet (angle corrected, scale factor, binarization method) is recorded in that Sheet's provenance in the run artifacts.
- [x] Geometry in detection records maps back to the original Sheet coordinates so overlays remain correct after normalization.
- [x] Deterministic classical CV throughout — no seam, no ML, offline test invariant holds.

## Comments

Implemented in `src/pidgraph/normalize.py` + pipeline wiring; tests in
`tests/test_normalization.py` (35 total in suite, all passing; mypy clean).

- Deskew: projection-profile search (coarse 0.5° over ±15°, fine 0.05°),
  stated tolerance `SKEW_TOLERANCE_DEGREES = 0.1`, held by tests at +3.2°,
  −2.4°, and 0.75° (a coarse-grid-boundary angle). A pixel-level test
  confirms the corrected raster lands back on the clean render (≥98% of
  ink within 1 px), not just that the angle number matches.
- Resolution: long side normalized to `TARGET_LONG_SIDE = 400` (the v1
  synthetic-corpus scale — revisit when the real corpus arrives, ticket
  16). Downscaling min-pools so 1-px strokes survive; upscaling is
  nearest-neighbor. Both directions tested.
- Binarize: Otsu, applied last; method and threshold recorded.
- Two frames from here on: extraction stages consume the normalized Sheet
  (raster and ground-truth annotations both transformed); detection
  geometry is mapped back through `Normalization.to_original()` before
  records and assembly, so detection records, DEXPI geometry, and
  overlays all stay in original Sheet coordinates. Skewed-Document seam
  test verifies polylines map back exactly and bboxes stay centered on
  (and covering) the original symbols — axis-aligned hulls grow slightly
  under a skew round-trip, by design.
- Per-Sheet provenance: `"normalization"` key in each detection record
  (angle_degrees, scale, binarization, threshold, original/normalized
  size), on disk in `detections/sheet_N.json`.
- Code review (standards + spec axes) found no hard violations; the spec
  axis's test gaps (raster-level deskew verification, upscale branch,
  rasterless pass-through, combined skew+scale round-trip) were closed
  with additional tests.
