# 02 — PDF Document intake, Sheet enumeration, and vector-PDF refusal

**What to build:** An operator points pidgraph at a real PDF Document instead of a fixture image. A scanned PDF is accepted: its Sheets are enumerated and rasterized into the Raster Path, and `digitize()` runs end-to-end on them. A vector PDF is detected at intake and refused with a clear message — v1 never silently produces raster-quality output for a Document that deserves the deterministic path.

**Blocked by:** 01 (Walking skeleton).

**Status:** ready-for-agent

- [ ] A scanned (raster) PDF Document is accepted at intake; its Sheets are enumerated and rasterized, and each flows through `digitize()` to detection records and DEXPI JSON.
- [ ] A vector PDF Document is detected at intake and refused with a message that names the Document, states why it was refused, and says what to do instead — the run produces no partial output.
- [ ] Sheet identity is preserved through the run: detection records and DEXPI JSON reference Sheets by identifier (glossary: "Sheet", never "page" outside PDF mechanics).
- [ ] Tests use small programmatically generated PDF fixtures — one raster-style, one vector — and assert acceptance and refusal at the intake seam; no real drawings enter git.
- [ ] The offline test invariant holds: no GPU, models, network, or Neo4j.

## Comments
