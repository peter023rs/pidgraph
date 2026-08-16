# 02 — PDF Document intake, Sheet enumeration, and vector-PDF refusal

**What to build:** An operator points pidgraph at a real PDF Document instead of a fixture image. A scanned PDF is accepted: its Sheets are enumerated and rasterized into the Raster Path, and `digitize()` runs end-to-end on them. A vector PDF is detected at intake and refused with a clear message — v1 never silently produces raster-quality output for a Document that deserves the deterministic path.

**Blocked by:** 01 (Walking skeleton).

**Status:** ready-for-agent

- [x] A scanned (raster) PDF Document is accepted at intake; its Sheets are enumerated and rasterized, and each flows through `digitize()` to detection records and DEXPI JSON.
- [x] A vector PDF Document is detected at intake and refused with a message that names the Document, states why it was refused, and says what to do instead — the run produces no partial output.
- [x] Sheet identity is preserved through the run: detection records and DEXPI JSON reference Sheets by identifier (glossary: "Sheet", never "page" outside PDF mechanics).
- [x] Tests use small programmatically generated PDF fixtures — one raster-style, one vector — and assert acceptance and refusal at the intake seam; no real drawings enter git.
- [x] The offline test invariant holds: no GPU, models, network, or Neo4j.

## Comments

2026-08-16 (agent): Implemented as `pidgraph.intake.load_document()` on
pypdf (first runtime dependency; pure-Python, fully local). Every page is
classified from its content stream before any raster is extracted, so a
refusal produces no partial output; a mixed Document is refused naming the
offending Sheets. Two-axis review findings worth keeping:

- Known limitation, deliberate: raster extraction reads 8-bit DeviceGray
  images that are uncompressed or FlateDecode. Real scanner output
  (DCTDecode/JPEG, CCITTFax, 1-bit) is refused with a clear IntakeError
  naming the encoding — widening decode support belongs with ticket 03
  (raster normalization), where an image library decision is due anyway.
- `digitize()` end-to-end on intake Sheets is proven by attaching ground
  truth annotations in the test (the ticket-01 stubs read annotations, not
  pixels); the unmodified `load_document → digitize` product path needs the
  real detector/OCR components (tickets 16/18).
- A review-claimed bug — `/Resources` inherited from the `/Pages` node
  being missed — turned out not to exist (pypdf flattens inheritable page
  attributes); a regression test now pins that.
- BI..EI inline images are not extractable in v1; such a Sheet fails the
  one-scanned-image check rather than being mislabeled a vector PDF.
