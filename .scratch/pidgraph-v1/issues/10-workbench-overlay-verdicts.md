# 10 — Review Workbench: Sheet overlay and pass/reject/edit verdicts

**What to build:** A reviewer opens the Review Workbench — a local web application — and sees the digitized P&ID overlaid on the original Sheet, so checking the extraction is visual and fast. Each detection takes one action: pass, reject, or edit (supplying the corrected tag text or geometry). Every verdict is automatically stored as a labeled example keyed to Convention Profile and Sheet — reviewing is simultaneously training-data creation. The Workbench reads run artifacts only; it never re-runs extraction.

**Blocked by:** 01 (Walking skeleton), 04 (Convention Profile as a versioned artifact).

**Status:** resolved

- [x] The Workbench serves locally and renders a Sheet with its detections overlaid on the original raster — symbols, line runs, and text each visually distinguishable and positioned by their recorded geometry.
- [x] Each detection can be given a verdict: pass, reject, or edit; edit accepts a corrected tag text or corrected geometry.
- [x] Every verdict persists as a labeled example keyed to the Convention Profile (identity + version) and the Sheet identifier, capturing the original detection and, for edits, the correction.
- [x] Verdicts survive a restart of the Workbench; re-opening a Sheet shows its existing verdicts.
- [x] The Workbench's data comes only from run artifacts — it has no path to invoke extraction.
- [x] Tests use the Flask test client against prepared run artifacts, asserting overlays render and verdicts persist as labeled examples (prior art: hazop-ai's s1_dim app tests); offline test invariant holds.
- [x] No drawing content enters git: fixtures are synthetic, and labeled examples referencing real Sheets live outside the repository per ADR-0001.

## Comments

Implemented as `src/pidgraph/workbench.py` (Flask app factory, `workbench`
optional extra) over `src/pidgraph/labels.py` (the LabelStore). Serve with
`python -m pidgraph.workbench <run_dir>` — local-only, 127.0.0.1.

One gap surfaced during implementation: run artifacts never persisted the
Sheet raster, but the Workbench must overlay detections on it while reading
run artifacts only. `write_run_outputs` now also writes each Sheet's
original raster as `sheets/sheet_<N>.png` (stdlib-zlib PNG encoder,
`src/pidgraph/pngio.py` — no imaging dependency), for single-shot and batch
runs alike. Detections are already recorded in original Sheet coordinates,
so the overlay is a direct SVG projection of the record.

Verdicts persist under `<run_dir>/labels/<profile-name>@<version>/
sheet_<N>.json` (override with `create_app(..., labels_dir=...)`) — one
file per (Convention Profile, Sheet), latest verdict per detection,
write-then-replace. Each labeled example snapshots the original detection
beside the verdict, and for edits the correction; `labels.make_example`
validates the correction against the detection kind (tag text only for
texts; bbox for symbols/texts, polyline for line runs), so ticket 12's
export inherits the invariants. The Workbench module holds no reference
into the extraction engine (asserted structurally in
`tests/test_workbench.py`), and its only writable route is the verdict
endpoint.
