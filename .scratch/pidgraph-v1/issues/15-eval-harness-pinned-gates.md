# 15 — Eval harness with pinned component gates

**What to build:** An ML developer runs the eval harness on demand against labeled eval sets and gets gate scores: symbol F1, tag exact-match, connectivity F1 — so any model or prompt change is gated by numbers, not impressions. The harness scores whatever SymbolDetector/TextRecognizer implementations configuration selects, and can put two implementations side by side — the mechanism by which per-company fine-tuning is compared against Legend Dictionary nearest-neighbor classification, and by which the OCR engine is chosen empirically.

**Blocked by:** 01 (Walking skeleton), 13 (Label factory).

**Status:** resolved

- [x] The harness runs a configured pipeline against a labeled eval set and reports symbol F1, tag exact-match, and connectivity F1.
- [x] Gates are pinned and versioned with the harness: symbol F1 ≥ 0.90, tag exact-match ≥ 0.98, connectivity F1 ≥ 0.70 (provisional, refined when the corpus lands); output states pass/fail per gate.
- [x] Any seam implementation configuration can select is scorable — stubs included, so the harness itself is verifiable end-to-end before real models exist.
- [x] Two configurations can be scored side by side in one comparison report (fine-tuning vs nearest-neighbor; candidate OCR engines).
- [x] The harness is a separate concern from the test suite and never runs in CI; harness runs are fully local.
- [x] Metric computation (matching predicted to ground-truth boxes/strings/connections, F1 math) is unit-tested deterministically under the offline invariant.

## Comments

Implemented as `src/pidgraph/eval_harness.py` (gates, metric math, eval-set
loader, evaluate/compare, `python -m pidgraph.eval_harness` CLI) plus
`pngio.decode_gray_png` so the harness reads the Sheet rasters the label
factory persisted. The eval set is a label-factory output directory
(`labels/`, `connectivity/`, `sheets/`); ground truth applies the same
verdict semantics as the training export via the shared
`labels.supervised_label`. Connectivity compares like for like by
translating predicted terminals into ground-truth symbol ids through the
same IoU matching the symbol gate uses; a Sheet that fails to digitize
keeps its ground truth in the denominators. `HARNESS_VERSION` stamps every
report; a `None` score fails its gate closed. Exit status 1 when a gate
fails, so model changes are gated by numbers in scripts too.

Two conscious metric choices to revisit with the corpus: tag exact-match
measures ground-truth tags read correctly and ignores predicted-text false
positives (a spammy OCR engine cannot lose the tag gate, ticket 18 should
weigh that); predicted-link translation collapses through node tags, so
duplicate tag strings on one Sheet attribute links to the first terminal
carrying the tag (commented in `_predicted_links`).

Review note: a smoke run on the real label-factory data (hand-built
profile bundle, sheet 4) scored symbol F1 1.000 / tag 0.451 /
connectivity 0.000 with the stubs — the connectivity zero is the current
line-extraction scale/tolerance behavior on 4967px rasters reported
honestly, not a harness defect; no `hazop-ai-2401@l1` profile bundle is
checked in, so real runs need one authored first.
