# 15 — Eval harness with pinned component gates

**What to build:** An ML developer runs the eval harness on demand against labeled eval sets and gets gate scores: symbol F1, tag exact-match, connectivity F1 — so any model or prompt change is gated by numbers, not impressions. The harness scores whatever SymbolDetector/TextRecognizer implementations configuration selects, and can put two implementations side by side — the mechanism by which per-company fine-tuning is compared against Legend Dictionary nearest-neighbor classification, and by which the OCR engine is chosen empirically.

**Blocked by:** 01 (Walking skeleton), 13 (Label factory).

**Status:** ready-for-agent

- [ ] The harness runs a configured pipeline against a labeled eval set and reports symbol F1, tag exact-match, and connectivity F1.
- [ ] Gates are pinned and versioned with the harness: symbol F1 ≥ 0.90, tag exact-match ≥ 0.98, connectivity F1 ≥ 0.70 (provisional, refined when the corpus lands); output states pass/fail per gate.
- [ ] Any seam implementation configuration can select is scorable — stubs included, so the harness itself is verifiable end-to-end before real models exist.
- [ ] Two configurations can be scored side by side in one comparison report (fine-tuning vs nearest-neighbor; candidate OCR engines).
- [ ] The harness is a separate concern from the test suite and never runs in CI; harness runs are fully local.
- [ ] Metric computation (matching predicted to ground-truth boxes/strings/connections, F1 math) is unit-tested deterministically under the offline invariant.

## Comments
