# 16 — Trained SymbolDetector behind the seam

**What to build:** A real symbol detector — fine-tuned on label-factory data with synthetic degradation — runs behind the SymbolDetector seam, selected by configuration, and is gated by the eval harness. This is the plan-of-record adaptation mechanism: per-company fine-tuning on that company's labeled examples. Fully local: training may use the second LAN machine; inference runs on the operator's MacBook.

**Blocked by:** 14 (Synthetic degradation), 15 (Eval harness).

**Status:** ready-for-human

- [ ] A training procedure produces a detector from label-factory data (clean + degraded); training artifacts and weights live outside git.
- [ ] The trained detector is selected by configuration behind the SymbolDetector seam; nothing above the seam changes, and detections carry confidence and provenance identifying the detector version.
- [ ] The harness scores the trained detector against the eval set; the symbol F1 ≥ 0.90 gate (provisional) is the acceptance bar.
- [ ] Inference is fully local on the target hardware — no network endpoint is called at inference time.
- [ ] The test suite is unaffected: the stub remains the test-time default, and CI still needs no GPU, models, or network.

## Comments
