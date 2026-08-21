# 16 — Trained SymbolDetector behind the seam

**What to build:** A real symbol detector — fine-tuned on label-factory data with synthetic degradation — runs behind the SymbolDetector seam, selected by configuration, and is gated by the eval harness. This is the plan-of-record adaptation mechanism: per-company fine-tuning on that company's labeled examples. Fully local: training may use the second LAN machine; inference runs on the operator's MacBook.

**Blocked by:** 14 (Synthetic degradation), 15 (Eval harness).

**Status:** ready-for-human

- [x] A training procedure produces a detector from label-factory data (clean + degraded); training artifacts and weights live outside git.
- [x] The trained detector is selected by configuration behind the SymbolDetector seam; nothing above the seam changes, and detections carry confidence and provenance identifying the detector version.
- [ ] The harness scores the trained detector against the eval set; the symbol F1 ≥ 0.90 gate (provisional) is the acceptance bar. *(Passes on operating-scale eval sets, clean and degraded; the real-corpus gate is blocked on the normalization-scale revisit — see the comment.)*
- [x] Inference is fully local on the target hardware — no network endpoint is called at inference time.
- [x] The test suite is unaffected: the stub remains the test-time default, and CI still needs no GPU, models, or network.

## Comments

2026-08-20 — Implemented as `src/pidgraph/detector.py`. `train_detector`
(CLI: `python -m pidgraph.detector <roots> --name --version [--out]`)
turns label-factory datasets — pass the clean root and degraded variant
roots together — into a versioned artifact: one ink-probability
prototype per symbol class (flow arrows: one per labeled direction
sector, so a matched arrow carries its orientation as evidence and the
assembly's direction honesty holds), each with an acceptance threshold
calibrated on the training crops, so degradation in the training data
widens what the detector accepts. Crops are taken in the pipeline's own
normalized frame (`normalize_sheet`), so training and inference share
one operating scale by construction. Artifacts live outside git
(default `data/detectors/`); the manifest carries a content-hash
version that is re-verified at load — a hand-edited artifact is refused
rather than misreported. Inference is stdlib-only sliding-window
Pearson correlation (ink-count prefilter via integral image, NMS with
IoU + containment), fully local; detections carry the match score as
confidence and `symbol_detector:trained@<version>` provenance.

Seam selection: `PipelineConfig(symbol_detector="trained:<artifact
dir>")` — selection strings now support `name:options` (a registry
factory exposing `from_options`), so the artifact path rides in the
existing config strings and nothing above the seam changes shape. The
stub remains the default; the suite (293 green, mypy clean) trains
throwaway artifacts in tmp dirs — CI needs no GPU, stored models, or
network.

Gate, proven end to end offline (in `tests/test_trained_detector.py`):
trained on clean + degraded (skew 0.8°, noise σ16) glyph Sheets at the
operating scale, scored by the real harness on held-out layouts —
symbol F1 1.000 on the clean eval set and 1.000 on the degraded one,
zero false positives (tag 1.0; the degraded run's connectivity 0.667 is
line extraction under noise, above this seam).

Real-corpus smoke (local, honest): trained on the real 2401 clean set
(644 examples, ~30 s — the LAN machine is not yet needed), sheets 5–12,
held out sheet 4: symbol F1 0.028 (precision 0.014 / recall 0.361;
class-blind localization 26/61). The cause is scale, not the mechanism:
`normalize_sheet`'s `TARGET_LONG_SIDE = 400` reduces 4967 px renders to
400 px, where a gate valve is ~3×3 px and a nozzle ~1×1 px — no pixel
detector clears 0.90 there. The spec's "revisited when the real corpus
arrives" note on that constant is now load-bearing: the real-corpus
gate run needs that revisit (a detector-friendly resolution, with line
extraction retuned to it), then retraining at the new scale. That
decision is the human part of this ticket.

Also found, needs its own ticket: the full harness on the current real
label-factory data fails above the seam for stub and trained configs
alike — assembly refuses multi-token instrument tags ("terminal
p4-sym50 already tagged 'PI'; refusing to overwrite with '00201'"),
because the ticket-14 regeneration classifies both bubble tokens ('PT',
'00204') as instrument_tag and both land inside the bubble. The
component-level smoke above sidesteps assembly on purpose. Run
artifacts outside git: `data/eval/ticket16-smoke.json` (stub vs trained
comparison, 0/2 Sheets scored for both), `data/eval/ticket16-holdout/`,
and a hand-authored profile bundle at `data/profiles/hazop-ai-2401@l1/`
(ticket 15 noted none was checked in; drawing-derived grammar, so it
stays outside git).
