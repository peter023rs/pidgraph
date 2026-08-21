# 17 — Legend Dictionary nearest-neighbor classifier

**What to build:** The candidate cheap adaptation mechanism: symbols classified by nearest-neighbor match against the Convention Profile's Legend Dictionary glyphs, behind the same SymbolDetector seam, selected by configuration. Architecture is identical to the fine-tuning path — only onboarding cost differs — and the eval harness judges the two on evidence (the side-by-side against fine-tuning needs 16 to exist, but building and gating this classifier does not).

**Blocked by:** 04 (Convention Profile as a versioned artifact), 15 (Eval harness).

**Status:** ready-for-human

- [x] A nearest-neighbor classifier assigns detected symbol candidates a Legend Dictionary class from the Convention Profile's glyphs, behind the SymbolDetector seam, selected by configuration.
- [x] Classifications carry confidence (match distance) and provenance naming the Legend Dictionary entry matched.
- [x] Below-threshold matches are surfaced as low-confidence/unclassified rather than force-assigned — fail-closed like the rest of the system.
- [x] The harness scores the classifier on the eval set, and the side-by-side comparison report can include it (against the trained detector once that exists).
- [x] Deterministic tests classify synthetic glyphs against a fixture Legend Dictionary under the offline invariant.

## Comments

2026-08-21 — Implemented in `src/pidgraph/detector.py` beside the trained
detector, selected as `legend-nn:<profile bundle dir>` (the eval CLI
example `nn:symbol_detector=legend-nn:<bundle>` now works). The glyphs
are a new optional part of the Convention Profile bundle: `glyphs/` holds
one PNG crop per Legend Dictionary class (URL-quoted class-name stem, the
prototype naming), hand-cropped at the operating scale with the trained
path's 2 px paper ring, which the emitted bbox trims back off. Candidate
windows come from the same sliding scan the trained path uses (extracted
to a shared `_scan`); the nearest entry is the glyph with the highest
Pearson correlation, confidence is the match score (distance = 1 − score,
spelled out in the evidence string — higher-is-better keeps the system's
confidence convention), and provenance names the entry, glyph file, and
distance under `symbol_detector:legend-nn@<glyph content hash>`.

Fail-closed decisions worth knowing:

- The acceptance band is pinned (`accept 0.6`, `candidate floor 0.35`):
  there are no labeled crops to calibrate per-class thresholds from —
  that missing calibration data is exactly the onboarding cost this path
  avoids, priced as fixed thresholds. A window under the floor is not a
  match at all (a window detector has no candidate without correlation);
  one between floor and acceptance surfaces as the reserved
  `unclassified` class, never force-assigned. `load_profile` now refuses
  a legend defining `unclassified`, and assembly skips it the way it
  skips unresolved tags — in the record for review, never a plant item.
- A below-acceptance window whose ink is mostly inside already-kept
  detections' windows is suppressed (`_NN_CLAIMED_INK`): offset windows
  correlate off a real symbol's own ink, and surfacing them would report
  the same ink twice.
- FlowArrow glyphs are refused at load: a static glyph match asserts no
  orientation, and a direction is never guessed. This is the one place
  the architecture is not quite "identical to the fine-tuning path" —
  the trained detector learns oriented arrow prototypes, this path
  honestly cannot, so arrows stay misses priced into its symbol F1 (the
  side-by-side test shows exactly that). The comparison on the real eval
  set runs on demand as before.

Tests: `tests/test_legend_nn.py` (glyph fixtures rendered with the
trained-detector suite's class-distinct ink; classification, the
unclassified band, harness scoring, and the trained-vs-legend-nn
`compare` report — all offline and deterministic), plus the reserved
class in `tests/test_profile.py`. Full suite 311 passed.
