# 17 — Legend Dictionary nearest-neighbor classifier

**What to build:** The candidate cheap adaptation mechanism: symbols classified by nearest-neighbor match against the Convention Profile's Legend Dictionary glyphs, behind the same SymbolDetector seam, selected by configuration. Architecture is identical to the fine-tuning path — only onboarding cost differs — and the eval harness judges the two on evidence (the side-by-side against fine-tuning needs 16 to exist, but building and gating this classifier does not).

**Blocked by:** 04 (Convention Profile as a versioned artifact), 15 (Eval harness).

**Status:** ready-for-agent

- [ ] A nearest-neighbor classifier assigns detected symbol candidates a Legend Dictionary class from the Convention Profile's glyphs, behind the SymbolDetector seam, selected by configuration.
- [ ] Classifications carry confidence (match distance) and provenance naming the Legend Dictionary entry matched.
- [ ] Below-threshold matches are surfaced as low-confidence/unclassified rather than force-assigned — fail-closed like the rest of the system.
- [ ] The harness scores the classifier on the eval set, and the side-by-side comparison report can include it (against the trained detector once that exists).
- [ ] Deterministic tests classify synthetic glyphs against a fixture Legend Dictionary under the offline invariant.

## Comments
