# 14 — Synthetic degradation of rendered Sheets

**What to build:** Rendered Sheets from the label factory are degraded synthetically — blur, skew, noise, compression — so models trained on clean renders survive real scans. Labels stay aligned through geometric transforms, and a given seed reproduces a dataset exactly.

**Blocked by:** 13 (Label factory).

**Status:** ready-for-agent

- [ ] Degradation transforms exist for at least blur, skew, noise, and compression, individually configurable in severity and composable in sequence.
- [ ] Geometric transforms (skew) transform the labels too: boxes and text-span geometry remain aligned with the degraded raster (verified on a fixture with known geometry).
- [ ] Datasets are reproducible: the same input, transform configuration, and seed produce identical output.
- [ ] The factory pipeline can emit clean and degraded variants of the same Sheet side by side for train/eval splits.
- [ ] Degraded output lives outside git like all derived drawing content; tests run on tiny synthetic fixtures under the offline invariant.

## Comments
