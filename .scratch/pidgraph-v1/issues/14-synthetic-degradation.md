# 14 — Synthetic degradation of rendered Sheets

**What to build:** Rendered Sheets from the label factory are degraded synthetically — blur, skew, noise, compression — so models trained on clean renders survive real scans. Labels stay aligned through geometric transforms, and a given seed reproduces a dataset exactly.

**Blocked by:** 13 (Label factory).

**Status:** resolved

- [x] Degradation transforms exist for at least blur, skew, noise, and compression, individually configurable in severity and composable in sequence.
- [x] Geometric transforms (skew) transform the labels too: boxes and text-span geometry remain aligned with the degraded raster (verified on a fixture with known geometry).
- [x] Datasets are reproducible: the same input, transform configuration, and seed produce identical output.
- [x] The factory pipeline can emit clean and degraded variants of the same Sheet side by side for train/eval splits.
- [x] Degraded output lives outside git like all derived drawing content; tests run on tiny synthetic fixtures under the offline invariant.

## Comments

2026-08-19 — Implemented as `src/pidgraph/degrade.py` plus a `variants`
parameter on `run_label_factory` (CLI: `--variants <file.json>`, a JSON
list of `{name, seed, transforms: [{kind, ...}]}`).

Transforms are stdlib-only over the Sheet raster form (the pngio
stance — dev-time tooling must not push an imaging dependency into the
product): box blur (severity: radius), rotation skew about the raster
center (severity: angle; bilinear resample, paper-white fill), additive
gaussian noise (severity: sigma), and JPEG-style compression (severity:
IJG quality; 8x8 DCT quantization without entropy coding — blocking and
ringing are what matter for training). Transforms compose in sequence
and preserve the canvas, keeping degraded and clean rasters
pixel-comparable. Skew exposes the exact forward point map it resamples
through, and `degrade_bbox`/`degrade_polyline` push label geometry
through the same map — alignment proven offline on a drawn fixture with
known geometry and end-to-end against real PyMuPDF ink (symbol box and
text span). Reproducibility: every draw comes from
`Random(f"{seed}/sheet{N}" + stage index + kind)`, so a dataset
reproduces exactly from configuration + seed; tests assert two runs are
byte-identical and a reseeded run is not.

Each variant is a self-contained dataset directory under
`out_dir/variants/<name>` — sheets/, labels/ (same LabelStore schema
and profile key), connectivity/, training/ — so the ticket-15 harness
points at a variant exactly the way it points at the clean set;
training examples are stamped `source "<pdf>#<variant>"`, and the
manifest echoes each variant's transform configuration and seed.

Code-reviewed (math + spec + standards axes, each finding adversarially
verified); fixed in response: labels whose geometry the skew pushes
fully off the canvas are dropped (their ink was paper-filled away — a
pass verdict must never assert invisible ink), links referencing a
vanished symbol drop with them, and partially visible boxes clip to
the canvas; label stores are recorded only after every file write, so
a Sheet failing mid-artifact reaches no training export and the clean
and degraded splits always cover the same Sheets; variant names that
collide case-insensitively are refused (they alias one directory on
APFS). Re-running with a different variant list deletes nothing — the
manifest describes the latest run alone (documented on
`run_label_factory`).

Real run (local): one real 2401 Sheet at 150 dpi through
blur1/skew1.5°/noise8/quality30 took ~40 s (pure Python; ~6 min for
the full Document) — fine for dev-time tooling. Note radius-1 blur
noticeably thins 1–2 px lines at that resolution; tune severity to the
variant's purpose. Tests: 49 new (transform core, factory integration,
degraded render fidelity), all offline on tiny synthetic rasters;
suite 220 green, mypy clean. The eval harness consuming these sets is
ticket 15.
