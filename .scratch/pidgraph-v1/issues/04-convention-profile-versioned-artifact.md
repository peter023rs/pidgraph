# 04 — Convention Profile as a versioned artifact

**What to build:** An onboarding engineer creates a Convention Profile for a new company by hand — its Legend Dictionary (symbol glyphs + semantics), tag grammar (lexicons/patterns per text class), and line-type semantics — as an on-disk bundle. Profiles are versioned artifacts: a re-run of an old Document with an old profile version is reproducible, and every run records exactly which profile version produced it.

**Blocked by:** 01 (Walking skeleton).

**Status:** resolved

- [x] A Convention Profile is authored as an on-disk bundle holding the three parts: Legend Dictionary, tag grammar, and line-type semantics.
- [x] Loading validates the bundle and reports clear errors for a malformed or incomplete profile — a bad profile fails at load, never mid-run.
- [x] Profiles carry an identity and version; `digitize()` accepts a loaded profile, and run artifacts (detection records, DEXPI JSON) record the profile identity + version that produced them.
- [x] Re-running with the same Document fixture and the same profile version yields the same artifacts (reproducibility).
- [x] Profiles are hand-built in v1 — no automated Legend Sheet ingest (explicitly out of scope).
- [x] Tests exercise load, validation failure, and version stamping against fixture profiles; offline test invariant holds.

## Comments

Implemented in `src/pidgraph/profile.py`; tests in `tests/test_profile.py`
(47 total in suite, all passing offline; mypy clean).

- Bundle = a directory: `profile.json` (identity: name + version),
  `legend.json`, `tag_grammar.json`, `line_semantics.json`. The checked-in
  fixture bundle is `tests/fixtures/profiles/synthetic-test/`; conftest's
  `synthetic_profile` now loads it, so the whole suite exercises the loader.
- `load_profile()` validates everything up front and raises one
  `ProfileError` listing every problem: missing/malformed files, missing
  identity, empty Legend Dictionary or tag grammar, unknown legend roles or
  entry keys, regexes that don't compile, line semantics outside the DEXPI
  segmentClass vocabulary (`SEGMENT_CLASSES` in `dexpi.py`). Empty
  `line_semantics.json` is allowed — emission has a documented default.
- Stamping: `ConventionProfile.identity_record()` is the single owner of
  the stamp shape; detection records carry it as `"profile"`, DEXPI JSON as
  top-level `"conventionProfile"` — declared in the contract fixture via a
  new `pidgraphAdditiveTopLevelKeys` (same mechanism as the existing
  record-level additive keys). Per ADR-0001, flag for coordination with
  hazop-ai's contract when reintegration nears.
- Deliberately not done: (a) no glyph representation in `legend.json` —
  nothing in the engine consumes glyphs until the trained detector /
  legend-NN work (tickets 16/17); adding glyph storage now would be
  speculative. (b) tag-grammar text-class keys and line-class keys are not
  validated at load — the profile *defines* those vocabularies; the engine
  side references them at run time, so there is no independent source of
  truth to validate against at load.
