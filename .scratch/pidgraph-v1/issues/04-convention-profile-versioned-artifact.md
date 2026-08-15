# 04 — Convention Profile as a versioned artifact

**What to build:** An onboarding engineer creates a Convention Profile for a new company by hand — its Legend Dictionary (symbol glyphs + semantics), tag grammar (lexicons/patterns per text class), and line-type semantics — as an on-disk bundle. Profiles are versioned artifacts: a re-run of an old Document with an old profile version is reproducible, and every run records exactly which profile version produced it.

**Blocked by:** 01 (Walking skeleton).

**Status:** ready-for-agent

- [ ] A Convention Profile is authored as an on-disk bundle holding the three parts: Legend Dictionary, tag grammar, and line-type semantics.
- [ ] Loading validates the bundle and reports clear errors for a malformed or incomplete profile — a bad profile fails at load, never mid-run.
- [ ] Profiles carry an identity and version; `digitize()` accepts a loaded profile, and run artifacts (detection records, DEXPI JSON) record the profile identity + version that produced them.
- [ ] Re-running with the same Document fixture and the same profile version yields the same artifacts (reproducibility).
- [ ] Profiles are hand-built in v1 — no automated Legend Sheet ingest (explicitly out of scope).
- [ ] Tests exercise load, validation failure, and version stamping against fixture profiles; offline test invariant holds.

## Comments
