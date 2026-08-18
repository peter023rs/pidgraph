# 06 — Lexicon-constrained decoding of tags against the tag grammar

**What to build:** Instrument tags, line numbers, and equipment tags read by the TextRecognizer are corrected against the Convention Profile's tag grammar, so a smudged "O" never silently becomes a wrong tag. The decoder is a pure function above the TextRecognizer seam — (candidate strings + tag grammar) → corrected tags with correction provenance — engine-independent by construction, and fail-closed: when no lexicon entry fits, the tag is flagged, never guessed.

**Blocked by:** 01 (Walking skeleton), 04 (Convention Profile as a versioned artifact).

**Status:** resolved

- [x] The decoder is a pure function: candidate strings + tag grammar in, corrected tags with correction provenance out — no engine, I/O, or state.
- [x] The stub TextRecognizer seeds realistic OCR noise (O/0, S/5 flips) and the decoder's corrections are proven deterministically: the corrected tag, the raw candidate, and the correction applied all appear in the detection record.
- [x] Fail-closed: a candidate that no lexicon entry or pattern fits is surfaced as an unresolved tag with its candidates — never silently accepted or guessed.
- [x] Corrected tags flow end-to-end into DEXPI JSON via `digitize()` on a synthetic fixture Sheet.
- [x] The decoder is additionally tested as a pure function across tag grammars and ambiguity cases (multiple near-matches, no match, exact match).
- [x] Offline test invariant holds; no real OCR engine is involved (that choice stays deferred behind the seam).

## Comments

Implemented in `src/pidgraph/lexicon.py` (decoder), with `TextDetection`
gaining `raw_string` / `correction` / `resolved` / `candidates` fields.
Corrections go only through a bidirectional confusion set (O/0, S/5, I/1,
B/8, Z/2) and are applied only when exactly one grammar-valid repair
exists; zero, several, or a variant explosion past the enumeration guard
all fail closed. The stub TextRecognizer now reads every 0 as O and 5 as S,
so the whole suite exercises correction end to end. `assemble_sheet` skips
unresolved texts, so an unresolved tag never names a plant item. Tests:
`tests/test_lexicon.py`.

Post-review hardening (code review, 2026-08-17): text classes with no
grammar entry now fail closed (no way to verify a read means unresolved,
not trusted); `resolved` defaults to False on `TextDetection` so only the
decoder ever grants trust; every decode branch sets the verdict fields
explicitly, making re-decoding idempotent-safe; the enumeration guard is a
variant budget (2^16) instead of a low fixed cap, so long-but-lightly-
smudged tags still repair; repaired reads carry a 0.9 confidence factor
and unresolved ones 0.5 (Workbench queue ordering); the stub's noise table
is derived from the decoder's confusion set (one source of truth). Known
limitations, accepted: under a permissive grammar a flipped read that
still matches is indistinguishable from a clean one (tight grammars are
the mitigation, noted in the module docstring); unresolved reads survive
only in detection records — surfacing them as reported gaps in the plant
model is ticket 08's gap-reporting work.
