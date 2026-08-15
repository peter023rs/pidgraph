# 06 — Lexicon-constrained decoding of tags against the tag grammar

**What to build:** Instrument tags, line numbers, and equipment tags read by the TextRecognizer are corrected against the Convention Profile's tag grammar, so a smudged "O" never silently becomes a wrong tag. The decoder is a pure function above the TextRecognizer seam — (candidate strings + tag grammar) → corrected tags with correction provenance — engine-independent by construction, and fail-closed: when no lexicon entry fits, the tag is flagged, never guessed.

**Blocked by:** 01 (Walking skeleton), 04 (Convention Profile as a versioned artifact).

**Status:** ready-for-agent

- [ ] The decoder is a pure function: candidate strings + tag grammar in, corrected tags with correction provenance out — no engine, I/O, or state.
- [ ] The stub TextRecognizer seeds realistic OCR noise (O/0, S/5 flips) and the decoder's corrections are proven deterministically: the corrected tag, the raw candidate, and the correction applied all appear in the detection record.
- [ ] Fail-closed: a candidate that no lexicon entry or pattern fits is surfaced as an unresolved tag with its candidates — never silently accepted or guessed.
- [ ] Corrected tags flow end-to-end into DEXPI JSON via `digitize()` on a synthetic fixture Sheet.
- [ ] The decoder is additionally tested as a pure function across tag grammars and ambiguity cases (multiple near-matches, no match, exact match).
- [ ] Offline test invariant holds; no real OCR engine is involved (that choice stays deferred behind the seam).

## Comments
