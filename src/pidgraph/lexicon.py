"""Lexicon-constrained decoding — a pure function above the TextRecognizer
seam: (candidate strings + tag grammar) -> corrected tags with correction
provenance. Engine-independent by construction: no engine, I/O, or state.

A candidate that already matches its class's grammar passes through
verbatim. One that does not is repaired only through the OCR confusion set
(O/0, S/5, ...), and only when exactly one grammar-valid repair exists —
so a smudged "O" never silently becomes a wrong tag. Fail-closed: zero
fits, several fits, no grammar to verify against, or a candidate degraded
past the enumeration budget all leave the detection unresolved — flagged,
never guessed. Only this decoder ever sets resolved=True.

Known limitation: the grammar is the whole lexicon. Under a permissive
grammar a flipped read that still matches (e.g. "T-1O1" against
[A-Z]-[A-Z0-9]{3}) is indistinguishable from a clean one; tight per-class
grammars are what give the decoder its power.
"""

from __future__ import annotations

import re
from dataclasses import replace
from itertools import product
from typing import Mapping

from .model import Provenance, TextDetection

# Glyph pairs a text recognizer plausibly swaps, applied both ways.
# Shared source of truth: the stub TextRecognizer derives its seeded noise
# from these same pairs, so the suite's correction proof cannot drift.
CONFUSION_PAIRS = (("O", "0"), ("S", "5"), ("I", "1"), ("B", "8"),
                   ("Z", "2"))
CONFUSABLE: dict[str, str] = {}
for _letter, _digit in CONFUSION_PAIRS:
    CONFUSABLE[_letter] = _digit
    CONFUSABLE[_digit] = _letter

# Enumeration budget (variants, i.e. 2^confusable-chars): bounds work per
# candidate without punishing long-but-lightly-smudged tags. Past it the
# candidate is too degraded to repair mechanically — fail closed.
_MAX_VARIANTS = 1 << 16

# Deterministic confidence discounts: a repaired read is not an exact one,
# and a fail-closed read is what the Review Workbench should see first.
_CORRECTED_FACTOR = 0.9
_UNRESOLVED_FACTOR = 0.5


def _variants(candidate: str) -> list[str] | None:
    """Every string reachable by swapping confusable characters (the
    candidate itself included), or None past the enumeration budget."""
    options = [(char, CONFUSABLE[char]) if char in CONFUSABLE else (char,)
               for char in candidate]
    count = 1
    for chars in options:
        count *= len(chars)
        if count > _MAX_VARIANTS:
            return None
    return ["".join(chars) for chars in product(*options)]


def _swaps(raw: str, corrected: str) -> str:
    return ", ".join(f"{r}->{c} at index {i}"
                     for i, (r, c) in enumerate(zip(raw, corrected))
                     if r != c)


def _noted(text: TextDetection, note: str, **changes) -> TextDetection:
    return replace(
        text,
        provenance=Provenance(
            component=text.provenance.component,
            evidence=f"{text.provenance.evidence}; lexicon: {note}"),
        **changes)


def _unresolved(text: TextDetection, note: str,
                candidates: tuple[str, ...] = ()) -> TextDetection:
    return _noted(text, note, resolved=False, raw_string=text.string,
                  correction=None, candidates=candidates,
                  confidence=text.confidence * _UNRESOLVED_FACTOR)


def _decode(text: TextDetection, pattern: str | None) -> TextDetection:
    if pattern is None:
        return _unresolved(
            text, f"no grammar for class {text.text_class!r} — cannot "
                  "verify, unresolved")
    if re.fullmatch(pattern, text.string):
        return _noted(text, f"matches {text.text_class} grammar",
                      resolved=True, candidates=())

    variants = _variants(text.string)
    if variants is None:
        return _unresolved(
            text, f"{text.string!r} is too degraded to repair (more than "
                  f"{_MAX_VARIANTS} confusion-set variants) — unresolved")
    fits = sorted(v for v in variants
                  if v != text.string and re.fullmatch(pattern, v))
    if len(fits) == 1:
        corrected = fits[0]
        correction = _swaps(text.string, corrected)
        return _noted(
            text, f"corrected {text.string!r} -> {corrected!r} "
                  f"({correction}) to match {text.text_class} grammar",
            string=corrected, raw_string=text.string, correction=correction,
            resolved=True, candidates=(),
            confidence=text.confidence * _CORRECTED_FACTOR)
    if not fits:
        return _unresolved(
            text, f"{text.string!r} does not match {text.text_class} "
                  "grammar and no confusion-set repair does — unresolved")
    return _unresolved(
        text, f"{text.string!r} has {len(fits)} grammar-valid repairs "
              f"{fits} — ambiguous, unresolved",
        candidates=tuple(fits))


def decode_tags(texts: list[TextDetection],
                tag_grammar: Mapping[str, str]) -> list[TextDetection]:
    return [_decode(text, tag_grammar.get(text.text_class))
            for text in texts]


def classify_candidate(candidate: str,
                       tag_grammar: Mapping[str, str]) -> str | None:
    """Which tag-grammar class a raw read belongs to — for a recognizer
    that reads pixels without knowing the class (the stub reads it off
    the annotations; a real engine cannot). Exact: the one class whose
    grammar the read fullmatches. Failing that, the one class a
    confusion-set repair reaches — the same reachability the decoder
    repairs by, so the string itself is left for the decoder to fix.
    Zero fits, several fits, or a read too degraded to enumerate all
    give None: the adapter never guesses between grammars, and the
    decoder then fails the read closed."""
    exact = sorted(text_class for text_class, pattern in tag_grammar.items()
                   if re.fullmatch(pattern, candidate))
    if exact:
        return exact[0] if len(exact) == 1 else None
    variants = _variants(candidate)
    if variants is None:
        return None
    reachable = sorted(
        text_class for text_class, pattern in tag_grammar.items()
        if any(re.fullmatch(pattern, variant) for variant in variants))
    return reachable[0] if len(reachable) == 1 else None
