"""Lexicon-constrained decoding — a pure function above the TextRecognizer
seam: (candidate strings + tag grammar) -> tags with grammar provenance.
Engine-independent by construction.

Walking-skeleton behavior: strings are kept verbatim and stamped with
whether they match their class's grammar. Ticket 06 deepens this into real
correction (O/0, S/5 flips) with correction provenance and fail-closed
ambiguity handling.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Mapping

from .model import Provenance, TextDetection


def decode_tags(texts: list[TextDetection],
                tag_grammar: Mapping[str, str]) -> list[TextDetection]:
    decoded = []
    for text in texts:
        pattern = tag_grammar.get(text.text_class)
        if pattern is None:
            note = f"lexicon: no grammar for class {text.text_class!r}"
        elif re.fullmatch(pattern, text.string):
            note = f"lexicon: matches {text.text_class} grammar"
        else:
            note = f"lexicon: does NOT match {text.text_class} grammar"
        decoded.append(replace(
            text,
            provenance=Provenance(
                component=text.provenance.component,
                evidence=f"{text.provenance.evidence}; {note}"),
        ))
    return decoded
