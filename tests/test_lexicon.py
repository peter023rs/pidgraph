"""Lexicon-constrained decoding (ticket 06).

Two layers of proof:
- decode_tags as a pure function across tag grammars and ambiguity cases
  (exact match, one correction, several near-matches, no match) — the
  decoder is engine-independent by construction, so no seam is involved;
- end-to-end through digitize(): the stub TextRecognizer seeds
  deterministic OCR noise (0 read as O, 5 read as S) and the corrected
  tag, raw candidate, and correction applied all appear in the detection
  record and flow into the DEXPI JSON and plant graph.
"""

from pidgraph.lexicon import decode_tags
from pidgraph.model import (
    Document,
    Provenance,
    Sheet,
    SheetAnnotations,
    SymbolAnnotation,
    TextAnnotation,
    TextDetection,
)
from pidgraph.pipeline import digitize

GRAMMAR = {
    "equipment_tag": r"[A-Z]-\d{3}",
    "instrument_tag": r"[A-Z]{2}-\d{3}",
    "line_number": r"\d{2,3}-[A-Z]{2}-\d{3}",
}


def _candidate(string: str, text_class: str = "equipment_tag",
               grammar: dict | None = None) -> TextDetection:
    return TextDetection(
        id="t0", sheet=1, string=string, text_class=text_class,
        bbox=(0.0, 0.0, 10.0, 10.0), confidence=1.0,
        provenance=Provenance(component="test", evidence="seeded"))


def _decode_one(string: str, text_class: str = "equipment_tag",
                grammar: dict | None = None) -> TextDetection:
    [decoded] = decode_tags([_candidate(string, text_class)],
                            GRAMMAR if grammar is None else grammar)
    return decoded


# --------------------------------------------------------------------------
# The decoder as a pure function

def test_exact_match_passes_verbatim_without_correction():
    decoded = _decode_one("T-101")

    assert decoded.string == "T-101"
    assert decoded.resolved
    assert decoded.correction is None
    assert decoded.raw_string is None
    assert decoded.candidates == ()
    assert decoded.confidence == 1.0  # nothing was repaired
    assert "matches equipment_tag grammar" in decoded.provenance.evidence


def test_smudged_O_is_corrected_to_0_with_provenance():
    decoded = _decode_one("T-1O1")

    assert decoded.string == "T-101"
    assert decoded.resolved
    assert decoded.raw_string == "T-1O1"
    assert decoded.correction == "O->0 at index 3"
    assert decoded.confidence < 1.0  # a mechanical repair is not an exact read
    assert "corrected" in decoded.provenance.evidence
    assert "T-1O1" in decoded.provenance.evidence


def test_smudged_S_is_corrected_to_5():
    decoded = _decode_one("FT-S01", text_class="instrument_tag")

    assert decoded.string == "FT-501"
    assert decoded.resolved
    assert decoded.correction == "S->5 at index 3"


def test_several_flips_in_one_candidate_are_all_described():
    decoded = _decode_one("1SO-GA-OO1", text_class="line_number")

    assert decoded.string == "150-GA-001"
    assert decoded.resolved
    assert decoded.raw_string == "1SO-GA-OO1"
    assert decoded.correction == ("S->5 at index 1, O->0 at index 2, "
                                  "O->0 at index 7, O->0 at index 8")


def test_several_near_matches_fail_closed_with_the_candidates():
    # "SO" against \d[A-Z0-9]: both "50" and "5O" are grammar-valid, so
    # the decoder must not pick one.
    decoded = _decode_one("SO", text_class="code",
                          grammar={"code": r"\d[A-Z0-9]"})

    assert not decoded.resolved
    assert decoded.string == "SO"          # kept verbatim, never guessed
    assert decoded.raw_string == "SO"
    assert decoded.correction is None
    assert decoded.candidates == ("50", "5O")
    assert "unresolved" in decoded.provenance.evidence


def test_no_fitting_correction_fails_closed():
    decoded = _decode_one("T-ABC")

    assert not decoded.resolved
    assert decoded.string == "T-ABC"
    assert decoded.raw_string == "T-ABC"
    assert decoded.candidates == ()
    assert decoded.confidence < 1.0
    assert "unresolved" in decoded.provenance.evidence


def test_class_without_grammar_is_unverifiable_and_fails_closed():
    # No grammar means no way to verify the read — surfacing it resolved
    # would let OCR noise through as verified truth.
    decoded = _decode_one("anything at all", text_class="free_text")

    assert decoded.string == "anything at all"
    assert not decoded.resolved
    assert decoded.raw_string == "anything at all"
    assert "no grammar" in decoded.provenance.evidence


def test_hopelessly_degraded_candidate_fails_closed_not_slow():
    # 20 confusable characters explode past the enumeration guard; the
    # decoder gives up honestly instead of guessing (or hanging).
    decoded = _decode_one("O" * 20, text_class="serial",
                          grammar={"serial": r"\d{20}"})

    assert not decoded.resolved
    assert decoded.string == "O" * 20
    assert "unresolved" in decoded.provenance.evidence


def test_long_but_lightly_smudged_tag_is_still_repaired():
    # The guard bounds enumeration, not tag length: 15 confusable
    # characters sit inside the budget and repair uniquely.
    decoded = _decode_one("1" * 14 + "O", text_class="serial",
                          grammar={"serial": r"\d{15}"})

    assert decoded.resolved
    assert decoded.string == "1" * 14 + "0"
    assert decoded.correction == "O->0 at index 14"


def test_redecoding_a_prior_verdict_updates_it_honestly():
    # Decoding must be idempotent-safe: a stored unresolved detection
    # re-decoded against a grammar it now satisfies becomes resolved and
    # sheds its stale candidates.
    unresolved = _decode_one("SO", text_class="code",
                             grammar={"code": r"\d[A-Z0-9]"})
    assert not unresolved.resolved

    [redecoded] = decode_tags([unresolved], {"code": r"[A-Z0-9]{2}"})

    assert redecoded.resolved
    assert redecoded.candidates == ()


def test_decoding_is_deterministic_and_leaves_input_untouched():
    texts = [_candidate("T-1O1"), _candidate("SO")]

    first = decode_tags(texts, GRAMMAR)
    second = decode_tags(texts, GRAMMAR)

    assert first == second
    assert texts[0].string == "T-1O1" and texts[1].string == "SO"


# --------------------------------------------------------------------------
# End to end through digitize(): stub noise in, corrected DEXPI JSON out

def test_stub_noise_is_corrected_into_the_detection_record(
        synthetic_document, synthetic_profile):
    record = digitize(synthetic_document,
                      synthetic_profile).detection_records[0]
    by_string = {t["string"]: t for t in record["texts"]}

    # every annotated tag came back corrected to its ground truth
    assert set(by_string) == {"T-101", "V-101", "T-102", "PI-100",
                              "150-GA-001", "DW02-0003"}
    # corrected tag + raw candidate + correction applied, all in the record
    t101 = by_string["T-101"]
    assert t101["resolved"]
    assert t101["raw_string"] == "T-IOI"
    assert t101["correction"] == ("I->1 at index 2, O->0 at index 3, "
                                  "I->1 at index 4")
    line = by_string["150-GA-001"]
    assert line["raw_string"] == "ISO-GA-OOI"
    assert "S->5 at index 1" in line["correction"]
    for text in by_string.values():
        assert text["resolved"]
        assert text["raw_string"] is not None  # every fixture tag was noisy


def test_corrected_tags_flow_end_to_end_into_dexpi_json(
        synthetic_document, synthetic_profile):
    artifacts = digitize(synthetic_document, synthetic_profile)

    conceptual = artifacts.plant_model["conceptualModel"]
    assert {e["tagName"] for e in conceptual["Equipment"]} == \
        {"T-101", "T-102"}
    assert [f["tagName"]
            for f in conceptual["ProcessInstrumentationFunction"]] == \
        ["PI-100"]
    directed = [s for s in conceptual["PipingNetworkSegment"]
                if "flowDirection" in s]
    assert directed[0]["lineNumber"] == ["150-GA-001"]

    assert {n["tag"] for n in artifacts.plant_graph["nodes"]} == \
        {"T-101", "V-101", "T-102", "PI-100", "OPC-DW02-0003"}


def test_unresolvable_tag_is_flagged_and_never_names_a_plant_item(
        synthetic_profile):
    # a tank whose tag cannot be repaired through the confusion set
    annotations = SheetAnnotations(
        symbols=(SymbolAnnotation("tank", (20.0, 80.0, 60.0, 140.0)),),
        texts=(TextAnnotation("T-8FQ", "equipment_tag",
                              (20.0, 60.0, 60.0, 72.0)),))
    document = Document(name="unresolvable.pdf", sheets=(
        Sheet(number=1, width=400, height=200,
              raster=b"\xff" * (400 * 200), annotations=annotations),))

    artifacts = digitize(document, synthetic_profile)

    [text] = artifacts.detection_records[0]["texts"]
    assert text["resolved"] is False
    assert text["string"] == "T-BFQ"      # the 8 was read as B; kept raw
    assert text["raw_string"] == "T-BFQ"

    [node] = artifacts.plant_graph["nodes"]
    assert node["tag"].startswith("UNTAGGED-")  # never tagged with a guess
