"""The emitted DEXPI JSON matches hazop-ai's plant-model contract in shape,
pinned by the contract fixture mirrored per ADR-0001. This test fails if
the shape drifts."""

import json
from pathlib import Path

from pidgraph.pipeline import digitize

CONTRACT = json.loads(
    (Path(__file__).parent / "fixtures" / "dexpi_contract_shape.json")
    .read_text(encoding="utf-8"))


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_point(v):
    return isinstance(v, list) and len(v) == 2 and all(map(_is_num, v))


_CHECKS = {
    "str": lambda v: isinstance(v, str),
    "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "num": _is_num,
    "str|null": lambda v: v is None or isinstance(v, str),
    "point": _is_point,
    "bbox": lambda v: (isinstance(v, list) and len(v) == 4
                       and all(map(_is_num, v))),
    "polyline": lambda v: (isinstance(v, list) and len(v) >= 2
                           and all(map(_is_point, v))),
    "str[]": lambda v: (isinstance(v, list)
                        and all(isinstance(s, str) for s in v)),
    "int[]": lambda v: (isinstance(v, list)
                        and all(isinstance(i, int) and not isinstance(i, bool)
                                for i in v)),
}


def _assert_record(record: dict, shape: dict, where: str) -> None:
    for key, typ in shape["required"].items():
        assert key in record, f"{where}: missing required key {key!r}"
        assert _CHECKS[typ](record[key]), \
            f"{where}: {key!r}={record[key]!r} is not a {typ}"
    for key, typ in shape["optional"].items():
        if key in record:
            assert _CHECKS[typ](record[key]), \
                f"{where}: {key!r}={record[key]!r} is not a {typ}"
    allowed = (set(shape["required"]) | set(shape["optional"])
               | set(CONTRACT["pidgraphAdditiveKeys"]))
    extras = set(record) - allowed
    assert not extras, f"{where}: undeclared keys {sorted(extras)} drift the contract"


def test_emitted_plant_model_matches_mirrored_contract(
        synthetic_document, synthetic_profile):
    model = digitize(synthetic_document, synthetic_profile).plant_model
    model = json.loads(json.dumps(model))  # must round-trip as JSON

    assert set(model) == (set(CONTRACT["topLevelKeys"])
                          | set(CONTRACT["pidgraphAdditiveTopLevelKeys"]))
    assert model["schema"] == CONTRACT["schema"]
    assert isinstance(model["sourceDrawing"], str)
    _assert_record(model["conventionProfile"],
                   CONTRACT["conventionProfileShape"], "conventionProfile")
    assert set(model["conceptualModel"]) == set(
        CONTRACT["conceptualModelKeys"])

    for type_name, records in model["conceptualModel"].items():
        assert records, (f"{type_name}: the fixture Sheet must exercise "
                         "every record type so its shape stays pinned")
        for i, record in enumerate(records):
            _assert_record(record, CONTRACT["recordShapes"][type_name],
                           f"{type_name}[{i}]")

    for i, link in enumerate(model["crossSheetLinks"]):
        _assert_record(link, CONTRACT["crossSheetLink"],
                       f"crossSheetLinks[{i}]")

    vocab = CONTRACT["vocab"]
    for segment in model["conceptualModel"]["PipingNetworkSegment"]:
        assert segment["segmentClass"] in vocab["segmentClass"]
        if "flowDirection" in segment:
            assert segment["flowDirection"] in vocab["flowDirection"]
            assert segment["flowDirectionSource"] in vocab[
                "flowDirectionSource"]
    for node in model["conceptualModel"]["PipingNode"]:
        assert node["nodeType"] in vocab["nodeType"]


def test_cross_sheet_links_match_the_mirrored_contract(synthetic_profile):
    """A line number drawn on two Sheets must surface as a crossSheetLink —
    keeping that record shape pinned, not vacuously satisfied."""
    from conftest import build_synthetic_sheet
    from pidgraph.model import Document

    document = Document(name="two-sheet-fixture.pdf",
                        sheets=(build_synthetic_sheet(1),
                                build_synthetic_sheet(2)))

    model = digitize(document, synthetic_profile).plant_model
    links = json.loads(json.dumps(model))["crossSheetLinks"]

    assert links == [{"lineNumber": "150-GA-001", "sheets": [1, 2]}]
    for i, link in enumerate(links):
        _assert_record(link, CONTRACT["crossSheetLink"],
                       f"crossSheetLinks[{i}]")


def test_every_dexpi_element_carries_confidence_and_provenance(
        synthetic_document, synthetic_profile):
    model = digitize(synthetic_document, synthetic_profile).plant_model

    for type_name, records in model["conceptualModel"].items():
        for record in records:
            where = f"{type_name} {record.get('id')}"
            assert 0.0 < record["confidence"] <= 1.0, where
            assert record["provenance"]["component"], where
            assert record["provenance"]["evidence"], where
