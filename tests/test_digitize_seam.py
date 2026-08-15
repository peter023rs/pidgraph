"""Pipeline-seam tests: digitize() the synthetic fixture Sheet and assert
on the emitted artifacts — external behavior at the seam, never internals."""

from pidgraph.pipeline import digitize


def test_detection_records_cover_every_annotated_element_with_confidence_and_provenance(
        synthetic_document, synthetic_profile):
    artifacts = digitize(synthetic_document, synthetic_profile)

    assert len(artifacts.detection_records) == 1
    record = artifacts.detection_records[0]
    assert record["sheet"] == 1

    assert len(record["symbols"]) == 7
    assert len(record["lines"]) == 3
    assert len(record["texts"]) == 6

    strings = {t["string"] for t in record["texts"]}
    assert strings == {"T-101", "V-101", "T-102", "PI-100",
                       "150-GA-001", "DW02-0003"}

    for element in (record["symbols"] + record["lines"] + record["texts"]):
        assert 0.0 < element["confidence"] <= 1.0
        assert element["provenance"]["component"]
        assert element["provenance"]["evidence"]


def test_plant_graph_asserts_direction_only_with_evidence(
        synthetic_document, synthetic_profile):
    graph = digitize(synthetic_document, synthetic_profile).plant_graph

    tags = {n["tag"]: n for n in graph["nodes"]}
    assert set(tags) == {"T-101", "V-101", "T-102", "PI-100",
                         "OPC-DW02-0003"}
    assert tags["T-101"]["equipment_type"] == "tank"
    assert tags["V-101"]["equipment_type"] == "valve"
    assert tags["PI-100"]["equipment_type"] == "instrument"
    # OPC terminals mirror hazop-ai's s2_pml adapter: OPC-<label>, "line"
    assert tags["OPC-DW02-0003"]["equipment_type"] == "line"
    assert tags["OPC-DW02-0003"]["attributes"]["labels"] == ["DW02-0003"]
    for node in graph["nodes"]:
        assert 0.0 < node["detection_confidence"] <= 1.0

    edges = {(e["source"], e["target"]): e for e in graph["edges"]}
    assert set(edges) == {("T-101", "V-101"), ("V-101", "T-102"),
                          ("T-102", "OPC-DW02-0003")}

    evidenced = edges[("T-101", "V-101")]["attributes"]
    assert evidenced["direction"] == "known"
    assert any(src.startswith("arrow:") for src in
               evidenced["direction_sources"])
    assert evidenced["line_numbers"] == ["150-GA-001"]

    for pair in (("V-101", "T-102"), ("T-102", "OPC-DW02-0003")):
        attrs = edges[pair]["attributes"]
        assert attrs["direction"] == "unknown"
        assert "direction_sources" not in attrs


def test_dexpi_model_reflects_the_drawn_sheet(synthetic_document,
                                              synthetic_profile):
    model = digitize(synthetic_document, synthetic_profile).plant_model
    conceptual = model["conceptualModel"]

    assert model["sourceDrawing"] == "synthetic-fixture.pdf"
    assert {e["tagName"] for e in conceptual["Equipment"]} == \
        {"T-101", "T-102"}
    assert [c["componentClass"] for c in conceptual["PipingComponent"]] == \
        ["GateValve"]
    assert [f["tagName"]
            for f in conceptual["ProcessInstrumentationFunction"]] == \
        ["PI-100"]
    assert [o["labels"] for o in conceptual["PipeOffPageConnector"]] == \
        [["DW02-0003"]]

    nozzles = conceptual["Nozzle"]
    assert len(nozzles) == 1
    t101_id = next(e["id"] for e in conceptual["Equipment"]
                   if e["tagName"] == "T-101")
    assert nozzles[0]["equipment"] == t101_id

    segments = conceptual["PipingNetworkSegment"]
    assert len(segments) == 4  # run1 (1 leg) + run2 (1 leg) + run3 (2 legs)
    directed = [s for s in segments if "flowDirection" in s]
    assert len(directed) == 1
    assert directed[0]["flowDirection"] == "from_to"
    assert directed[0]["flowDirectionSource"] == "arrow"
    assert directed[0]["lineNumber"] == ["150-GA-001"]
    assert directed[0]["from"] == nozzles[0]["id"]

    corners = [n for n in conceptual["PipingNode"]
               if n["nodeType"] == "corner"]
    assert [n["position"] for n in corners] == [[340.0, 152.0]]


def test_default_store_emits_honest_cypher(synthetic_document,
                                           synthetic_profile):
    artifacts = digitize(synthetic_document, synthetic_profile)

    assert artifacts.store_summary["kind"] == "cypher-script"
    script = artifacts.cypher_script
    matches = [l for l in script.splitlines() if l.startswith("MATCH")]
    assert len(matches) == 3
    assert sum("FLOWS_TO" in l for l in matches) == 1
    assert sum("CONNECTED_TO" in l for l in matches) == 2
    flows = next(l for l in matches if "FLOWS_TO" in l)
    assert '(a:PlantItem {tag: "T-101"})' in flows
    assert '(b:PlantItem {tag: "V-101"})' in flows
    assert "direction_sources" in flows


def test_run_artifacts_are_inspectable_on_disk(tmp_path, synthetic_document,
                                               synthetic_profile):
    import json

    artifacts = digitize(synthetic_document, synthetic_profile,
                         out_dir=tmp_path)

    model = json.loads((tmp_path / "plant_model_dexpi.json")
                       .read_text(encoding="utf-8"))
    assert model["schema"] == artifacts.plant_model["schema"]

    record = json.loads((tmp_path / "detections" / "sheet_1.json")
                        .read_text(encoding="utf-8"))
    assert record["sheet"] == 1
    assert len(record["symbols"]) == 7

    assert "FLOWS_TO" in (tmp_path / "plant_graph.cypher").read_text(
        encoding="utf-8")
    assert set(artifacts.paths) == {"plant_model", "detections/sheet_1",
                                    "plant_graph"}
