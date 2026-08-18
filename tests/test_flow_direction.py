"""Flow direction asserted only from evidence (ticket 07).

Digitize-level fixtures prove each evidence rule end to end — arrow and
off-page-connector direction text become FLOWS_TO carrying their evidence
source; no evidence stays CONNECTED_TO; opposing evidence surfaces as a
conflict, never an overwrite. Assemble-level tests pin the conservative
propagation mechanics: direction crosses only two-run junctions, and
propagated provenance keeps the seed's identity.
"""

from conftest import build_sheet
from pidgraph.assemble import assemble_sheet, build_plant_graph
from pidgraph.model import (
    Document,
    LineAnnotation,
    LineDetection,
    Provenance,
    Sheet,
    SheetAnnotations,
    SymbolAnnotation,
    SymbolDetection,
    TextAnnotation,
)
from pidgraph.pipeline import digitize

_TANK1 = SymbolAnnotation("tank", (20.0, 80.0, 60.0, 140.0))
_TANK1_TAG = TextAnnotation("T-101", "equipment_tag", (20.0, 60.0, 60.0, 72.0))


def _digitize(annotations: SheetAnnotations, profile, name="fixture.pdf"):
    document = Document(name=name, sheets=(build_sheet(1, annotations),))
    return digitize(document, profile)


def _edges(artifacts) -> dict:
    return {(e["source"], e["target"]): e["attributes"]
            for e in artifacts.plant_graph["edges"]}


def _match_lines(artifacts) -> list[str]:
    return [l for l in artifacts.cypher_script.splitlines()
            if l.startswith("MATCH")]


# --------------------------------------------------------------------------
# Seeding: arrows, connector text, and the no-evidence default

def test_arrow_evidence_becomes_flows_to_with_its_source(
        synthetic_document, synthetic_profile):
    artifacts = digitize(synthetic_document, synthetic_profile)

    attrs = _edges(artifacts)[("T-101", "V-101")]
    assert attrs["direction"] == "known"
    assert attrs["direction_sources"] == ["arrow:p1-sym6"]

    directed = [s for s in artifacts.plant_model["conceptualModel"]
                ["PipingNetworkSegment"] if "flowDirection" in s]
    assert {s["flowDirectionSource"] for s in directed} == {"arrow"}

    flows = [l for l in _match_lines(artifacts) if "FLOWS_TO" in l]
    assert len(flows) == 1
    assert "arrow:p1-sym6" in flows[0]


def test_connector_direction_text_seeds_flows_to_from_the_opc(
        synthetic_profile):
    annotations = SheetAnnotations(
        symbols=(_TANK1, SymbolAnnotation("opc", (365.0, 100.0, 395.0, 120.0))),
        lines=(LineAnnotation(((60.0, 110.0), (365.0, 110.0))),),
        texts=(_TANK1_TAG,
               TextAnnotation("DW01-0001", "opc_label",
                              (367.0, 103.0, 393.0, 117.0)),
               TextAnnotation("FROM DW01-0001", "opc_direction",
                              (300.0, 88.0, 360.0, 100.0))))
    artifacts = _digitize(annotations, synthetic_profile)

    # "FROM" the connector: flow enters the sheet, OPC -> tank
    attrs = _edges(artifacts)[("OPC-DW01-0001", "T-101")]
    assert attrs["direction"] == "known"
    [source] = attrs["direction_sources"]
    assert source.startswith("connector:")

    directed = [s for s in artifacts.plant_model["conceptualModel"]
                ["PipingNetworkSegment"] if "flowDirection" in s]
    assert directed and all(s["flowDirectionSource"] == "connector"
                            for s in directed)
    flows = [l for l in _match_lines(artifacts) if "FLOWS_TO" in l]
    assert len(flows) == 1
    assert '"OPC-DW01-0001"' in flows[0]


def test_to_connector_text_directs_flow_toward_the_opc(synthetic_profile):
    annotations = SheetAnnotations(
        symbols=(_TANK1, SymbolAnnotation("opc", (365.0, 100.0, 395.0, 120.0))),
        lines=(LineAnnotation(((60.0, 110.0), (365.0, 110.0))),),
        texts=(_TANK1_TAG,
               TextAnnotation("DW01-0001", "opc_label",
                              (367.0, 103.0, 393.0, 117.0)),
               TextAnnotation("TO DW01-0001", "opc_direction",
                              (300.0, 88.0, 360.0, 100.0))))
    artifacts = _digitize(annotations, synthetic_profile)

    attrs = _edges(artifacts)[("T-101", "OPC-DW01-0001")]
    assert attrs["direction"] == "known"


def test_without_evidence_everything_stays_connected_to(synthetic_profile):
    annotations = SheetAnnotations(
        symbols=(_TANK1, SymbolAnnotation("tank", (320.0, 80.0, 360.0, 140.0))),
        lines=(LineAnnotation(((60.0, 110.0), (320.0, 110.0))),),
        texts=(_TANK1_TAG,
               TextAnnotation("T-102", "equipment_tag",
                              (320.0, 60.0, 360.0, 72.0))))
    artifacts = _digitize(annotations, synthetic_profile)

    assert all(a["direction"] == "unknown"
               for a in _edges(artifacts).values())
    assert not any("FLOWS_TO" in l for l in _match_lines(artifacts))
    assert not any("flowDirection" in s
                   for s in artifacts.plant_model["conceptualModel"]
                   ["PipingNetworkSegment"])


# --------------------------------------------------------------------------
# Conflicts: opposing evidence surfaces, never overwrites

def test_opposing_arrows_surface_a_conflict_not_an_overwrite(
        synthetic_profile):
    annotations = SheetAnnotations(
        symbols=(_TANK1, SymbolAnnotation("tank", (320.0, 80.0, 360.0, 140.0)),
                 SymbolAnnotation("flow_arrow", (115.0, 105.0, 125.0, 115.0),
                                  direction=(1.0, 0.0)),
                 SymbolAnnotation("flow_arrow", (255.0, 105.0, 265.0, 115.0),
                                  direction=(-1.0, 0.0))),
        lines=(LineAnnotation(((60.0, 110.0), (320.0, 110.0))),),
        texts=(_TANK1_TAG,
               TextAnnotation("T-102", "equipment_tag",
                              (320.0, 60.0, 360.0, 72.0))))
    artifacts = _digitize(annotations, synthetic_profile)

    [attrs] = _edges(artifacts).values()
    assert attrs["direction"] == "conflict"
    conflicts = attrs["direction_conflicts"]
    assert len(conflicts) == 2
    assert all(ref.startswith("arrow:") for ref in conflicts)

    assert not any("FLOWS_TO" in l for l in _match_lines(artifacts))
    conflict_lines = [l for l in _match_lines(artifacts)
                      if '"conflict"' in l and "CONNECTED_TO" in l]
    assert len(conflict_lines) == 1
    assert "direction_conflicts" in conflict_lines[0]

    # neither direction is asserted in DEXPI either
    assert not any("flowDirection" in s
                   for s in artifacts.plant_model["conceptualModel"]
                   ["PipingNetworkSegment"])


# --------------------------------------------------------------------------
# Conservative propagation through the drawn line network

def test_direction_propagates_across_a_two_run_junction(synthetic_profile):
    # T-101 --arrow--> ... corner gap (two-run junction) ... down to T-102:
    # the arrow's direction carries across the plain continuation.
    annotations = SheetAnnotations(
        symbols=(_TANK1, SymbolAnnotation("tank", (180.0, 140.0, 220.0, 180.0)),
                 SymbolAnnotation("flow_arrow", (115.0, 105.0, 125.0, 115.0),
                                  direction=(1.0, 0.0))),
        lines=(LineAnnotation(((60.0, 110.0), (200.0, 110.0))),
               LineAnnotation(((200.0, 112.0), (200.0, 140.0)))),
        texts=(_TANK1_TAG,
               TextAnnotation("T-102", "equipment_tag",
                              (170.0, 182.0, 230.0, 194.0))))
    artifacts = _digitize(annotations, synthetic_profile)

    attrs = _edges(artifacts)[("T-101", "T-102")]
    assert attrs["direction"] == "known"
    sources = attrs["direction_sources"]
    assert any(ref.startswith("arrow:") for ref in sources)
    assert any(ref.startswith("propagated(arrow:") for ref in sources)

    directed = [s for s in artifacts.plant_model["conceptualModel"]
                ["PipingNetworkSegment"] if "flowDirection" in s]
    assert {s["flowDirectionSource"] for s in directed} == \
        {"arrow", "propagated"}

    flows = [l for l in _match_lines(artifacts) if "FLOWS_TO" in l]
    assert len(flows) == 1


def test_propagation_stops_at_a_branch(synthetic_profile):
    # A T-junction: the arrow directs its own arm, but neither branch
    # inherits a direction — a branch is not evidence.
    annotations = SheetAnnotations(
        symbols=(_TANK1, SymbolAnnotation("tank", (320.0, 80.0, 360.0, 140.0)),
                 SymbolAnnotation("tank", (180.0, 140.0, 220.0, 180.0)),
                 SymbolAnnotation("flow_arrow", (115.0, 105.0, 125.0, 115.0),
                                  direction=(1.0, 0.0))),
        lines=(LineAnnotation(((60.0, 110.0), (320.0, 110.0))),
               LineAnnotation(((200.0, 110.0), (200.0, 140.0)))),
        texts=(_TANK1_TAG,
               TextAnnotation("T-102", "equipment_tag",
                              (320.0, 60.0, 360.0, 72.0)),
               TextAnnotation("T-103", "equipment_tag",
                              (170.0, 182.0, 230.0, 194.0))))
    artifacts = _digitize(annotations, synthetic_profile)

    assert all(a["direction"] == "unknown"
               for a in _edges(artifacts).values())
    assert not any("FLOWS_TO" in l for l in _match_lines(artifacts))
    # the arrow's own arm still records its seeded direction in DEXPI
    directed = [s for s in artifacts.plant_model["conceptualModel"]
                ["PipingNetworkSegment"] if "flowDirection" in s]
    assert directed and all(s["flowDirectionSource"] == "arrow"
                            for s in directed)


# --------------------------------------------------------------------------
# Propagation mechanics, pinned at the assembly layer

def _line(id_: str, polyline) -> LineDetection:
    return LineDetection(id=id_, sheet=1, polyline=tuple(polyline),
                         line_class="pipe", confidence=1.0,
                         provenance=Provenance("test", "drawn"))


def _tank(id_: str, bbox) -> SymbolDetection:
    return SymbolDetection(id=id_, sheet=1, symbol_class="tank", bbox=bbox,
                           confidence=1.0,
                           provenance=Provenance("test", "drawn"))


def _arrow(id_: str, bbox, direction) -> SymbolDetection:
    return SymbolDetection(id=id_, sheet=1, symbol_class="flow_arrow",
                           bbox=bbox, confidence=1.0, direction=direction,
                           provenance=Provenance("test", "drawn"))


def _sheet() -> Sheet:
    return Sheet(number=1, width=400, height=200)


def test_direction_propagates_hop_by_hop_with_seed_provenance(
        synthetic_profile):
    symbols = [_tank("tank-a", (0.0, 100.0, 20.0, 120.0)),
               _tank("tank-b", (300.0, 100.0, 320.0, 120.0)),
               _arrow("arrow-1", (55.0, 105.0, 65.0, 115.0), (1.0, 0.0))]
    lines = [_line("run-a", ((20.0, 110.0), (100.0, 110.0))),
             _line("run-b", ((100.0, 110.0), (200.0, 110.0))),
             _line("run-c", ((200.0, 110.0), (300.0, 110.0)))]

    assembly = assemble_sheet(_sheet(), symbols, lines, [],
                              synthetic_profile)
    graph = build_plant_graph([assembly])

    [edge] = graph["edges"]
    assert edge["source"] == "UNTAGGED-tank-a"
    assert edge["target"] == "UNTAGGED-tank-b"
    assert edge["attributes"]["direction"] == "known"
    assert edge["attributes"]["direction_sources"] == \
        ["arrow:arrow-1", "propagated(arrow:arrow-1)"]

    # the middle and far runs carry the propagated evidence, seed intact
    for run in assembly.runs[1:]:
        assert run.flow is not None and run.flow.propagated
        assert run.flow.evidence_id == "arrow-1"


def test_propagated_directions_meeting_head_on_conflict(synthetic_profile):
    symbols = [_tank("tank-a", (0.0, 100.0, 20.0, 120.0)),
               _tank("tank-b", (300.0, 100.0, 320.0, 120.0)),
               _arrow("arrow-1", (55.0, 105.0, 65.0, 115.0), (1.0, 0.0)),
               _arrow("arrow-2", (255.0, 105.0, 265.0, 115.0), (-1.0, 0.0))]
    lines = [_line("run-a", ((20.0, 110.0), (100.0, 110.0))),
             _line("run-b", ((100.0, 110.0), (200.0, 110.0))),
             _line("run-c", ((200.0, 110.0), (300.0, 110.0)))]

    assembly = assemble_sheet(_sheet(), symbols, lines, [],
                              synthetic_profile)
    graph = build_plant_graph([assembly])

    [edge] = graph["edges"]
    assert edge["attributes"]["direction"] == "conflict"
    assert edge["attributes"]["direction_conflicts"] == [
        "arrow:arrow-1", "arrow:arrow-2",
        "propagated(arrow:arrow-1)", "propagated(arrow:arrow-2)"]
