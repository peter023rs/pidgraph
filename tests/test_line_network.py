"""Pipeline-seam tests for line network extraction (ticket 05).

Synthetic fixture Sheets with exactly known topology are digitized offline
(stub detector/recognizer read the annotations; the line network is
extracted from the rendered pixels) and connectivity is asserted on the
emitted artifacts: a T-junction becomes three runs meeting at one junction
node, a line drawn through a symbol becomes two runs terminating at it, and
dashed strokes are classified by the Convention Profile's line styles.
"""

import math

from pidgraph.model import (
    Document,
    LineAnnotation,
    Sheet,
    SheetAnnotations,
    SymbolAnnotation,
    TextAnnotation,
)
from pidgraph.pipeline import digitize

from conftest import _render

SHEET_W, SHEET_H = 400, 200


def _document(name, symbols, lines, texts):
    annotations = SheetAnnotations(symbols=tuple(symbols),
                                   lines=tuple(lines), texts=tuple(texts))
    sheet = Sheet(number=1, width=SHEET_W, height=SHEET_H,
                  raster=_render(SHEET_W, SHEET_H, annotations),
                  annotations=annotations)
    return Document(name=name, sheets=(sheet,))


def _endpoints(line_record):
    return [tuple(line_record["polyline"][0]),
            tuple(line_record["polyline"][-1])]


# --------------------------------------------------------------------------
# T-junction: three runs meeting at one junction node

def t_junction_document():
    """T-201 --- + --- T-202 with a branch dropping from + to T-203; the
    junction is at (200, 100)."""
    symbols = [
        SymbolAnnotation("tank", (20.0, 80.0, 60.0, 120.0)),
        SymbolAnnotation("tank", (340.0, 80.0, 380.0, 120.0)),
        SymbolAnnotation("tank", (180.0, 150.0, 220.0, 190.0)),
    ]
    lines = [
        LineAnnotation(((60.0, 100.0), (340.0, 100.0))),
        LineAnnotation(((200.0, 100.0), (200.0, 150.0))),
    ]
    texts = [
        TextAnnotation("T-201", "equipment_tag", (20.0, 60.0, 60.0, 72.0)),
        TextAnnotation("T-202", "equipment_tag", (340.0, 60.0, 380.0, 72.0)),
        TextAnnotation("T-203", "equipment_tag", (240.0, 160.0, 280.0, 172.0)),
    ]
    return _document("t-junction.pdf", symbols, lines, texts)


def test_t_junction_yields_three_runs_meeting_at_one_junction_node(
        synthetic_profile):
    artifacts = digitize(t_junction_document(), synthetic_profile)

    lines = artifacts.detection_records[0]["lines"]
    assert len(lines) == 3
    for line in lines:  # every run has one endpoint at the branch point
        assert any(math.dist(end, (200.0, 100.0)) <= 1.5
                   for end in _endpoints(line)), line["polyline"]

    nodes = artifacts.plant_model["conceptualModel"]["PipingNode"]
    junctions = [n for n in nodes if n["nodeType"] == "junction"]
    assert len(junctions) == 1  # one shared node, not three loose ends
    assert [n for n in nodes if n["nodeType"] == "dead-end"] == []

    segments = artifacts.plant_model["conceptualModel"][
        "PipingNetworkSegment"]
    incident = [s for s in segments
                if junctions[0]["id"] in (s["from"], s["to"])]
    assert len(incident) == 3

    # deterministic classical CV: a rerun reproduces the artifacts exactly
    again = digitize(t_junction_document(), synthetic_profile)
    assert again.detection_records == artifacts.detection_records
    assert again.plant_model == artifacts.plant_model


def test_t_junction_connectivity_reaches_all_three_tanks(synthetic_profile):
    artifacts = digitize(t_junction_document(), synthetic_profile)

    edges = {(e["source"], e["target"]): e["attributes"]
             for e in artifacts.plant_graph["edges"]}
    assert set(edges) == {("T-201", "T-202"), ("T-201", "T-203"),
                          ("T-202", "T-203")}
    for attrs in edges.values():
        assert attrs["direction"] == "unknown"  # no evidence crossed the tee
        assert 0.0 < attrs["confidence"] <= 1.0
        assert attrs["provenance"]

    matches = [l for l in artifacts.cypher_script.splitlines()
               if l.startswith("MATCH")]
    assert len(matches) == 3
    for line in matches:
        assert "CONNECTED_TO" in line and "FLOWS_TO" not in line
        assert "detection_confidence" in line and "provenance" in line


# --------------------------------------------------------------------------
# Split at symbol boxes: a line drawn through a valve is two connections

def through_valve_document():
    """One straight stroke from T-301 to T-302 drawn through valve V-301 —
    the drawing every P&ID has, and exactly what must never become a single
    polyline traced through the valve."""
    symbols = [
        SymbolAnnotation("tank", (20.0, 80.0, 60.0, 120.0)),
        SymbolAnnotation("gate_valve", (170.0, 90.0, 190.0, 110.0)),
        SymbolAnnotation("tank", (320.0, 80.0, 360.0, 120.0)),
    ]
    lines = [LineAnnotation(((60.0, 100.0), (320.0, 100.0)))]
    texts = [
        TextAnnotation("T-301", "equipment_tag", (20.0, 60.0, 60.0, 72.0)),
        TextAnnotation("V-301", "equipment_tag", (165.0, 115.0, 195.0, 127.0)),
        TextAnnotation("T-302", "equipment_tag", (320.0, 60.0, 360.0, 72.0)),
    ]
    return _document("through-valve.pdf", symbols, lines, texts)


def test_line_through_symbol_box_splits_into_two_runs_at_the_symbol(
        synthetic_profile):
    artifacts = digitize(through_valve_document(), synthetic_profile)

    lines = artifacts.detection_records[0]["lines"]
    assert len(lines) == 2
    valve_edges = (170.0, 190.0)
    for line in lines:  # each run terminates at the valve box, one side each
        touching = [end for end in _endpoints(line)
                    if min(abs(end[0] - x) for x in valve_edges) <= 2.0
                    and abs(end[1] - 100.0) <= 2.0]
        assert len(touching) == 1, line["polyline"]

    # DEXPI connectivity shows the valve between the two runs
    conceptual = artifacts.plant_model["conceptualModel"]
    valve_id = conceptual["PipingComponent"][0]["id"]
    segments = conceptual["PipingNetworkSegment"]
    assert len(segments) == 2
    assert sorted(("to" if s["to"] == valve_id else "from")
                  for s in segments
                  if valve_id in (s["from"], s["to"])) == ["from", "to"]


def test_symbol_between_runs_shows_in_the_equipment_graph_and_cypher(
        synthetic_profile):
    artifacts = digitize(through_valve_document(), synthetic_profile)

    edges = {(e["source"], e["target"]): e["attributes"]
             for e in artifacts.plant_graph["edges"]}
    assert set(edges) == {("T-301", "V-301"), ("V-301", "T-302")}
    for attrs in edges.values():
        assert attrs["direction"] == "unknown"
        assert 0.0 < attrs["confidence"] <= 1.0
        assert attrs["provenance"]

    matches = [l for l in artifacts.cypher_script.splitlines()
               if l.startswith("MATCH")]
    assert sum("CONNECTED_TO" in l for l in matches) == 2


# --------------------------------------------------------------------------
# Line-type semantics from the Convention Profile: dashed = signal

def dashed_signal_document():
    """T-401 -- V-401 -- T-402 in solid process line; instrument bubble
    PI-401 connected to the valve by a dashed (signal) line."""
    symbols = [
        SymbolAnnotation("tank", (20.0, 80.0, 60.0, 120.0)),
        SymbolAnnotation("gate_valve", (140.0, 90.0, 160.0, 110.0)),
        SymbolAnnotation("tank", (320.0, 80.0, 360.0, 120.0)),
        SymbolAnnotation("instrument_bubble", (140.0, 30.0, 160.0, 50.0)),
    ]
    dashes = [((150.0, y0), (150.0, y1))
              for y0, y1 in ((52.0, 55.0), (59.0, 62.0), (66.0, 69.0),
                             (73.0, 76.0), (80.0, 83.0), (87.0, 88.0))]
    lines = ([LineAnnotation(((60.0, 100.0), (320.0, 100.0)))]
             + [LineAnnotation(d, line_class="signal") for d in dashes])
    texts = [
        TextAnnotation("T-401", "equipment_tag", (20.0, 60.0, 60.0, 72.0)),
        TextAnnotation("V-401", "equipment_tag", (135.0, 115.0, 165.0, 127.0)),
        TextAnnotation("T-402", "equipment_tag", (320.0, 60.0, 360.0, 72.0)),
        TextAnnotation("PI-401", "instrument_tag", (135.0, 10.0, 165.0, 22.0)),
    ]
    return _document("dashed-signal.pdf", symbols, lines, texts)


def test_dashed_stroke_classified_by_profile_line_styles(synthetic_profile):
    artifacts = digitize(dashed_signal_document(), synthetic_profile)

    lines = artifacts.detection_records[0]["lines"]
    assert sorted(l["line_class"] for l in lines) == ["pipe", "pipe",
                                                      "signal"]
    signal = next(l for l in lines if l["line_class"] == "signal")
    assert "dash gaps merged" in signal["provenance"]["evidence"]

    segments = artifacts.plant_model["conceptualModel"][
        "PipingNetworkSegment"]
    assert sorted(s["segmentClass"] for s in segments) == \
        ["PipingNetworkSegment", "PipingNetworkSegment", "SignalLine"]


def test_dashed_signal_line_is_one_connection_not_fragments(
        synthetic_profile):
    artifacts = digitize(dashed_signal_document(), synthetic_profile)

    edges = {(e["source"], e["target"]) for e in
             artifacts.plant_graph["edges"]}
    assert edges == {("T-401", "V-401"), ("V-401", "T-402"),
                     ("PI-401", "V-401")}
