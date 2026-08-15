"""GraphStore seam: the default store emits an offline Cypher script in the
s2_pml schema (ADR-0001). Assertions are on the emitted script — the
external artifact — mirroring hazop-ai's offline to_cypher prior art."""

import pytest

from pidgraph.cypher_store import CypherScriptGraphStore, to_cypher


def _graph():
    return {
        "nodes": [
            {"tag": "T-101", "name": "T-101", "equipment_type": "tank",
             "detection_confidence": 1.0, "attributes": {}},
            {"tag": "V-101", "name": "V-101", "equipment_type": "valve",
             "detection_confidence": 1.0, "attributes": {}},
            {"tag": "T-102", "name": "T-102", "equipment_type": "tank",
             "detection_confidence": 1.0, "attributes": {}},
        ],
        "edges": [
            {"source": "T-101", "target": "V-101", "line_tag": None,
             "attributes": {"direction": "known",
                            "direction_sources": ["arrow:p1n5"],
                            "line_numbers": ["150-GA-001"]}},
            {"source": "V-101", "target": "T-102",
             "attributes": {"direction": "unknown"}},
        ],
    }


def test_emits_s2_pml_constraint_nodes_and_relationships():
    script = to_cypher(_graph())

    assert ("CREATE CONSTRAINT plant_item_tag IF NOT EXISTS "
            "FOR (n:PlantItem) REQUIRE n.tag IS UNIQUE;") in script
    assert 'MERGE (n:PlantItem {tag: "T-101"}) SET n:Tank,' in script
    assert 'MERGE (n:PlantItem {tag: "V-101"}) SET n:Valve,' in script
    assert "detection_confidence: 1.0" in script


def test_direction_evidence_splits_flows_to_from_connected_to():
    script = to_cypher(_graph())

    assert script.count("FLOWS_TO") >= 1
    flows_line = next(l for l in script.splitlines() if "FLOWS_TO" in l and "MATCH" in l)
    assert '(a:PlantItem {tag: "T-101"})' in flows_line
    assert '(b:PlantItem {tag: "V-101"})' in flows_line
    assert 'direction_sources: ["arrow:p1n5"]' in flows_line
    assert 'line_numbers: ["150-GA-001"]' in flows_line

    connected_line = next(l for l in script.splitlines() if "CONNECTED_TO" in l and "MATCH" in l)
    assert '(a:PlantItem {tag: "V-101"})' in connected_line
    assert '(b:PlantItem {tag: "T-102"})' in connected_line
    assert 'direction: "unknown"' in connected_line


def test_no_flows_to_without_an_evidence_source():
    graph = _graph()
    graph["edges"][0]["attributes"].pop("direction_sources")

    with pytest.raises(ValueError, match="direction_sources"):
        to_cypher(graph)


def test_store_writes_inspectable_script(tmp_path):
    store = CypherScriptGraphStore()
    summary = store.store(_graph(), tmp_path)

    path = tmp_path / "plant_graph.cypher"
    assert path.exists()
    assert summary["kind"] == "cypher-script"
    assert summary["nodes"] == 3
    assert summary["relationships"] == 2
    assert "FLOWS_TO" in path.read_text(encoding="utf-8")
