"""Label factory (ticket 13): projecting hazop-ai vector artifacts into
pixel-level ground-truth labels, offline on synthetic artifact fixtures.

The rendering edge (PyMuPDF) is covered separately in
test_label_factory_render.py; everything here runs stdlib-only — batch
tests inject a fake page reader, the way the Neo4j tests inject a fake
driver.
"""

import json

import pytest

from pidgraph.label_factory import (
    BoxSynthesis,
    PageRender,
    TextSpan,
    build_sheet_labels,
    run_label_factory,
    validate_artifact,
)
from pidgraph.labels import LabelStore


def make_artifact(**overrides) -> dict:
    """A tiny synthetic artifact in hazop-ai's topology_page schema:

        equipment 10 (capsule, tag E-1) --edge0-- valve 11 (gate)
        valve 11 --edge1-- corner 12 --edge2-- instrument 13 (PT 001)
        off-page 14 stands alone; nozzle 15 sits on equipment 10.
    """
    artifact = {
        "page": 4,
        "sheet_prefix": "2401-",
        "nodes": [
            {"id": 10, "kind": "equipment", "xy": [50.0, 30.0],
             "tag": "E-1", "shape": "capsule",
             "bbox": [40.0, 20.0, 60.0, 40.0]},
            {"id": 11, "kind": "valve", "xy": [100.0, 30.0],
             "ball": False, "subclass": "gate"},
            {"id": 12, "kind": "corner", "xy": [100.0, 60.0]},
            {"id": 13, "kind": "instrument", "xy": [140.0, 60.0],
             "tag": "PT 001", "full_tag": "2401-PT-001"},
            {"id": 14, "kind": "off-page", "xy": [180.0, 30.0],
             "labels": ["DW02-0003"]},
            {"id": 15, "kind": "nozzle", "xy": [60.0, 30.0],
             "equipment": 0},
        ],
        "edges": [
            {"a": 15, "b": 11, "style": "solid", "length": 40.0,
             "points": [[60.0, 30.0], [100.0, 30.0]],
             "line_numbers": ["100-GA-001"],
             "direction": "ab", "direction_source": "arrow"},
            {"a": 11, "b": 12, "style": "solid", "length": 30.0,
             "points": [[100.0, 30.0], [100.0, 60.0]]},
            {"a": 12, "b": 13, "style": "dashed", "length": 40.0,
             "points": [[100.0, 60.0], [140.0, 60.0]]},
            {"a": 15, "b": 10, "style": "internal", "length": 1.0,
             "points": [[60.0, 30.0], [59.0, 30.0]]},
        ],
        "connectors": [],
        "direction_stats": {"pipe_runs": 3, "directed": 1, "conflicts": 0},
    }
    artifact.update(overrides)
    return artifact


# --------------------------------------------------------------------------
# Artifact validation

def test_stale_artifact_without_direction_stats_is_refused():
    artifact = make_artifact()
    del artifact["direction_stats"]
    with pytest.raises(ValueError, match="direction_stats"):
        validate_artifact(artifact, "topology_page1.json")


def test_artifact_missing_top_level_keys_is_refused_by_name():
    with pytest.raises(ValueError, match="topology_page9.json"):
        validate_artifact({"page": 9}, "topology_page9.json")


def test_unknown_node_kind_is_refused_naming_the_node():
    artifact = make_artifact()
    artifact["nodes"].append({"id": 99, "kind": "mystery",
                              "xy": [1.0, 1.0]})
    with pytest.raises(ValueError, match=r"99.*mystery|mystery.*99"):
        build_sheet_labels(artifact, [], scale=1.0)


# --------------------------------------------------------------------------
# Symbol labels

def test_equipment_bbox_projects_by_scale():
    labels = build_sheet_labels(make_artifact(), [], scale=2.0)
    symbols = {s["id"]: s for s in labels.symbols}
    equipment = symbols["p4-node10"]
    assert equipment["symbol_class"] == "equipment_capsule"
    assert equipment["bbox"] == [80.0, 40.0, 120.0, 80.0]
    assert equipment["sheet"] == 4
    assert equipment["confidence"] == 1.0
    assert "node 10" in equipment["provenance"]["evidence"]


def test_center_only_symbols_get_synthesized_boxes():
    synthesis = BoxSynthesis(valve_half_pt=(5.0, 4.0),
                             instrument_half_pt=(10.0, 10.0),
                             nozzle_half_pt=(2.0, 2.0),
                             opc_half_pt=(8.0, 4.0))
    labels = build_sheet_labels(make_artifact(), [], scale=1.0,
                                synthesis=synthesis)
    symbols = {s["id"]: s for s in labels.symbols}
    assert symbols["p4-node11"]["symbol_class"] == "gate_valve"
    assert symbols["p4-node11"]["bbox"] == [95.0, 26.0, 105.0, 34.0]
    assert symbols["p4-node13"]["symbol_class"] == "instrument_bubble"
    assert symbols["p4-node13"]["bbox"] == [130.0, 50.0, 150.0, 70.0]
    assert symbols["p4-node14"]["symbol_class"] == "opc"
    assert symbols["p4-node14"]["bbox"] == [172.0, 26.0, 188.0, 34.0]
    assert symbols["p4-node15"]["symbol_class"] == "nozzle"
    assert symbols["p4-node15"]["bbox"] == [58.0, 28.0, 62.0, 32.0]


def test_pass_through_nodes_are_not_symbols():
    labels = build_sheet_labels(make_artifact(), [], scale=1.0)
    kinds = {s["id"] for s in labels.symbols}
    assert "p4-node12" not in kinds  # the corner
    assert len(labels.symbols) == 5


# --------------------------------------------------------------------------
# Text labels: ground-truth strings from the page's text layer, classed by
# matching against the artifact's tag/label/line-number strings

SPANS = [
    TextSpan("E-1", (40.0, 8.0, 52.0, 18.0)),          # equipment tag
    TextSpan("DW02-0003", (170.0, 26.0, 190.0, 34.0)),  # opc label
    TextSpan("100-GA-001", (65.0, 18.0, 95.0, 26.0)),   # line number
    TextSpan("PT", (136.0, 52.0, 144.0, 59.0)),         # tag token in bubble
    TextSpan("001", (135.0, 61.0, 145.0, 68.0)),        # tag token in bubble
    TextSpan("备注文字", (10.0, 90.0, 40.0, 100.0)),      # unmatched note
    TextSpan("   ", (0.0, 0.0, 1.0, 1.0)),              # whitespace only
]


def test_text_spans_class_by_exact_match_against_artifact_strings():
    labels = build_sheet_labels(make_artifact(), SPANS, scale=2.0)
    by_string = {t["string"]: t for t in labels.texts}
    assert by_string["E-1"]["text_class"] == "equipment_tag"
    assert by_string["DW02-0003"]["text_class"] == "opc_label"
    assert by_string["100-GA-001"]["text_class"] == "line_number"
    assert by_string["E-1"]["bbox"] == [80.0, 16.0, 104.0, 36.0]
    assert by_string["E-1"]["sheet"] == 4
    assert by_string["E-1"]["confidence"] == 1.0


def test_instrument_tag_tokens_inside_the_bubble_class_as_instrument_tag():
    labels = build_sheet_labels(make_artifact(), SPANS, scale=1.0)
    by_string = {t["string"]: t for t in labels.texts}
    assert by_string["PT"]["text_class"] == "instrument_tag"
    assert by_string["001"]["text_class"] == "instrument_tag"
    assert "PT 001" in by_string["PT"]["provenance"]["evidence"]


def test_unmatched_spans_are_free_text_and_blank_spans_are_dropped():
    labels = build_sheet_labels(make_artifact(), SPANS, scale=1.0)
    by_string = {t["string"]: t for t in labels.texts}
    assert by_string["备注文字"]["text_class"] == "free_text"
    assert len(labels.texts) == 6  # the whitespace-only span is dropped


def test_a_tag_token_outside_its_bubble_does_not_class_as_instrument():
    # same token text, but nowhere near the instrument at (140, 60)
    spans = [TextSpan("PT", (10.0, 10.0, 18.0, 17.0))]
    labels = build_sheet_labels(make_artifact(), spans, scale=1.0)
    assert labels.texts[0]["text_class"] == "free_text"


# --------------------------------------------------------------------------
# Line labels: drawn runs only, style mapped to line_class

def test_line_labels_project_polylines_and_skip_internal_edges():
    labels = build_sheet_labels(make_artifact(), [], scale=2.0)
    lines = {l["id"]: l for l in labels.lines}
    assert len(lines) == 3  # the internal nozzle-shell edge is not drawn
    assert lines["p4-edge0"]["line_class"] == "pipe"
    assert lines["p4-edge0"]["polyline"] == [[120.0, 60.0], [200.0, 60.0]]
    assert lines["p4-edge2"]["line_class"] == "signal"
    assert lines["p4-edge0"]["sheet"] == 4
    assert lines["p4-edge0"]["confidence"] == 1.0
    assert "edge 0" in lines["p4-edge0"]["provenance"]["evidence"]


# --------------------------------------------------------------------------
# Connectivity: terminal-level links in the s2_pml edge shape the
# pipeline's build_plant_graph emits (ADR-0001), nozzles folded into
# their equipment, pass-through nodes contracted, direction only from
# edge evidence

def test_direct_link_folds_nozzle_and_carries_direction_evidence():
    labels = build_sheet_labels(make_artifact(), [], scale=1.0)
    direct = [l for l in labels.links if "p4-node10" in
              (l["source"], l["target"])]
    assert direct == [{
        "source": "p4-node10",   # flow leaves the equipment (edge is ab
        "target": "p4-node11",   # from its nozzle) toward the valve
        "attributes": {
            "direction": "known",
            "direction_sources": ["arrow"],
            "line_numbers": ["100-GA-001"],
            "confidence": 1.0,
            "provenance": "label_factory:projection: hazop-ai artifact"
                          " edges 0",
        },
    }]


def test_chain_through_pass_through_node_links_terminals_undirected():
    labels = build_sheet_labels(make_artifact(), [], scale=1.0)
    chained = [l for l in labels.links if l["source"] == "p4-node11"]
    assert chained == [{
        "source": "p4-node11",
        "target": "p4-node13",
        "attributes": {
            "direction": "unknown",
            "confidence": 1.0,
            "provenance": "label_factory:projection: hazop-ai artifact"
                          " edges 1, 2",
        },
    }]
    # the isolated off-page connector appears in no link
    assert not any("p4-node14" in (l["source"], l["target"])
                   for l in labels.links)


def test_parallel_runs_between_one_pair_stay_separate_links():
    """Two drawn runs between the same terminals are two connections —
    the multiplicity build_plant_graph gives predicted edges, so the
    connectivity gate compares like for like."""
    artifact = make_artifact()
    artifact["nodes"] = [artifact["nodes"][0], artifact["nodes"][1]]
    artifact["edges"] = [
        {"a": 10, "b": 11, "style": "solid", "length": 50.0,
         "points": [[60.0, 25.0], [100.0, 25.0]]},
        {"a": 10, "b": 11, "style": "solid", "length": 50.0,
         "points": [[60.0, 35.0], [100.0, 35.0]]},
    ]
    labels = build_sheet_labels(artifact, [], scale=1.0)
    assert len(labels.links) == 2
    assert all((l["source"], l["target"]) == ("p4-node10", "p4-node11")
               for l in labels.links)
    assert [l["attributes"]["provenance"][-1] for l in labels.links] \
        == ["0", "1"]


def junction_artifact() -> dict:
    """equipment 21 --> junction 20 --> valve 22, plus valve 23 branching
    off the junction with no direction evidence."""
    return {
        "page": 7,
        "sheet_prefix": "2401-",
        "nodes": [
            {"id": 20, "kind": "junction", "xy": [50.0, 50.0]},
            {"id": 21, "kind": "equipment", "xy": [10.0, 50.0],
             "tag": "E-2", "shape": "circle",
             "bbox": [0.0, 40.0, 20.0, 60.0]},
            {"id": 22, "kind": "valve", "xy": [90.0, 50.0],
             "subclass": "gate"},
            {"id": 23, "kind": "valve", "xy": [50.0, 90.0],
             "subclass": "gate"},
        ],
        "edges": [
            {"a": 21, "b": 20, "style": "solid", "length": 40.0,
             "points": [[20.0, 50.0], [50.0, 50.0]],
             "direction": "ab", "direction_source": "arrow"},
            {"a": 20, "b": 22, "style": "solid", "length": 40.0,
             "points": [[50.0, 50.0], [90.0, 50.0]],
             "direction": "ab", "direction_source": "propagated"},
            {"a": 20, "b": 23, "style": "solid", "length": 40.0,
             "points": [[50.0, 50.0], [50.0, 90.0]]},
        ],
        "connectors": [],
        "direction_stats": {"pipe_runs": 3, "directed": 2, "conflicts": 0},
    }


def test_junction_links_all_terminal_pairs_direction_only_with_evidence():
    labels = build_sheet_labels(junction_artifact(), [], scale=1.0)
    by_pair = {(l["source"], l["target"]): l for l in labels.links}
    assert set(by_pair) == {
        ("p7-node21", "p7-node22"),   # in -> out: direction known
        ("p7-node21", "p7-node23"),   # branch without evidence
        ("p7-node22", "p7-node23"),
    }
    flow = by_pair[("p7-node21", "p7-node22")]["attributes"]
    assert flow["direction"] == "known"
    assert flow["direction_sources"] == ["arrow", "propagated"]
    assert by_pair[("p7-node21", "p7-node23")]["attributes"][
        "direction"] == "unknown"
    assert by_pair[("p7-node22", "p7-node23")]["attributes"][
        "direction"] == "unknown"


def test_nozzle_with_bad_equipment_index_is_refused():
    artifact = make_artifact()
    artifact["nodes"][5]["equipment"] = 7  # only one bboxed node on sheet
    with pytest.raises(ValueError, match="nozzle.*15|15.*nozzle"):
        build_sheet_labels(artifact, [], scale=1.0)


def test_nozzle_folds_into_a_bboxed_off_page_item():
    """On the real 2401 sheets an off-page connector drawn as an
    equipment-like glyph carries a bbox and counts in the nozzle parent
    index (verified against hazop-ai's DEXPI export, 188/188): the index
    walks the sheet's bboxed nodes in order of appearance, not only the
    equipment ones."""
    artifact = make_artifact()
    artifact["nodes"][4] = {"id": 14, "kind": "off-page",
                            "xy": [180.0, 30.0], "labels": ["DW02-0003"],
                            "shape": "circle",
                            "bbox": [170.0, 20.0, 190.0, 40.0]}
    artifact["nodes"].append({"id": 16, "kind": "nozzle",
                              "xy": [180.0, 40.0], "equipment": 1})
    artifact["edges"].append(
        {"a": 16, "b": 13, "style": "solid", "length": 20.0,
         "points": [[180.0, 40.0], [140.0, 60.0]]})
    labels = build_sheet_labels(artifact, [], scale=1.0)
    symbols = {s["id"]: s for s in labels.symbols}
    # the artifact's own bbox wins over synthesis, whatever the kind
    assert symbols["p4-node14"]["bbox"] == [170.0, 20.0, 190.0, 40.0]
    # the nozzle's run links the off-page connector, not the nozzle
    assert any(set((l["source"], l["target"]))
               == {"p4-node14", "p4-node13"} for l in labels.links)


# --------------------------------------------------------------------------
# The batch: an artifact directory in, rendered Sheets + labels +
# connectivity + a training set out, per-artifact failure isolation

PROFILE = {"name": "hazop-ai-2401", "version": "l1"}


def fake_reader(spans=(), fail_pages=()):
    """Stands in for the PyMuPDF-backed reader: a blank page of
    200 x 100 pt whose raster size honors the requested dpi."""
    def read(pdf_path, page_number, dpi):
        if page_number in fail_pages:
            raise ValueError(f"page {page_number} unreadable")
        scale = dpi / 72.0
        width, height = round(200 * scale), round(100 * scale)
        return PageRender(width=width, height=height,
                          pixels=b"\xff" * (width * height),
                          spans=tuple(spans),
                          page_width_pt=200.0, page_height_pt=100.0)
    return read


def artifact_dir_with(tmp_path, artifacts):
    directory = tmp_path / "l1_output"
    directory.mkdir()
    for name, artifact in artifacts.items():
        (directory / name).write_text(
            json.dumps(artifact, ensure_ascii=False), encoding="utf-8")
    return directory


def run(tmp_path, artifacts, **kwargs):
    directory = artifact_dir_with(tmp_path, artifacts)
    pdf = tmp_path / "2401.pdf"
    pdf.write_bytes(b"%PDF-fake")
    kwargs.setdefault("reader", fake_reader())
    kwargs.setdefault("progress", lambda outcome: None)
    return run_label_factory(directory, pdf, tmp_path / "out",
                             profile=PROFILE, dpi=144, **kwargs), \
        tmp_path / "out"


def test_batch_writes_sheets_labels_connectivity_and_training_set(tmp_path):
    report, out = run(tmp_path, {
        "topology_page4.json": make_artifact(),
        "topology_page7.json": junction_artifact(),
    }, reader=fake_reader(spans=SPANS))
    assert report["coverage"] == {"total_artifacts": 2, "succeeded": 2,
                                  "failed": 0}

    png = (out / "sheets" / "sheet_4.png").read_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    # IHDR width/height at the requested resolution: 200 x 100 pt * 2
    assert int.from_bytes(png[16:20], "big") == 400
    assert int.from_bytes(png[20:24], "big") == 200

    store = LabelStore(out / "labels")
    assert store.sheets(PROFILE) == [4, 7]
    examples = store.sheet_labels(PROFILE, 4)["examples"]
    equipment = examples["p4-node10"]
    assert equipment["verdict"] == "pass"
    assert equipment["kind"] == "symbol"
    assert equipment["detection"]["symbol_class"] == "equipment_capsule"
    assert {examples[key]["kind"] for key in examples} == {
        "symbol", "text", "line"}

    connectivity = json.loads(
        (out / "connectivity" / "sheet_4.json").read_text(encoding="utf-8"))
    assert connectivity["sheet"] == 4
    assert connectivity["profile"] == PROFILE
    assert connectivity["links"] == build_sheet_labels(
        make_artifact(), [], scale=2.0,
        source="topology_page4.json").links

    training = (out / "training").glob("*.jsonl")
    lines = [json.loads(line) for path in training
             for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == report["training"]["count"]
    assert all(line["verdict"] == "pass" for line in lines)
    assert {line["sheet"] for line in lines} == {4, 7}
    assert lines[0]["source"] == "2401.pdf"

    manifest = json.loads(
        (out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == report


def test_a_failing_artifact_is_reported_and_the_batch_continues(tmp_path):
    stale = make_artifact(page=1)
    del stale["direction_stats"]
    report, out = run(tmp_path, {
        "topology_page1.json": stale,
        "topology_page4.json": make_artifact(),
    })
    assert report["coverage"] == {"total_artifacts": 2, "succeeded": 1,
                                  "failed": 1}
    failed = [a for a in report["artifacts"] if a["status"] == "failed"]
    assert failed[0]["artifact"] == "topology_page1.json"
    assert "direction_stats" in failed[0]["reason"]
    assert LabelStore(out / "labels").sheets(PROFILE) == [4]


def test_two_artifacts_claiming_one_sheet_fail_the_second(tmp_path):
    report, _ = run(tmp_path, {
        "topology_page4.json": make_artifact(),
        "topology_page4_copy.json": make_artifact(),
    })
    statuses = {a["artifact"]: a["status"] for a in report["artifacts"]}
    assert statuses["topology_page4.json"] == "succeeded"
    assert statuses["topology_page4_copy.json"] == "failed"
    failed = [a for a in report["artifacts"] if a["status"] == "failed"]
    assert "Sheet 4" in failed[0]["reason"]


def test_a_reader_failure_fails_only_that_artifact(tmp_path):
    report, _ = run(tmp_path, {
        "topology_page4.json": make_artifact(),
        "topology_page7.json": junction_artifact(),
    }, reader=fake_reader(fail_pages=(7,)))
    statuses = {a["artifact"]: a["status"] for a in report["artifacts"]}
    assert statuses["topology_page4.json"] == "succeeded"
    assert statuses["topology_page7.json"] == "failed"


def test_all_artifacts_failing_yields_no_training_set(tmp_path):
    stale = make_artifact(page=1)
    del stale["direction_stats"]
    report, out = run(tmp_path, {"topology_page1.json": stale})
    assert report["training"] is None
    assert not (out / "training").exists()


def test_an_empty_artifact_directory_is_refused_by_name(tmp_path):
    directory = tmp_path / "empty"
    directory.mkdir()
    pdf = tmp_path / "2401.pdf"
    pdf.write_bytes(b"%PDF-fake")
    with pytest.raises(ValueError, match="empty"):
        run_label_factory(directory, pdf, tmp_path / "out",
                          profile=PROFILE, reader=fake_reader(),
                          progress=lambda outcome: None)


def test_default_output_root_is_ignored_by_git():
    import subprocess
    from pathlib import Path

    from pidgraph.label_factory import DEFAULT_OUT_DIR
    repo_root = Path(__file__).resolve().parent.parent
    if not (repo_root / ".git").exists():
        pytest.skip("not running from a git checkout")
    probe = str(DEFAULT_OUT_DIR / "sheets" / "sheet_4.png")
    try:
        result = subprocess.run(["git", "check-ignore", "-q", probe],
                                cwd=repo_root)
    except OSError:
        pytest.skip("git is not available")
    assert result.returncode == 0, (
        f"{probe} would enter git — derived drawing content must stay"
        f" outside the repository (ADR-0001)")
