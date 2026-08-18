"""Review Workbench seam tests (ticket 10): the Flask test client against
prepared run artifacts, asserting overlays render and verdicts persist as
labeled examples (prior art: hazop-ai's s1_dim app tests). Offline
throughout: the artifacts are prepared by the stub pipeline on the
synthetic fixture Sheet; the Workbench itself never runs extraction."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from pidgraph.pipeline import digitize
from pidgraph.workbench import create_app


@pytest.fixture
def run_dir(tmp_path, synthetic_document, synthetic_profile) -> Path:
    """Prepared run artifacts — the only data the Workbench reads."""
    out = tmp_path / "run"
    digitize(synthetic_document, synthetic_profile, out_dir=out)
    return out


@pytest.fixture
def client(run_dir):
    app = create_app(run_dir)
    app.testing = True
    return app.test_client()


def _overlay(html: str) -> ET.Element:
    """The page's SVG overlay, parsed — assertions read geometry off the
    rendered elements instead of grepping for attribute order."""
    start = html.index("<svg")
    end = html.index("</svg>") + len("</svg>")
    return ET.fromstring(html[start:end])


def _record(run_dir: Path) -> dict:
    return json.loads((run_dir / "detections" / "sheet_1.json")
                      .read_text(encoding="utf-8"))


def test_sheet_page_overlays_every_detection_on_the_original_raster(
        client, run_dir):
    page = client.get("/sheet/1")
    assert page.status_code == 200
    html = page.get_data(as_text=True)

    # the original raster sits under an SVG overlay in Sheet coordinates
    assert 'src="/sheet/1/raster.png"' in html
    svg = _overlay(html)
    assert svg.get("viewBox") == "0 0 400 200"

    # symbols, line runs and texts each get their own element kind —
    # visually distinguishable — positioned by the recorded geometry
    symbols = svg.findall('.//rect[@class="symbol"]')
    lines = svg.findall('.//polyline[@class="line"]')
    texts = svg.findall('.//rect[@class="text"]')
    assert len(symbols) == 7
    assert len(lines) == 3
    assert len(texts) == 6

    # gate valve drawn at (180,100)-(200,120) on the fixture Sheet
    boxes = {(s.get("x"), s.get("y"), s.get("width"), s.get("height"))
             for s in symbols}
    assert ("180", "100", "20", "20") in boxes
    # line runs land exactly where the run artifact recorded them (the
    # extractor traces ink, so its polylines are its own — the overlay's
    # job is fidelity to the record)
    recorded = {" ".join(f"{x:g},{y:g}" for x, y in line["polyline"])
                for line in _record(run_dir)["lines"]}
    assert {l.get("points") for l in lines} == recorded
    # the line-number text box, its decoded string in the tooltip
    line_number = next(t for t in texts if t.get("x") == "90")
    assert (line_number.get("y"), line_number.get("width"),
            line_number.get("height")) == ("90", "60", "10")
    assert "150-GA-001" in line_number.find("title").text

    # every overlay element names its detection for verdict-taking
    ids = [e.get("data-id") for e in symbols + lines + texts]
    assert all(ids) and len(set(ids)) == 16


def test_original_raster_is_served_from_the_run_artifact(client, run_dir):
    response = client.get("/sheet/1/raster.png")
    assert response.status_code == 200
    assert response.mimetype == "image/png"
    assert response.data == (run_dir / "sheets" / "sheet_1.png").read_bytes()


def test_sheets_absent_from_the_run_artifacts_are_404(client):
    assert client.get("/sheet/2").status_code == 404
    assert client.get("/sheet/2/raster.png").status_code == 404


def test_index_links_every_sheet_the_run_digitized(
        tmp_path, synthetic_profile):
    from conftest import build_synthetic_sheet
    from pidgraph.model import Document

    document = Document(name="three-sheets.pdf",
                        sheets=tuple(build_synthetic_sheet(n)
                                     for n in (1, 2, 3)))
    digitize(document, synthetic_profile, out_dir=tmp_path / "run")
    client = create_app(tmp_path / "run").test_client()

    html = client.get("/").get_data(as_text=True)
    positions = [html.index(f'href="/sheet/{n}"') for n in (1, 2, 3)]
    assert positions == sorted(positions)


def test_pass_verdict_persists_as_a_labeled_example_keyed_to_profile_and_sheet(
        client, run_dir):
    symbol = _record(run_dir)["symbols"][0]

    response = client.post("/sheet/1/verdicts", json={
        "detection_id": symbol["id"], "verdict": "pass"})
    assert response.status_code == 200

    # the labeled example lives with the run artifacts (outside git,
    # ADR-0001), keyed to the Convention Profile identity + version and
    # the Sheet — the partition ticket 12 exports by
    stored = json.loads(
        (run_dir / "labels" / "synthetic-test@0" / "sheet_1.json")
        .read_text(encoding="utf-8"))
    assert stored["profile"] == {"name": "synthetic-test", "version": "0"}
    assert stored["sheet"] == 1
    example = stored["examples"][symbol["id"]]
    assert example["verdict"] == "pass"
    assert example["kind"] == "symbol"
    assert example["detection"] == symbol  # the original, snapshotted
    assert example["correction"] is None

    # re-opening the Sheet's verdicts shows the stored example
    fetched = client.get("/sheet/1/verdicts").get_json()
    assert fetched["examples"][symbol["id"]] == example


def test_edit_verdicts_carry_the_correction_beside_the_original(
        client, run_dir):
    record = _record(run_dir)
    text = record["texts"][0]
    symbol = record["symbols"][0]
    line = record["lines"][0]

    corrections = {
        text["id"]: {"string": "T-101A"},
        symbol["id"]: {"bbox": [22.0, 82.0, 62.0, 142.0]},
        line["id"]: {"polyline": [[66.0, 110.0], [180.0, 110.0]]},
    }
    for detection_id, correction in corrections.items():
        response = client.post("/sheet/1/verdicts", json={
            "detection_id": detection_id, "verdict": "edit",
            "correction": correction})
        assert response.status_code == 200

    examples = client.get("/sheet/1/verdicts").get_json()["examples"]
    for original in (text, symbol, line):
        example = examples[original["id"]]
        assert example["verdict"] == "edit"
        assert example["correction"] == corrections[original["id"]]
        # the original detection stays untouched beside the correction
        assert example["detection"] == original


def test_malformed_verdicts_are_refused_and_persist_nothing(
        client, run_dir):
    record = _record(run_dir)
    symbol_id = record["symbols"][0]["id"]
    text_id = record["texts"][0]["id"]
    line_id = record["lines"][0]["id"]

    refused = [
        {"detection_id": "p9-sym99", "verdict": "pass"},   # unknown id
        {"detection_id": symbol_id, "verdict": "accept"},  # not a verdict
        {"detection_id": symbol_id},                       # verdict missing
        {"detection_id": text_id, "verdict": "edit"},      # no correction
        {"detection_id": symbol_id, "verdict": "pass",     # pass corrects
         "correction": {"bbox": [0.0, 0.0, 1.0, 1.0]}},
        {"detection_id": symbol_id, "verdict": "edit",     # tag text on a
         "correction": {"string": "V-999"}},               # symbol
        {"detection_id": text_id, "verdict": "edit",       # polyline on a
         "correction": {"polyline": [[0.0, 0.0], [1.0, 1.0]]}},  # text
        {"detection_id": line_id, "verdict": "edit",       # bbox on a line
         "correction": {"bbox": [0.0, 0.0, 1.0, 1.0]}},
        {"detection_id": text_id, "verdict": "edit",       # malformed bbox
         "correction": {"bbox": [1.0, 2.0, 3.0]}},
        {"detection_id": line_id, "verdict": "edit",       # one-point line
         "correction": {"polyline": [[0.0, 0.0]]}},
        {"detection_id": text_id, "verdict": "edit",       # empty string
         "correction": {"string": ""}},
        {"detection_id": text_id, "verdict": "edit",       # unknown field
         "correction": {"tag": "T-999"}},
        {"detection_id": text_id, "verdict": "edit",
         "correction": {}},                                # empty edit
        # JSON's NaN/Infinity leak through parsing; a non-finite
        # coordinate would poison the training export (ticket 12)
        {"detection_id": symbol_id, "verdict": "edit",
         "correction": {"bbox": [float("nan"), 0.0, 1.0, 1.0]}},
        {"detection_id": line_id, "verdict": "edit",
         "correction": {"polyline": [[0.0, 0.0], [float("inf"), 1.0]]}},
    ]
    for payload in refused:
        response = client.post("/sheet/1/verdicts", json=payload)
        assert response.status_code == 400, payload

    assert client.get("/sheet/1/verdicts").get_json()["examples"] == {}
    assert not (run_dir / "labels").exists()


def test_verdicts_survive_a_workbench_restart_and_the_latest_wins(
        client, run_dir):
    record = _record(run_dir)
    symbol_id = record["symbols"][0]["id"]
    text_id = record["texts"][0]["id"]
    client.post("/sheet/1/verdicts", json={
        "detection_id": symbol_id, "verdict": "pass"})
    client.post("/sheet/1/verdicts", json={
        "detection_id": text_id, "verdict": "pass"})
    # the reviewer reconsiders: one detection, one verdict — latest wins
    client.post("/sheet/1/verdicts", json={
        "detection_id": symbol_id, "verdict": "reject"})

    # a fresh Workbench over the same run directory — a restart
    reopened = create_app(run_dir).test_client()

    examples = reopened.get("/sheet/1/verdicts").get_json()["examples"]
    assert examples[symbol_id]["verdict"] == "reject"
    assert examples[text_id]["verdict"] == "pass"

    # re-opening the Sheet shows the verdicts on the overlay itself
    svg = _overlay(reopened.get("/sheet/1").get_data(as_text=True))
    verdicts = {e.get("data-id"): e.get("data-verdict")
                for e in svg.iter() if e.get("data-id")}
    assert verdicts[symbol_id] == "reject"
    assert verdicts[text_id] == "pass"
    undecided = [v for v in verdicts.values() if v is None]
    assert len(undecided) == 14  # 16 detections, 2 verdicts given


def test_the_workbench_has_no_path_to_invoke_extraction(client):
    """Reads run artifacts only: the Workbench module holds no reference
    into the extraction engine, and no route accepts anything but
    verdicts."""
    import pidgraph.workbench as workbench

    engine = {"pidgraph.pipeline", "pidgraph.batch", "pidgraph.seams",
              "pidgraph.intake", "pidgraph.normalize", "pidgraph.lines",
              "pidgraph.lexicon", "pidgraph.assemble", "pidgraph.dexpi",
              "pidgraph.cypher_store", "pidgraph.neo4j_store"}
    for name, value in vars(workbench).items():
        origin = getattr(value, "__module__", None) \
            if not isinstance(value, type(workbench)) \
            else value.__name__
        assert origin not in engine, \
            f"workbench.{name} reaches the engine via {origin}"

    writable = [rule.rule for rule in
                client.application.url_map.iter_rules()
                if rule.methods - {"GET", "HEAD", "OPTIONS"}]
    assert writable == ["/sheet/<int:number>/verdicts"]


def test_sheet_page_offers_one_action_per_verdict(client):
    html = client.get("/sheet/1").get_data(as_text=True)
    for verdict in ("pass", "reject", "edit"):
        assert f'data-action="{verdict}"' in html
    # the controls post to the Sheet's verdict endpoint
    assert "/sheet/1/verdicts" in html
