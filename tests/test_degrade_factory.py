"""The label factory emitting degraded dataset variants (ticket 14):
clean and degraded Sheets side by side for train/eval splits, each
variant a self-contained dataset directory reproducible from its seed.

Offline with the injected fake page reader, like the ticket-13 batch
tests; the geometric alignment of degraded labels against real rendered
ink is covered in test_degrade.py and test_label_factory_render.py.
"""

import json
from pathlib import Path

import pytest

from pidgraph.degrade import (
    Blur,
    Compression,
    Noise,
    Skew,
    Variant,
    load_variants,
)
from pidgraph.labels import LabelStore, profile_key
from test_label_factory import (
    PROFILE,
    SPANS,
    fake_reader,
    junction_artifact,
    make_artifact,
    run,
)

SCAN = Variant("scan", (Blur(1), Skew(2.0), Noise(6.0), Compression(30)),
               seed=7)


def test_variants_write_clean_and_degraded_side_by_side(tmp_path):
    report, out = run(tmp_path, {
        "topology_page4.json": make_artifact(),
        "topology_page7.json": junction_artifact(),
    }, reader=fake_reader(spans=SPANS), variants=[SCAN])
    assert report["coverage"]["failed"] == 0

    clean_png = (out / "sheets" / "sheet_4.png").read_bytes()
    degraded_png = (out / "variants" / "scan" / "sheets"
                    / "sheet_4.png").read_bytes()
    assert degraded_png[:8] == b"\x89PNG\r\n\x1a\n"
    # the canvas-size invariant: degraded dims match the clean render
    assert int.from_bytes(degraded_png[16:20], "big") == 400
    assert int.from_bytes(degraded_png[20:24], "big") == 200
    assert degraded_png != clean_png

    clean = LabelStore(out / "labels").sheet_labels(PROFILE, 4)["examples"]
    store = LabelStore(out / "variants" / "scan" / "labels")
    assert store.sheets(PROFILE) == [4, 7]
    examples = store.sheet_labels(PROFILE, 4)["examples"]
    assert set(examples) == set(clean)

    # the skew moved geometry; ids, classes and strings are untouched
    symbol, clean_symbol = (labels["p4-node10"]["detection"]
                            for labels in (examples, clean))
    assert symbol["bbox"] != clean_symbol["bbox"]
    assert symbol["symbol_class"] == clean_symbol["symbol_class"]
    text, clean_text = (labels["p4-span0"]["detection"]
                        for labels in (examples, clean))
    assert text["string"] == clean_text["string"]
    assert text["bbox"] != clean_text["bbox"]
    line, clean_line = (labels["p4-edge0"]["detection"]
                        for labels in (examples, clean))
    assert line["polyline"] != clean_line["polyline"]
    assert len(line["polyline"]) == len(clean_line["polyline"])

    # connectivity has no geometry: the variant carries the clean links
    connectivity = json.loads(
        (out / "variants" / "scan" / "connectivity" / "sheet_4.json")
        .read_text(encoding="utf-8"))
    assert connectivity == json.loads(
        (out / "connectivity" / "sheet_4.json").read_text(encoding="utf-8"))

    entry = report["variants"][0]
    assert entry["name"] == "scan"
    assert entry["seed"] == 7
    assert entry["transforms"] == [
        {"kind": "blur", "radius": 1},
        {"kind": "skew", "angle_deg": 2.0},
        {"kind": "noise", "sigma": 6.0},
        {"kind": "compression", "quality": 30},
    ]
    lines = [json.loads(line) for line in
             Path(entry["paths"]["training"])
             .read_text(encoding="utf-8").splitlines()]
    assert len(lines) == report["training"]["count"]
    assert all(line["source"] == "2401.pdf#scan" for line in lines)

    outcome = next(a for a in report["artifacts"] if a["sheet"] == 4)
    assert outcome["paths"]["variants"]["scan"]["sheet_png"] \
        == str(out / "variants" / "scan" / "sheets" / "sheet_4.png")

    manifest = json.loads(
        (out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == report


def test_variant_rasters_reproduce_from_the_seed(tmp_path):
    artifacts = {"topology_page4.json": make_artifact()}
    outs = {}
    for key, variant in (("a", SCAN), ("b", SCAN),
                         ("c", Variant("scan", SCAN.transforms, seed=8))):
        base = tmp_path / key
        base.mkdir()
        _, outs[key] = run(base, artifacts, variants=[variant])

    def variant_file(key, *parts):
        return Path(outs[key], "variants", "scan", *parts).read_bytes()

    assert variant_file("a", "sheets", "sheet_4.png") \
        == variant_file("b", "sheets", "sheet_4.png")
    assert variant_file("a", "sheets", "sheet_4.png") \
        != variant_file("c", "sheets", "sheet_4.png")
    # label geometry depends on the transforms alone, never the seed
    labels = ("labels", profile_key(PROFILE), "sheet_4.json")
    assert variant_file("a", *labels) == variant_file("c", *labels)


def test_a_photometric_variant_keeps_the_clean_geometry(tmp_path):
    report, out = run(tmp_path, {"topology_page4.json": make_artifact()},
                      reader=fake_reader(spans=SPANS),
                      variants=[Variant("noisy", (Noise(5.0),), seed=1)])
    clean = LabelStore(out / "labels").sheet_labels(PROFILE, 4)
    noisy = LabelStore(out / "variants" / "noisy"
                       / "labels").sheet_labels(PROFILE, 4)
    assert noisy["examples"] == clean["examples"]
    assert (out / "variants" / "noisy" / "sheets"
            / "sheet_4.png").read_bytes() \
        != (out / "sheets" / "sheet_4.png").read_bytes()


def test_colliding_variant_names_are_refused(tmp_path):
    with pytest.raises(ValueError, match="scan"):
        run(tmp_path, {"topology_page4.json": make_artifact()},
            variants=[Variant("scan", (Blur(1),)),
                      Variant("scan", (Noise(2.0),), seed=1)])


def test_names_colliding_only_by_case_are_refused(tmp_path):
    """'scan' and 'Scan' are one directory on a case-insensitive
    filesystem — the second dataset would silently overwrite the
    first while the manifest lists both."""
    with pytest.raises(ValueError, match="Scan"):
        run(tmp_path, {"topology_page4.json": make_artifact()},
            variants=[Variant("scan", (Blur(1),)),
                      Variant("Scan", (Noise(2.0),), seed=1)])


def test_labels_pushed_off_canvas_drop_from_the_variant(tmp_path):
    """The canvas-preserving skew paper-fills ink it rotates off the
    page: a symbol whose whole box leaves the canvas must not survive
    as a full-confidence label for invisible ink, links referencing it
    drop with it, and a partially visible box clips to the canvas."""
    artifact = make_artifact()
    # the instrument (node 13, linked to the valve through the corner)
    # moves to the far edge: a quarter turn of the 400x200 px canvas
    # carries its box entirely off the page
    artifact["nodes"][3]["xy"] = [195.0, 50.0]
    report, out = run(tmp_path, {"topology_page4.json": artifact},
                      variants=[Variant("quarter", (Skew(90.0),))])
    assert report["coverage"]["failed"] == 0

    clean = LabelStore(out / "labels").sheet_labels(PROFILE, 4)["examples"]
    degraded = LabelStore(out / "variants" / "quarter"
                          / "labels").sheet_labels(PROFILE, 4)["examples"]
    assert "p4-node13" in clean
    assert "p4-node13" not in degraded
    assert set(degraded) < set(clean)

    # the equipment box straddles the top edge after the turn: clipped
    equipment = degraded["p4-node10"]["detection"]["bbox"]
    assert equipment[1] == 0.0
    assert all(coordinate >= 0.0 for coordinate in equipment)

    def links(path):
        return json.loads(path.read_text(encoding="utf-8"))["links"]

    clean_links = links(out / "connectivity" / "sheet_4.json")
    degraded_links = links(out / "variants" / "quarter" / "connectivity"
                           / "sheet_4.json")
    assert any("p4-node13" in (link["source"], link["target"])
               for link in clean_links)
    assert not any("p4-node13" in (link["source"], link["target"])
                   for link in degraded_links)
    assert [link for link in degraded_links] \
        == [link for link in clean_links
            if "p4-node13" not in (link["source"], link["target"])]


def test_a_variant_write_failure_keeps_the_datasets_aligned(tmp_path):
    """An artifact that fails at the variant stage must contribute to
    no dataset at all: the clean and degraded training exports cover
    the same Sheets, or the manifest says failed."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "variants").write_text("not a directory", encoding="utf-8")
    report, _ = run(tmp_path, {"topology_page4.json": make_artifact()},
                    variants=[SCAN])
    assert report["coverage"] == {"total_artifacts": 1, "succeeded": 0,
                                  "failed": 1}
    assert LabelStore(out / "labels").sheets(PROFILE) == []
    assert report["training"] is None
    assert report["variants"][0]["training"] is None


def test_without_variants_the_manifest_shape_is_unchanged(tmp_path):
    report, out = run(tmp_path, {"topology_page4.json": make_artifact()})
    assert "variants" not in report
    assert "variants" not in report["artifacts"][0]["paths"]
    assert not (out / "variants").exists()


def test_a_failed_batch_writes_no_variant_dataset(tmp_path):
    stale = make_artifact(page=1)
    del stale["direction_stats"]
    report, out = run(tmp_path, {"topology_page1.json": stale},
                      variants=[SCAN])
    assert report["variants"][0]["training"] is None
    assert not (out / "variants" / "scan" / "sheets").exists()
    assert not (out / "variants" / "scan" / "training").exists()


def test_variants_load_from_a_json_file(tmp_path):
    path = tmp_path / "variants.json"
    path.write_text(json.dumps([
        {"name": "scan", "seed": 7, "transforms": [
            {"kind": "blur", "radius": 1},
            {"kind": "skew", "angle_deg": 2.0}]},
        {"name": "fax", "transforms": [
            {"kind": "compression", "quality": 10}]},
    ]), encoding="utf-8")
    assert load_variants(path) == [
        Variant("scan", (Blur(1), Skew(2.0)), seed=7),
        Variant("fax", (Compression(10),), seed=0)]


def test_an_empty_variants_file_is_refused_by_name(tmp_path):
    path = tmp_path / "variants.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="variants.json"):
        load_variants(path)


def test_cli_loads_variants_from_the_file(tmp_path, monkeypatch):
    from pidgraph import label_factory

    path = tmp_path / "variants.json"
    path.write_text(json.dumps(
        [{"name": "scan",
          "transforms": [{"kind": "noise", "sigma": 4.0}]}]),
        encoding="utf-8")
    seen = {}

    def fake_run(artifact_dir, pdf, out, profile, dpi, variants=None):
        seen["variants"] = variants
        return {"coverage": {"total_artifacts": 0, "succeeded": 0,
                             "failed": 0},
                "training": None, "paths": {}}

    monkeypatch.setattr(label_factory, "run_label_factory", fake_run)
    label_factory.main([str(tmp_path), str(tmp_path / "x.pdf"),
                        "--variants", str(path)])
    assert seen["variants"] == [Variant("scan", (Noise(4.0),), seed=0)]


def test_variant_output_root_is_ignored_by_git():
    import subprocess

    from pidgraph.label_factory import DEFAULT_OUT_DIR
    repo_root = Path(__file__).resolve().parent.parent
    if not (repo_root / ".git").exists():
        pytest.skip("not running from a git checkout")
    probe = str(DEFAULT_OUT_DIR / "variants" / "scan" / "sheets"
                / "sheet_4.png")
    try:
        result = subprocess.run(["git", "check-ignore", "-q", probe],
                                cwd=repo_root)
    except OSError:
        pytest.skip("git is not available")
    assert result.returncode == 0, (
        f"{probe} would enter git — degraded output is derived drawing"
        f" content and must stay outside the repository (ADR-0001)")
