"""Convention Profile bundles (ticket 04): hand-built on-disk artifacts
holding the Legend Dictionary, tag grammar, and line-type semantics.
Loading validates everything up front — a bad bundle fails at load, never
mid-run — and every run artifact records the profile identity + version
that produced it."""

import json
import shutil
from pathlib import Path

import pytest

from pidgraph.model import UNCLASSIFIED_SYMBOL, LegendEntry
from pidgraph.pipeline import digitize
from pidgraph.profile import ProfileError, load_profile

BUNDLE = Path(__file__).parent / "fixtures" / "profiles" / "synthetic-test"


def editable_copy(tmp_path: Path) -> Path:
    return shutil.copytree(BUNDLE, tmp_path / "bundle")


def rewrite(bundle: Path, part: str, edit) -> None:
    path = bundle / part
    data = json.loads(path.read_text(encoding="utf-8"))
    edit(data)
    path.write_text(json.dumps(data), encoding="utf-8")


# --------------------------------------------------------------------------
# Load

def test_bundle_loads_identity_and_all_three_parts():
    profile = load_profile(BUNDLE)

    assert (profile.name, profile.version) == ("synthetic-test", "0")
    assert profile.legend["gate_valve"] == LegendEntry(
        role="PipingComponent", equipment_type="valve",
        component_class="GateValve")
    assert profile.legend["nozzle"] == LegendEntry(role="Nozzle")
    assert profile.tag_grammar["equipment_tag"] == r"[A-Z]-\d{3}"
    assert profile.line_semantics == {"pipe": "PipingNetworkSegment",
                                      "signal": "SignalLine"}
    assert profile.line_styles == {"solid": "pipe", "dashed": "signal"}


def test_line_styles_part_is_optional(tmp_path):
    bundle = editable_copy(tmp_path)
    (bundle / "line_styles.json").unlink()

    assert load_profile(bundle).line_styles == {}


# --------------------------------------------------------------------------
# Validation failure: incomplete or malformed bundles fail at load

def test_missing_part_fails_at_load_naming_the_file(tmp_path):
    bundle = editable_copy(tmp_path)
    (bundle / "line_semantics.json").unlink()

    with pytest.raises(ProfileError, match=r"line_semantics\.json: missing"):
        load_profile(bundle)


def test_malformed_json_fails_at_load_naming_the_file(tmp_path):
    bundle = editable_copy(tmp_path)
    (bundle / "legend.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(ProfileError, match=r"legend\.json: "):
        load_profile(bundle)


def test_missing_version_fails_at_load(tmp_path):
    bundle = editable_copy(tmp_path)
    rewrite(bundle, "profile.json", lambda d: d.pop("version"))

    with pytest.raises(ProfileError, match=r"profile\.json: .*version"):
        load_profile(bundle)


def test_uncompilable_tag_grammar_fails_at_load_not_mid_run(tmp_path):
    bundle = editable_copy(tmp_path)
    rewrite(bundle, "tag_grammar.json",
            lambda d: d.__setitem__("equipment_tag", "[A-Z"))

    with pytest.raises(ProfileError,
                       match=r"tag_grammar\.json: .*equipment_tag"):
        load_profile(bundle)


def test_empty_tag_grammar_is_incomplete_and_fails_at_load(tmp_path):
    bundle = editable_copy(tmp_path)
    rewrite(bundle, "tag_grammar.json", lambda d: d.clear())

    with pytest.raises(ProfileError, match=r"tag_grammar\.json: empty"):
        load_profile(bundle)


def test_unknown_legend_role_fails_at_load(tmp_path):
    bundle = editable_copy(tmp_path)
    rewrite(bundle, "legend.json",
            lambda d: d["tank"].__setitem__("role", "Vessel"))

    with pytest.raises(ProfileError, match=r"legend\.json: .*tank.*Vessel"):
        load_profile(bundle)


def test_the_reserved_unclassified_class_fails_at_load(tmp_path):
    """UNCLASSIFIED_SYMBOL is what a detector emits for ink it could not
    name (ticket 17) — a Legend Dictionary defining it as a real class
    would let assembly silently skip real symbols."""
    bundle = editable_copy(tmp_path)
    rewrite(bundle, "legend.json",
            lambda d: d.__setitem__(UNCLASSIFIED_SYMBOL,
                                    {"role": "Equipment"}))

    with pytest.raises(ProfileError, match="reserved"):
        load_profile(bundle)


def test_line_semantics_outside_dexpi_vocab_fails_at_load(tmp_path):
    bundle = editable_copy(tmp_path)
    rewrite(bundle, "line_semantics.json",
            lambda d: d.__setitem__("pipe", "Pipe"))

    with pytest.raises(ProfileError,
                       match=r"line_semantics\.json: .*'Pipe'"):
        load_profile(bundle)


def test_line_style_outside_extractor_vocab_fails_at_load(tmp_path):
    bundle = editable_copy(tmp_path)
    rewrite(bundle, "line_styles.json",
            lambda d: d.__setitem__("dotted", "signal"))

    with pytest.raises(ProfileError, match=r"line_styles\.json: .*'dotted'"):
        load_profile(bundle)


def test_every_problem_is_reported_not_just_the_first(tmp_path):
    bundle = editable_copy(tmp_path)
    rewrite(bundle, "profile.json", lambda d: d.pop("version"))
    rewrite(bundle, "legend.json",
            lambda d: d["tank"].__setitem__("role", "Vessel"))

    with pytest.raises(ProfileError) as excinfo:
        load_profile(bundle)

    message = str(excinfo.value)
    assert "profile.json" in message and "legend.json" in message


# --------------------------------------------------------------------------
# Version stamping and reproducibility at the digitize() seam

def test_run_artifacts_record_profile_identity_and_version(
        synthetic_document):
    artifacts = digitize(synthetic_document, load_profile(BUNDLE))

    stamp = {"name": "synthetic-test", "version": "0"}
    assert artifacts.detection_records[0]["profile"] == stamp
    assert artifacts.plant_model["conventionProfile"] == stamp


def test_stamped_identity_survives_to_the_on_disk_artifacts(
        tmp_path, synthetic_document):
    out_dir = tmp_path / "run"
    digitize(synthetic_document, load_profile(BUNDLE), out_dir=out_dir)

    stamp = {"name": "synthetic-test", "version": "0"}
    model = json.loads((out_dir / "plant_model_dexpi.json")
                       .read_text(encoding="utf-8"))
    record = json.loads((out_dir / "detections" / "sheet_1.json")
                        .read_text(encoding="utf-8"))
    assert model["conventionProfile"] == stamp
    assert record["profile"] == stamp


def test_same_document_and_profile_version_reproduce_identical_artifacts(
        synthetic_document):
    first = digitize(synthetic_document, load_profile(BUNDLE))
    second = digitize(synthetic_document, load_profile(BUNDLE))

    assert first.detection_records == second.detection_records
    assert first.plant_model == second.plant_model
    assert first.plant_graph == second.plant_graph
    assert first.store_summary == second.store_summary
