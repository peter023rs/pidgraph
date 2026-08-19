"""Training-set export seam tests (ticket 12): reviewer verdicts leave
the LabelStore as a supervised training set per Convention Profile —
pass verdicts as positives, rejects as negatives, edits carrying the
human-corrected label. Offline throughout: the verdict store fixture is
prepared directly through the LabelStore, and the export touches nothing
but it."""

import json
from pathlib import Path

import pytest

from pidgraph.labels import LabelStore, make_example, profile_key
from pidgraph.training import export_training_set, main

# a name that needs percent-quoting on disk, so the export resolves the
# store partition exactly the way the LabelStore writes it
PROFILE = {"name": "acme p&id/rev A", "version": "3"}
OTHER = {"name": "other", "version": "1"}


def _symbol(number: int, sheet: int = 1, confidence: float = 0.4) -> dict:
    return {"id": f"p{sheet}-sym{number}", "sheet": sheet,
            "symbol_class": "gate_valve",
            "bbox": [10.0 * number, 0.0, 10.0 * number + 8.0, 8.0],
            "confidence": confidence}


def _line(number: int, sheet: int = 1) -> dict:
    return {"id": f"p{sheet}-line{number}", "sheet": sheet,
            "line_class": "process",
            "polyline": [[0.0, 5.0 * number], [40.0, 5.0 * number]],
            "confidence": 0.8}


def _text(number: int, sheet: int = 1) -> dict:
    return {"id": f"p{sheet}-txt{number}", "sheet": sheet,
            "string": f"V-10{number}", "text_class": "equipment_tag",
            "bbox": [0.0, 10.0 * number, 30.0, 10.0 * number + 10.0],
            "confidence": 0.6}


@pytest.fixture
def labels_root(tmp_path) -> Path:
    """A verdict store two profiles deep, the way multi-Document review
    leaves it: all three verdict kinds, all three detection kinds, two
    Sheets — and a second profile sharing detection ids, so filtering is
    actually exercised."""
    root = tmp_path / "labels"
    store = LabelStore(root)
    store.record(PROFILE, 1, make_example("symbol", _symbol(1), "pass"))
    store.record(PROFILE, 1, make_example("line", _line(1), "reject"))
    store.record(PROFILE, 1, make_example(
        "text", _text(1), "edit",
        {"string": "V-101A", "bbox": [0.0, 10.0, 32.0, 20.0]}))
    store.record(PROFILE, 2, make_example(
        "symbol", _symbol(1, sheet=2), "edit",
        {"bbox": [12.0, 0.0, 20.0, 8.0]}))
    store.record(PROFILE, 2, make_example(
        "line", _line(1, sheet=2), "edit",
        {"polyline": [[0.0, 5.0], [20.0, 5.0], [20.0, 30.0]]}))
    store.record(OTHER, 1, make_example("symbol", _symbol(1), "reject"))
    return root


def _export(labels_root: Path, out: Path, profile=PROFILE) -> list[dict]:
    summary = export_training_set(labels_root, profile, out)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert summary["count"] == len(lines)
    return [json.loads(line) for line in lines]


def test_export_filters_to_one_profile_and_carries_it_per_example(
        labels_root, tmp_path):
    examples = _export(labels_root, tmp_path / "set.jsonl")
    assert len(examples) == 5  # OTHER's verdict on the same id is absent
    assert all(e["profile"] == PROFILE for e in examples)

    other = _export(labels_root, tmp_path / "other.jsonl", OTHER)
    assert [(e["profile"], e["verdict"]) for e in other] == \
        [(OTHER, "reject")]


def test_all_three_verdict_kinds_export_usefully(labels_root, tmp_path):
    examples = _export(labels_root, tmp_path / "set.jsonl")
    by_id = {e["detection"]["id"]: e for e in examples}

    # pass: a positive example — the detection confirmed as ground truth
    confirmed = by_id["p1-sym1"]
    assert confirmed["verdict"] == "pass"
    assert confirmed["example"] == "positive"
    assert confirmed["label"] == {"symbol_class": "gate_valve",
                                  "bbox": [10.0, 0.0, 18.0, 8.0]}

    # reject: a negative example — the detection known to be wrong, so
    # there is no ground-truth label
    negative = by_id["p1-line1"]
    assert negative["verdict"] == "reject"
    assert negative["example"] == "negative"
    assert negative["label"] is None

    # edit: the human correction overrides exactly the corrected fields
    corrected_text = by_id["p1-txt1"]
    assert corrected_text["example"] == "corrected"
    assert corrected_text["label"] == {
        "text_class": "equipment_tag", "string": "V-101A",
        "bbox": [0.0, 10.0, 32.0, 20.0]}
    assert corrected_text["detection"]["string"] == "V-101"  # untouched

    corrected_symbol = by_id["p2-sym1"]
    assert corrected_symbol["label"] == {
        "symbol_class": "gate_valve", "bbox": [12.0, 0.0, 20.0, 8.0]}
    corrected_line = by_id["p2-line1"]
    assert corrected_line["label"] == {
        "line_class": "process",
        "polyline": [[0.0, 5.0], [20.0, 5.0], [20.0, 30.0]]}


def test_examples_reference_sheets_and_snapshot_the_detection(
        labels_root, tmp_path):
    examples = _export(labels_root, tmp_path / "set.jsonl")
    for example in examples:
        assert isinstance(example["sheet"], int)
        assert example["kind"] in ("symbol", "line", "text")
        # the original detection rides along untouched — model input and
        # provenance for every example, whatever the verdict
        detection = example["detection"]
        assert "confidence" in detection
        assert "bbox" in detection or "polyline" in detection


def test_the_store_partition_names_the_sheet_not_the_snapshot(tmp_path):
    """Labels key on the Sheet they were recorded under — the store
    partition — even if a snapshotted detection's internal field
    disagrees (same identity rule as the Workbench's)."""
    store = LabelStore(tmp_path / "labels")
    store.record(PROFILE, 7, make_example("symbol", _symbol(1), "pass"))
    out = tmp_path / "set.jsonl"
    export_training_set(tmp_path / "labels", PROFILE, out)
    [example] = [json.loads(line) for line in
                 out.read_text(encoding="utf-8").splitlines()]
    assert example["sheet"] == 7
    assert example["detection"]["sheet"] == 1  # snapshot untouched


def test_partial_corrections_keep_the_uncorrected_fields(tmp_path):
    # the Workbench's text-edit prompt produces exactly {string: ...} —
    # the corrected label must keep the recorded bbox, and vice versa
    store = LabelStore(tmp_path / "labels")
    store.record(PROFILE, 1, make_example(
        "text", _text(1), "edit", {"string": "V-101B"}))
    store.record(PROFILE, 1, make_example(
        "text", _text(2), "edit", {"bbox": [1.0, 2.0, 3.0, 4.0]}))
    out = tmp_path / "set.jsonl"
    export_training_set(tmp_path / "labels", PROFILE, out)
    by_id = {e["detection"]["id"]: e for e in
             (json.loads(line) for line in
              out.read_text(encoding="utf-8").splitlines())}
    assert by_id["p1-txt1"]["label"] == {
        "text_class": "equipment_tag", "string": "V-101B",
        "bbox": [0.0, 10.0, 30.0, 20.0]}
    assert by_id["p1-txt2"]["label"] == {
        "text_class": "equipment_tag", "string": "V-102",
        "bbox": [1.0, 2.0, 3.0, 4.0]}


def test_export_is_sorted_by_sheet_then_detection_id(
        labels_root, tmp_path):
    examples = _export(labels_root, tmp_path / "set.jsonl")
    keys = [(e["sheet"], e["detection"]["id"]) for e in examples]
    assert keys == sorted(keys)


def test_reexport_is_byte_identical_and_insertion_order_free(
        labels_root, tmp_path):
    first = tmp_path / "first.jsonl"
    export_training_set(labels_root, PROFILE, first)
    again = tmp_path / "again.jsonl"
    export_training_set(labels_root, PROFILE, again)
    assert first.read_bytes() == again.read_bytes()

    # a store holding the same verdicts recorded in a different order
    # exports the same bytes: the set depends on the store's content,
    # not on the review session's history
    reordered_root = tmp_path / "reordered"
    store = LabelStore(reordered_root)
    store.record(PROFILE, 2, make_example(
        "line", _line(1, sheet=2), "edit",
        {"polyline": [[0.0, 5.0], [20.0, 5.0], [20.0, 30.0]]}))
    store.record(PROFILE, 2, make_example(
        "symbol", _symbol(1, sheet=2), "edit",
        {"bbox": [12.0, 0.0, 20.0, 8.0]}))
    store.record(PROFILE, 1, make_example(
        "text", _text(1), "edit",
        {"string": "V-101A", "bbox": [0.0, 10.0, 32.0, 20.0]}))
    store.record(PROFILE, 1, make_example("line", _line(1), "reject"))
    store.record(PROFILE, 1, make_example("symbol", _symbol(1), "pass"))
    reordered = tmp_path / "reordered.jsonl"
    export_training_set(reordered_root, PROFILE, reordered)
    assert reordered.read_bytes() == first.read_bytes()

    # re-export onto the same path replaces it whole (write-then-
    # replace, like the LabelStore): same bytes, no temp file beside it
    export_training_set(labels_root, PROFILE, first)
    assert first.read_bytes() == again.read_bytes()
    assert list(first.parent.glob("*.tmp")) == []


def test_labels_recorded_for_another_identity_are_refused(
        labels_root, tmp_path):
    """A case-insensitive filesystem can resolve a case-typo'd profile
    to another partition's directory; every store file records the
    profile it was written for, and the export refuses to re-stamp
    verdicts with an identity the file itself contradicts."""
    path = labels_root / profile_key(OTHER) / "sheet_1.json"
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["profile"] = {"name": "Other", "version": "1"}
    path.write_text(json.dumps(stored), encoding="utf-8")

    with pytest.raises(ValueError) as refusal:
        export_training_set(labels_root, OTHER, tmp_path / "set.jsonl")
    assert "Other" in str(refusal.value)
    assert not (tmp_path / "set.jsonl").exists()


def test_an_empty_profile_directory_is_refused_not_exported_empty(
        labels_root, tmp_path):
    # an interrupted first save leaves the directory without any sheet
    # file (write-then-replace protects the file, not the directory);
    # that must refuse like a typo'd profile, never yield an empty set
    (labels_root / "empty@1").mkdir()
    with pytest.raises(ValueError) as refusal:
        export_training_set(labels_root, {"name": "empty",
                                          "version": "1"},
                            tmp_path / "set.jsonl")
    assert "no labeled examples" in str(refusal.value)
    assert not (tmp_path / "set.jsonl").exists()


def test_malformed_store_records_are_refused_by_name(tmp_path):
    # the store's write contract validates verdicts and corrections but
    # not kinds or detection contents; the export names what it refuses
    # instead of dying with a bare KeyError
    store = LabelStore(tmp_path / "labels")
    store.record(PROFILE, 1, make_example(
        "blob", {"id": "p1-x1", "sheet": 1}, "reject"))
    with pytest.raises(ValueError) as unknown_kind:
        export_training_set(tmp_path / "labels", PROFILE,
                            tmp_path / "a.jsonl")
    assert "blob" in str(unknown_kind.value)
    assert "p1-x1" in str(unknown_kind.value)

    incomplete = LabelStore(tmp_path / "incomplete")
    incomplete.record(PROFILE, 1, make_example(
        "symbol", {"id": "p1-sym9", "sheet": 1, "symbol_class": "tank",
                   "confidence": 1.0}, "pass"))  # no bbox
    with pytest.raises(ValueError) as missing_field:
        export_training_set(tmp_path / "incomplete", PROFILE,
                            tmp_path / "b.jsonl")
    assert "bbox" in str(missing_field.value)
    assert "p1-sym9" in str(missing_field.value)


def test_stray_files_in_the_profile_directory_are_ignored(
        labels_root, tmp_path):
    profile_dir = labels_root / profile_key(PROFILE)
    (profile_dir / "sheet_old.json").write_text("not a store file",
                                                encoding="utf-8")
    (profile_dir / "sheet_1.json.tmp").write_text("{}", encoding="utf-8")
    assert len(_export(labels_root, tmp_path / "set.jsonl")) == 5


def test_chinese_drawing_content_round_trips_deterministically(tmp_path):
    # the corpus is Chinese oil-company drawings (ADR-0001): multibyte
    # profile names quote into store paths, and tag strings export as
    # readable UTF-8, byte-identical on re-export
    profile = {"name": "中石化-管道仪表", "version": "乙"}
    store = LabelStore(tmp_path / "labels")
    store.record(profile, 1, make_example(
        "text",
        {"id": "p1-txt1", "sheet": 1, "string": "泵-101",
         "text_class": "equipment_tag",
         "bbox": [0.0, 0.0, 30.0, 10.0], "confidence": 0.5},
        "edit", {"string": "泵-101A"}))

    out = tmp_path / "set.jsonl"
    export_training_set(tmp_path / "labels", profile, out)
    raw = out.read_text(encoding="utf-8")
    assert "泵-101A" in raw  # not \\u-escaped
    [example] = [json.loads(line) for line in raw.splitlines()]
    assert example["profile"] == profile
    assert example["label"]["string"] == "泵-101A"
    assert example["detection"]["string"] == "泵-101"

    again = tmp_path / "again.jsonl"
    export_training_set(tmp_path / "labels", profile, again)
    assert again.read_bytes() == out.read_bytes()


def test_export_stamps_a_source_when_given(labels_root, tmp_path):
    # the labels store is per-run; a source lets per-profile sets from
    # different Documents merge without ambiguous (sheet, id) keys
    out = tmp_path / "set.jsonl"
    export_training_set(labels_root, PROFILE, out, source="2401-run-3")
    examples = [json.loads(line) for line in
                out.read_text(encoding="utf-8").splitlines()]
    assert all(e["source"] == "2401-run-3" for e in examples)

    # the field is present either way, keeping the line schema stable
    export_training_set(labels_root, PROFILE, out)
    examples = [json.loads(line) for line in
                out.read_text(encoding="utf-8").splitlines()]
    assert all(e["source"] is None for e in examples)


def test_a_profile_absent_from_the_store_is_refused_naming_what_exists(
        labels_root, tmp_path):
    with pytest.raises(ValueError) as refusal:
        export_training_set(labels_root, {"name": "acme p&id/rev A",
                                          "version": "999"},
                            tmp_path / "set.jsonl")
    message = str(refusal.value)
    assert "999" in message
    # the refusal names the store's actual partitions, so a typo'd
    # version is a visible mistake instead of an empty training set
    assert profile_key(PROFILE) in message
    assert profile_key(OTHER) in message
    assert not (tmp_path / "set.jsonl").exists()


def test_cli_exports_to_a_sibling_training_directory_by_default(
        labels_root, capsys):
    main([str(labels_root), "--name", PROFILE["name"],
          "--version", PROFILE["version"]])
    # default lands beside the labels, with the run artifacts outside
    # git (ADR-0001), named by the same quoted profile key as the store
    out = (labels_root.parent / "training"
           / f"{profile_key(PROFILE)}.jsonl")
    assert out.is_file()
    assert len(out.read_text(encoding="utf-8").splitlines()) == 5
    assert str(out) in capsys.readouterr().out


def test_cli_honours_an_explicit_out_path_and_source(
        labels_root, tmp_path):
    out = tmp_path / "elsewhere" / "set.jsonl"
    main([str(labels_root), "--name", OTHER["name"],
          "--version", OTHER["version"], "--out", str(out),
          "--source", "run-7"])
    lines = [json.loads(line) for line in
             out.read_text(encoding="utf-8").splitlines()]
    assert [line["source"] for line in lines] == ["run-7"]
