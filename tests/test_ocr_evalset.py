"""Synthetic OCR eval sets (ticket 18): tag Sheets rendered at the
operating scale — Latin and Chinese, horizontal, vertical and rotated,
clean and degraded — laid out the way the label factory lays out ground
truth, so the eval harness scores candidate OCR engines on tag
exact-match. Rendering needs PyMuPDF (the labelfactory extra); no OCR
engine is involved here — the stub pipeline proves the sets are
internally consistent."""

import re

import pytest

pytest.importorskip("pymupdf")

from pidgraph.eval_harness import evaluate, load_eval_set  # noqa: E402
from pidgraph.ocr_evalset import main, write_ocr_eval_sets  # noqa: E402
from pidgraph.profile import load_profile  # noqa: E402
from pidgraph.seams import PipelineConfig  # noqa: E402


def _is_cjk(char):
    return "一" <= char <= "鿿"


@pytest.fixture(scope="module")
def eval_sets(tmp_path_factory):
    out = tmp_path_factory.mktemp("ocr-eval")
    paths = write_ocr_eval_sets(out, sheets=2, seed=7)
    return out, paths


def test_the_sets_load_as_harness_eval_sets_for_their_profile(eval_sets):
    out, paths = eval_sets
    profile = load_profile(paths["profile"])
    assert profile.identity_record() == {"name": "ocr-evalset",
                                         "version": "1"}

    clean = load_eval_set(paths["clean"], profile)
    degraded = load_eval_set(paths["degraded"], profile)
    assert [s.sheet.number for s in clean] == [1, 2]
    assert [s.sheet.number for s in degraded] == [1, 2]
    for eval_sheet in clean + degraded:
        assert max(eval_sheet.sheet.width, eval_sheet.sheet.height) == 400
        assert eval_sheet.sheet.raster is not None
        assert len(eval_sheet.tags) >= 8


def test_the_tags_cover_the_pinned_requirements(eval_sets):
    """Mixed Chinese + Latin, rotated and vertical text: every set has
    Chinese-only labels, mixed Chinese + Latin labels, Latin tags, and
    tags standing vertically — and every tag string is grammar-valid
    for its class, so the stub pipeline can read it back exactly."""
    out, paths = eval_sets
    profile = load_profile(paths["profile"])
    tags = [tag for eval_sheet in load_eval_set(paths["clean"], profile)
            for tag in eval_sheet.tags]

    strings = [tag["string"] for tag in tags]
    assert any(all(_is_cjk(c) for c in s) for s in strings)
    assert any(any(_is_cjk(c) for c in s) and any(c.isascii() for c in s)
               for s in strings)
    assert any(s.isascii() for s in strings)
    vertical = [tag for tag in tags
                if (tag["bbox"][3] - tag["bbox"][1])
                > (tag["bbox"][2] - tag["bbox"][0])]
    assert len(vertical) >= 3
    for tag in tags:
        assert re.fullmatch(profile.tag_grammar[tag["text_class"]],
                            tag["string"]), tag


def test_the_stub_pipeline_reads_every_tag_back(eval_sets):
    """Internal consistency: ground truth, raster and grammar agree, so
    the stub (annotation-reading, noise-seeded, decoder-repaired) scores
    tag exact-match 1.0 on the clean and the degraded set alike. An
    engine's score is then entirely the engine's doing."""
    out, paths = eval_sets
    profile = load_profile(paths["profile"])
    for name in ("clean", "degraded"):
        report = evaluate(load_eval_set(paths[name], profile), profile,
                          PipelineConfig())
        assert report["coverage"]["failed"] == 0, (name, report["failures"])
        assert report["metrics"]["tag"]["exact_match"] == 1.0, name


def test_the_sets_are_reproducible_from_their_seed(tmp_path):
    first = write_ocr_eval_sets(tmp_path / "a", sheets=1, seed=11)
    second = write_ocr_eval_sets(tmp_path / "b", sheets=1, seed=11)
    other = write_ocr_eval_sets(tmp_path / "c", sheets=1, seed=12)

    def png(paths):
        return (paths["clean"] / "sheets" / "sheet_1.png").read_bytes()

    assert png(first) == png(second)
    assert png(first) != png(other)


def test_cli_writes_the_sets_and_reports_counts(tmp_path, capsys):
    main([str(tmp_path / "sets"), "--sheets", "1", "--seed", "2"])

    out = capsys.readouterr().out
    assert "clean" in out and "degraded" in out and "tags" in out
    profile = load_profile(tmp_path / "sets" / "profile")
    assert load_eval_set(tmp_path / "sets" / "clean", profile)
