"""Intake-seam tests: load_document() on programmatically generated PDF
fixtures — a scanned Document is accepted and flows through digitize();
a vector Document is refused with a clear message and no partial output."""

from dataclasses import replace

import pytest

from pidgraph.intake import IntakeError, VectorPdfRefusal, load_document
from pidgraph.model import SheetAnnotations
from pidgraph.pipeline import digitize

from conftest import LINES, SHEET_H, SHEET_W, SYMBOLS, TEXTS, _render
from pdf_fixtures import write_inherited_resources_pdf, write_pdf

ANNOTATIONS = SheetAnnotations(symbols=SYMBOLS, lines=LINES, texts=TEXTS)
RASTER = _render(SHEET_W, SHEET_H, ANNOTATIONS)


def test_scanned_pdf_sheets_enumerated_and_rasterized(tmp_path):
    pdf = write_pdf(tmp_path / "unit-4100.pdf",
                    [("scanned", SHEET_W, SHEET_H, RASTER),
                     ("scanned", SHEET_W, SHEET_H, RASTER)])

    document = load_document(pdf)

    assert document.name == "unit-4100.pdf"
    assert [sheet.number for sheet in document.sheets] == [1, 2]
    for sheet in document.sheets:
        assert (sheet.width, sheet.height) == (SHEET_W, SHEET_H)
        assert sheet.raster == RASTER
        assert sheet.annotations is None  # a real scan carries no ground truth


def test_scanned_pdf_flows_through_digitize_to_records_and_dexpi(
        tmp_path, synthetic_profile):
    pdf = write_pdf(tmp_path / "unit-4100.pdf",
                    [("scanned", SHEET_W, SHEET_H, RASTER),
                     ("scanned", SHEET_W, SHEET_H, RASTER)])
    document = load_document(pdf)
    # Stub components read ground truth, so the test attaches the known
    # annotations to the intake Sheets; the rasters are the PDF's own.
    document = replace(document, sheets=tuple(
        replace(sheet, annotations=ANNOTATIONS) for sheet in document.sheets))

    artifacts = digitize(document, synthetic_profile,
                         out_dir=tmp_path / "out")

    # Sheet identity is preserved: records and DEXPI JSON reference Sheets
    # by identifier.
    assert [record["sheet"] for record in artifacts.detection_records] == [1, 2]
    equipment = artifacts.plant_model["conceptualModel"]["Equipment"]
    assert {item["sheet"] for item in equipment} == {1, 2}
    assert (tmp_path / "out" / "plant_model_dexpi.json").exists()


def test_resources_inherited_from_pages_node_are_resolved(tmp_path):
    pdf = write_inherited_resources_pdf(tmp_path / "inherited.pdf",
                                        SHEET_W, SHEET_H, RASTER)

    document = load_document(pdf)

    assert len(document.sheets) == 1
    assert document.sheets[0].raster == RASTER


def test_vector_pdf_refused_with_clear_message(tmp_path):
    pdf = write_pdf(tmp_path / "unit-4100-cad.pdf", [("vector",)])

    with pytest.raises(VectorPdfRefusal) as excinfo:
        load_document(pdf)

    message = str(excinfo.value)
    assert "unit-4100-cad.pdf" in message          # names the Document
    assert "vector" in message                     # states why
    assert "deterministic vector" in message       # says what to do instead


def test_mixed_document_refused_naming_offending_sheets(tmp_path):
    pdf = write_pdf(tmp_path / "mixed.pdf",
                    [("scanned", SHEET_W, SHEET_H, RASTER),
                     ("vector",),
                     ("vector",)])

    with pytest.raises(VectorPdfRefusal) as excinfo:
        load_document(pdf)

    message = str(excinfo.value)
    assert "Sheets 2, 3" in message


def test_refused_run_produces_no_partial_output(tmp_path, synthetic_profile):
    pdf = write_pdf(tmp_path / "cad.pdf", [("vector",)])
    out_dir = tmp_path / "out"

    with pytest.raises(VectorPdfRefusal):
        digitize(load_document(pdf), synthetic_profile, out_dir=out_dir)

    assert not out_dir.exists()


def test_sheet_without_scanned_image_is_an_intake_error(tmp_path):
    pdf = write_pdf(tmp_path / "blank.pdf",
                    [("scanned", SHEET_W, SHEET_H, RASTER), ("blank",)])

    with pytest.raises(IntakeError) as excinfo:
        load_document(pdf)

    assert "Sheet 2" in str(excinfo.value)


def test_undecodable_image_encoding_is_an_intake_error(tmp_path):
    pdf = write_pdf(tmp_path / "jpeg-scan.pdf",
                    [("raw_image", 8, 4, b"\xff\xd8notarealjpeg",
                      "/DCTDecode")])

    with pytest.raises(IntakeError) as excinfo:
        load_document(pdf)

    message = str(excinfo.value)
    assert "Sheet 1" in message
    assert "DCTDecode" in message
