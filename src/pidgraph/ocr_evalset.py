"""Synthetic OCR eval sets at the operating scale (ticket 18).

Candidate OCR engines are judged by the eval harness on tag exact-match,
and the harness needs labeled eval sets whose text the engines can
fairly be asked to read at the Raster Path's operating scale — the
normalized frame, TARGET_LONG_SIDE px on the long side — before the
corpus arrives. This module renders such sets: tag Sheets of Latin tags
(equipment, instrument, line number, off-page connector), Chinese
equipment labels and mixed Chinese + Latin service labels, standing
horizontal, vertical and rotated among distractor line art (bubbles
around instrument tags, pipes beside line numbers, equipment outlines),
in a clean set and a degraded one (ticket 14's transforms). Ground
truth is laid out the way the label factory lays it out — labels/,
connectivity/, sheets/ — beside the Convention Profile bundle it is
labeled for, so

    python -m pidgraph.eval_harness <out>/clean <out>/profile \\
        --config stub --config rapidocr:text_recognizer=rapidocr ...

scores any selectable TextRecognizer on it. Rendering is PyMuPDF (the
labelfactory extra): Helvetica for Latin, its built-in CJK font for
Chinese — fully local and nothing drawing-derived, so the sets carry no
sensitive content (they still live under data/ like every generated
artifact). Seeded, so a set reproduces exactly from its seed.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .degrade import (
    Blur,
    Noise,
    Raster,
    Skew,
    clip_bbox,
    degrade_bbox,
    degrade_raster,
    transform_config,
)
from .labels import LabelStore, make_example
from .normalize import TARGET_LONG_SIDE
from .pngio import encode_gray_png

PROFILE = {"name": "ocr-evalset", "version": "1"}
COMPONENT = "ocr_evalset:render"

# The tag grammar the sets are labeled against: P&ID tag shapes for the
# Latin classes, Chinese-only equipment labels, and mixed Chinese + Latin
# service labels on one line (the pinned mixed-script requirement).
TAG_GRAMMAR = {
    "equipment_tag": r"[A-Z]{1,2}-\d{3}[A-Z]?",
    "instrument_tag": r"[A-Z]{2,3}-\d{3,4}",
    "line_number": r"\d{2,3}-[A-Z]{2}-\d{3,5}(?:-[A-Z]\d{2}[A-Z])?",
    "opc_label": r"[A-Z]{2}\d{2}-\d{4}",
    "equipment_label": r"[一-鿿]{2,6}",
    "service_label": r"[一-鿿]{1,4} [A-Z]{1,3}",
}
LEGEND = {"instrument_bubble": {"role": "ProcessInstrumentationFunction",
                                "equipment_type": "instrument"}}
LINE_SEMANTICS = {"pipe": "PipingNetworkSegment"}
LINE_STYLES = {"solid": "pipe"}

# What real scans do to text: a little skew, a little blur, sensor noise.
DEGRADATION = (Skew(angle_deg=0.8), Blur(radius=1), Noise(sigma=16.0))

SHEET_WIDTH, SHEET_HEIGHT = TARGET_LONG_SIDE, 240   # the operating scale
_COLUMNS, _ROWS = 4, 3                               # one tag per cell
_MARGIN = 6.0
_FONT_SIZES = (9, 10, 11, 12, 13)
_ROTATIONS = ((0, 6), (90, 3), (270, 1))             # (degrees ccw, weight)

_LATIN_FONT = "helv"
_CJK_FONT = "china-s"

_EQUIPMENT_WORDS = ("原料罐", "冷却器", "压缩机", "反应器", "分离器", "换热器",
                    "回流罐", "缓冲罐", "排污罐", "放空总管", "蒸汽发生器",
                    "循环水泵")
_SERVICES = (("冷却水", "CW"), ("蒸汽", "LS"), ("仪表风", "IA"), ("氮气", "N"),
             ("燃料气", "FG"), ("工艺水", "PW"), ("循环水", "CWS"),
             ("凝结水", "CD"))
_INSTRUMENTS = ("PI", "TI", "FI", "LI", "PT", "TT", "FT", "LT", "FIC",
                "LIC", "TIC", "PIC", "PSV", "FE")
_FLUIDS = ("GA", "PL", "CW", "LS", "HS", "IA", "PG", "CA")
_EQUIPMENT_LETTERS = "TPKEVCDRX"


@dataclass(frozen=True)
class _Item:
    """One tag to render: its runs (text, font) laid end to end along the
    text direction, its class, and the geometry chosen for it."""
    runs: tuple[tuple[str, str], ...]
    text_class: str
    fontsize: int
    rotate: int

    @property
    def string(self) -> str:
        return "".join(text for text, _ in self.runs)


@dataclass(frozen=True)
class RenderedSheet:
    number: int
    width: int
    height: int
    pixels: bytes
    tags: tuple[dict, ...]   # {"string", "text_class", "bbox", "evidence"}


def _digits(rng: random.Random, count: int) -> str:
    return "".join(rng.choice("0123456789") for _ in range(count))


def _letters(rng: random.Random, count: int) -> str:
    return "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                   for _ in range(count))


def _tag_string(rng: random.Random, text_class: str) -> tuple[tuple[str, str],
                                                              ...]:
    """A grammar-valid string for the class, as rendering runs."""
    if text_class == "equipment_tag":
        letters = "".join(rng.choice(_EQUIPMENT_LETTERS)
                          for _ in range(rng.choice((1, 1, 2))))
        suffix = rng.choice(("", "", "A", "B"))
        return ((f"{letters}-{_digits(rng, 3)}{suffix}", _LATIN_FONT),)
    if text_class == "instrument_tag":
        return ((f"{rng.choice(_INSTRUMENTS)}-{_digits(rng, rng.choice((3, 4)))}",
                 _LATIN_FONT),)
    if text_class == "line_number":
        size = rng.choice(("50", "80", "100", "150", "200", "250"))
        number = _digits(rng, rng.choice((3, 5)))
        spec = (f"-{_letters(rng, 1)}{_digits(rng, 2)}{_letters(rng, 1)}"
                if rng.random() < 0.5 else "")
        return ((f"{size}-{rng.choice(_FLUIDS)}-{number}{spec}",
                 _LATIN_FONT),)
    if text_class == "opc_label":
        return ((f"{_letters(rng, 2)}{_digits(rng, 2)}-{_digits(rng, 4)}",
                 _LATIN_FONT),)
    if text_class == "equipment_label":
        return ((rng.choice(_EQUIPMENT_WORDS), _CJK_FONT),)
    if text_class == "service_label":
        service, code = rng.choice(_SERVICES)
        return ((service, _CJK_FONT), (" ", _LATIN_FONT), (code, _LATIN_FONT))
    raise ValueError(f"no generator for text class {text_class!r}")


def _choose_rotation(rng: random.Random) -> int:
    [rotation] = rng.choices([r for r, _ in _ROTATIONS],
                             weights=[w for _, w in _ROTATIONS])
    return rotation


def _pymupdf():
    try:
        import pymupdf  # type: ignore[import-not-found]
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "OCR eval sets are rendered with PyMuPDF; install the"
            " labelfactory extra (pip install 'pidgraph[labelfactory]')"
        ) from error
    return pymupdf


def _advance(pymupdf, item: _Item) -> float:
    return sum(pymupdf.get_text_length(text, fontname=font,
                                       fontsize=item.fontsize)
               for text, font in item.runs)


def render_tag_sheet(number: int, seed: int | str,
                     width: int = SHEET_WIDTH,
                     height: int = SHEET_HEIGHT) -> RenderedSheet:
    """One tag Sheet: a grid of cells, one tag per cell cycling through
    the classes, each with its own size and orientation, plus distractor
    line art; the raster and the tags' exact boxes read back from the
    rendered page."""
    pymupdf = _pymupdf()
    rng = random.Random(f"ocr-evalset/{seed}/{number}")
    document = pymupdf.open()
    page = document.new_page(width=width, height=height)
    shape = page.new_shape()
    classes = list(TAG_GRAMMAR)
    rng.shuffle(classes)
    cell_w, cell_h = width / _COLUMNS, height / _ROWS
    placed: list[tuple[_Item, list[tuple[float, float]]]] = []

    for index in range(_COLUMNS * _ROWS):
        text_class = classes[index % len(classes)]
        cell_x = (index % _COLUMNS) * cell_w
        cell_y = (index // _COLUMNS) * cell_h
        rotate = _choose_rotation(rng)
        fontsize = rng.choice(_FONT_SIZES)
        runs = _tag_string(rng, text_class)
        item = _Item(runs, text_class, fontsize, rotate)
        # shrink until the text fits its cell along its direction; a tag
        # too long for a vertical cell even at the smallest size lies
        # horizontal instead (and is shrunk again against that room)
        def fitted(item: _Item) -> _Item:
            room = ((cell_w if item.rotate == 0 else cell_h)
                    - 2 * _MARGIN - item.fontsize)
            while _advance(pymupdf, item) > room and item.fontsize > 8:
                item = _Item(runs, text_class, item.fontsize - 1,
                             item.rotate)
            return item

        item = fitted(item)
        if item.rotate != 0 and _advance(pymupdf, item) > (
                cell_h - 2 * _MARGIN - item.fontsize):
            item = fitted(_Item(runs, text_class, fontsize, 0))
        rotate = item.rotate
        advance = _advance(pymupdf, item)
        glyph = item.fontsize  # generous ascender + descender allowance
        if rotate == 0:
            x = cell_x + _MARGIN + rng.uniform(0, max(0.0, cell_w - 2 * _MARGIN
                                                    - advance))
            y = cell_y + _MARGIN + glyph + rng.uniform(
                0, max(0.0, cell_h - 2 * _MARGIN - glyph - 4))
            step = (1.0, 0.0)
        elif rotate == 90:   # text runs upward
            x = cell_x + _MARGIN + glyph + rng.uniform(
                0, max(0.0, cell_w - 2 * _MARGIN - glyph - 4))
            y = cell_y + cell_h - _MARGIN - rng.uniform(
                0, max(0.0, cell_h - 2 * _MARGIN - advance))
            step = (0.0, -1.0)
        else:                # 270: text runs downward
            x = cell_x + _MARGIN + 4 + rng.uniform(
                0, max(0.0, cell_w - 2 * _MARGIN - glyph - 4))
            y = cell_y + _MARGIN + rng.uniform(
                0, max(0.0, cell_h - 2 * _MARGIN - advance))
            step = (0.0, 1.0)

        origins = []
        cursor = (x, y)
        for text, font in item.runs:
            page.insert_text(pymupdf.Point(*cursor), text,
                             fontsize=item.fontsize, fontname=font,
                             rotate=rotate)
            origins.append(cursor)
            length = pymupdf.get_text_length(text, fontname=font,
                                             fontsize=item.fontsize)
            cursor = (cursor[0] + step[0] * length,
                      cursor[1] + step[1] * length)
        placed.append((item, origins))

        # distractor ink the way a drawing surrounds its tags
        mid = (x + step[0] * advance / 2 - (glyph * 0.35 if rotate == 90
                                            else -glyph * 0.35 if rotate == 270
                                            else 0.0),
               y + step[1] * advance / 2 - (glyph * 0.35 if rotate == 0
                                            else 0.0))
        if text_class == "instrument_tag":
            radius = max(advance, glyph) / 2 + 3
            shape.draw_circle(pymupdf.Point(*mid), radius)
        elif text_class == "line_number":
            offset = glyph * 0.9
            if rotate == 0:
                shape.draw_line(pymupdf.Point(cell_x + 2, y + offset),
                                pymupdf.Point(cell_x + cell_w - 2, y + offset))
            else:
                side = x + (offset if rotate == 90 else -offset)
                shape.draw_line(pymupdf.Point(side, cell_y + 2),
                                pymupdf.Point(side, cell_y + cell_h - 2))
        elif text_class == "equipment_tag" and rotate == 0:
            top = y + glyph * 0.6
            if top + 12 < cell_y + cell_h - 2:
                shape.draw_rect(pymupdf.Rect(x, top, x + advance,
                                             min(top + 14, cell_y + cell_h - 2)))
    shape.finish(color=(0, 0, 0), width=0.8)
    shape.commit()

    # exact boxes read back from the page: every run's span, found by
    # its baseline origin, unioned per item
    spans = [span for block in page.get_text("dict")["blocks"]
             for line in block.get("lines", []) for span in line["spans"]]
    tags = []
    for item, origins in placed:
        boxes = []
        for origin in origins:
            nearest = min(spans, key=lambda s: (s["origin"][0] - origin[0]) ** 2
                          + (s["origin"][1] - origin[1]) ** 2)
            if not nearest["text"].strip():
                continue
            boxes.append(nearest["bbox"])
        if not boxes:
            raise RuntimeError(f"Sheet {number}: rendered {item.string!r}"
                               " but read no span back")
        bbox = (min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes))
        tags.append({
            "string": item.string, "text_class": item.text_class,
            "bbox": [round(c, 2) for c in bbox],
            "evidence": f"rendered {item.text_class} {item.string!r} at"
                        f" {item.fontsize}pt, rotated {item.rotate}°"})

    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(1, 1),
                             colorspace=pymupdf.csGRAY)
    rendered = RenderedSheet(number=number, width=pixmap.width,
                             height=pixmap.height,
                             pixels=bytes(pixmap.samples),
                             tags=tuple(tags))
    document.close()
    return rendered


def degrade_sheet(sheet: RenderedSheet, seed: int | str) -> RenderedSheet:
    """The Sheet through the degradation transforms, its boxes mapped
    and clipped along; a tag pushed off the Sheet is dropped."""
    raster = degrade_raster(Raster(sheet.width, sheet.height, sheet.pixels),
                            DEGRADATION, seed)
    tags = []
    for tag in sheet.tags:
        bbox = clip_bbox(degrade_bbox(DEGRADATION, tag["bbox"],
                                      sheet.width, sheet.height),
                         raster.width, raster.height)
        if bbox is None:
            continue
        tags.append({**tag, "bbox": [round(c, 2) for c in bbox],
                     "evidence": tag["evidence"] + ", degraded"})
    return RenderedSheet(sheet.number, raster.width, raster.height,
                         raster.pixels, tuple(tags))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    os.replace(tmp, path)


def write_profile_bundle(bundle: Path) -> Path:
    _write_json(bundle / "profile.json", PROFILE)
    _write_json(bundle / "legend.json", LEGEND)
    _write_json(bundle / "tag_grammar.json", TAG_GRAMMAR)
    _write_json(bundle / "line_semantics.json", LINE_SEMANTICS)
    _write_json(bundle / "line_styles.json", LINE_STYLES)
    return bundle


def write_eval_set(root: Path, sheets: Sequence[RenderedSheet]) -> int:
    """The Sheets as a label-factory-shaped eval set; returns the tag
    count."""
    store = LabelStore(root / "labels")
    tags = 0
    for sheet in sheets:
        examples = [
            make_example("text", {
                "id": f"p{sheet.number}-text{i}", "sheet": sheet.number,
                "string": tag["string"], "text_class": tag["text_class"],
                "bbox": list(tag["bbox"]), "confidence": 1.0,
                "provenance": {"component": COMPONENT,
                               "evidence": tag["evidence"]}}, "pass")
            for i, tag in enumerate(sheet.tags)]
        store.record_many(PROFILE, sheet.number, examples, replace=True)
        tags += len(examples)
        _write_json(root / "connectivity" / f"sheet_{sheet.number}.json",
                    {"profile": PROFILE, "sheet": sheet.number, "links": []})
        png = root / "sheets" / f"sheet_{sheet.number}.png"
        png.parent.mkdir(parents=True, exist_ok=True)
        png.write_bytes(encode_gray_png(sheet.width, sheet.height,
                                        sheet.pixels))
    return tags


def write_ocr_eval_sets(out_dir: Path | str, sheets: int = 12,
                        seed: int = 0) -> dict[str, Path]:
    """The profile bundle, the clean set and the degraded set under
    out_dir, plus a manifest; returns their paths."""
    out = Path(out_dir)
    if sheets < 1:
        raise ValueError(f"an eval set has at least one Sheet, got {sheets}")
    rendered = [render_tag_sheet(number, seed)
                for number in range(1, sheets + 1)]
    degraded = [degrade_sheet(sheet, f"{seed}/{sheet.number}")
                for sheet in rendered]
    paths = {"profile": write_profile_bundle(out / "profile"),
             "clean": out / "clean", "degraded": out / "degraded"}
    counts = {"clean": write_eval_set(paths["clean"], rendered),
              "degraded": write_eval_set(paths["degraded"], degraded)}
    _write_json(out / "manifest.json", {
        "profile": PROFILE, "seed": seed, "sheets": sheets,
        "sheet_size": [SHEET_WIDTH, SHEET_HEIGHT],
        "tags": counts,
        "degradation": [transform_config(t) for t in DEGRADATION],
        "fonts": {"latin": _LATIN_FONT, "cjk": _CJK_FONT}})
    return paths


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m pidgraph.ocr_evalset",
        description="Render synthetic OCR eval sets at the operating"
                    " scale — Latin and Chinese tags, horizontal, vertical"
                    " and rotated, clean and degraded — for the eval"
                    " harness to score candidate TextRecognizers on.")
    parser.add_argument("out_dir", type=Path,
                        help="where to write profile/, clean/, degraded/")
    parser.add_argument("--sheets", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    paths = write_ocr_eval_sets(args.out_dir, args.sheets, args.seed)
    manifest = json.loads((args.out_dir / "manifest.json").read_text(
        encoding="utf-8"))
    print(f"profile {PROFILE['name']}@{PROFILE['version']} ->"
          f" {paths['profile']}")
    for name in ("clean", "degraded"):
        print(f"{name}: {args.sheets} Sheets, {manifest['tags'][name]} tags"
              f" -> {paths[name]}")


if __name__ == "__main__":
    main()
