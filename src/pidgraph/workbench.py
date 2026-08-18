"""Review Workbench (ticket 10): the local web application where a
reviewer sees the digitized P&ID overlaid on the original Sheet and gives
each detection one verdict — pass, reject, or edit (supplying the
correction). Every verdict persists as a labeled example keyed to
Convention Profile and Sheet, so reviewing is simultaneously
training-data creation.

The Workbench reads run artifacts only — detections/sheet_N.json and
sheets/sheet_N.png — and has no path to invoke extraction: nothing from
the engine (pipeline, seams, batch) is imported here.
"""

from __future__ import annotations

import json
import re
from html import escape
from pathlib import Path

from flask import Flask, abort, render_template_string, request, send_file

from .labels import LabelStore, make_example

_INDEX_HTML = """<!doctype html>
<html>
<head><meta charset="utf-8"><title>Review Workbench</title></head>
<body>
<h1>Review Workbench</h1>
<ul>
{% for number in numbers %}
  <li><a href="/sheet/{{ number }}">Sheet {{ number }}</a></li>
{% endfor %}
</ul>
</body>
</html>
"""

_SHEET_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Sheet {{ number }} — Review Workbench</title>
<style>
  body { font-family: sans-serif; margin: 1rem; }
  .stage { position: relative; display: inline-block; }
  .stage img { display: block; max-width: 100%; }
  .stage svg { position: absolute; inset: 0; width: 100%; height: 100%; }
  rect.symbol { fill: none; stroke: #d33; stroke-width: 1.5; }
  polyline.line { fill: none; stroke: #36c; stroke-width: 1.5; }
  rect.text { fill: none; stroke: #2a2; stroke-width: 1; }
  [data-id] { pointer-events: all; cursor: pointer; }
  .selected { stroke-width: 3; }
  [data-verdict="pass"] { opacity: 0.35; }
  [data-verdict="reject"] { opacity: 0.35; stroke-dasharray: 4 3; }
  [data-verdict="edit"] { stroke-dasharray: 1 2; }
  .panel { margin-bottom: 0.5rem; }
  .panel button { margin-right: 0.25rem; }
</style>
</head>
<body>
<h1>Sheet {{ number }}</h1>
<div class="panel">
  <button data-action="pass" disabled>Pass</button>
  <button data-action="reject" disabled>Reject</button>
  <button data-action="edit" disabled>Edit</button>
  <span id="selection">click a detection</span>
  <span id="status"></span>
</div>
<div class="stage">
  <img src="/sheet/{{ number }}/raster.png"
       alt="Sheet {{ number }} original raster">
  {{ overlay | safe }}
</div>
<script>
const VERDICTS_URL = "/sheet/{{ number }}/verdicts";
const buttons = document.querySelectorAll("button[data-action]");
const info = document.getElementById("selection");
const status = document.getElementById("status");
let selected = null;

document.querySelector(".stage svg").addEventListener("click", (event) => {
  const el = event.target.closest("[data-id]");
  if (!el) return;
  if (selected) selected.classList.remove("selected");
  selected = el;
  el.classList.add("selected");
  info.textContent = el.querySelector("title").textContent;
  buttons.forEach((b) => { b.disabled = false; });
});

function promptCorrection(el) {
  if (el.classList.contains("text")) {
    const value = window.prompt("Corrected tag text:");
    return value ? {string: value} : null;
  }
  if (el.classList.contains("symbol")) {
    const value = window.prompt("Corrected bbox x0,y0,x1,y1:");
    return value ? {bbox: value.split(",").map(Number)} : null;
  }
  const value = window.prompt("Corrected polyline as x,y x,y ...:");
  if (!value) return null;
  return {polyline: value.trim().split(/\\s+/)
                    .map((p) => p.split(",").map(Number))};
}

async function give(action) {
  if (!selected) return;
  const payload = {detection_id: selected.getAttribute("data-id"),
                   verdict: action};
  if (action === "edit") {
    const correction = promptCorrection(selected);
    if (!correction) return;
    payload.correction = correction;
  }
  status.textContent = "saving…";
  const response = await fetch(VERDICTS_URL, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload),
  });
  if (response.ok) {
    selected.setAttribute("data-verdict", action);
    status.textContent = "saved";
  } else {
    status.textContent = "refused: " + await response.text();
  }
}

buttons.forEach((b) => b.addEventListener(
  "click", () => give(b.getAttribute("data-action"))));
</script>
</body>
</html>
"""


def _fmt(value: float) -> str:
    return format(value, "g")


def _svg_title(*parts: object) -> str:
    return "<title>" + escape(" ".join(str(p) for p in parts)) + "</title>"


def _overlay_svg(record: dict, examples: dict) -> str:
    """The detections of one Sheet as SVG in original Sheet coordinates —
    symbols, line runs and texts each their own element kind, every one
    naming its detection id and carrying its stored verdict, if any."""

    def attrs(detection: dict) -> str:
        out = f' data-id="{escape(detection["id"])}"'
        example = examples.get(detection["id"])
        if example is not None:
            out += f' data-verdict="{escape(example["verdict"])}"'
        return out

    def bbox_rect(css_class: str, detection: dict, title: str) -> str:
        x0, y0, x1, y1 = detection["bbox"]
        return (f'<rect class="{css_class}"{attrs(detection)}'
                f' x="{_fmt(x0)}" y="{_fmt(y0)}"'
                f' width="{_fmt(x1 - x0)}" height="{_fmt(y1 - y0)}">'
                + title + "</rect>")

    width, height = record["normalization"]["original_size"]
    parts = [f'<svg viewBox="0 0 {_fmt(width)} {_fmt(height)}">']
    for sym in record["symbols"]:
        parts.append(bbox_rect(
            "symbol", sym,
            _svg_title(sym["symbol_class"], sym["id"],
                       "conf", sym["confidence"])))
    for line in record["lines"]:
        points = " ".join(f"{_fmt(x)},{_fmt(y)}"
                          for x, y in line["polyline"])
        parts.append(
            f'<polyline class="line"{attrs(line)}'
            f' points="{points}">'
            + _svg_title(line["line_class"], line["id"],
                         "conf", line["confidence"])
            + "</polyline>")
    for text in record["texts"]:
        parts.append(bbox_rect(
            "text", text,
            _svg_title(text["text_class"], repr(text["string"]),
                       text["id"], "conf", text["confidence"])))
    parts.append("</svg>")
    return "".join(parts)


def _sheet_record(run_dir: Path, number: int) -> dict | None:
    path = run_dir / "detections" / f"sheet_{number}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


_KINDS = (("symbol", "symbols"), ("line", "lines"), ("text", "texts"))


def _find_detection(record: dict,
                    detection_id: object) -> tuple[str, dict] | None:
    for kind, field in _KINDS:
        for detection in record[field]:
            if detection["id"] == detection_id:
                return kind, detection
    return None


def _sheet_numbers(run_dir: Path) -> list[int]:
    pattern = re.compile(r"sheet_(\d+)\.json")
    return sorted(
        int(match.group(1))
        for path in (run_dir / "detections").glob("sheet_*.json")
        if (match := pattern.fullmatch(path.name)))


def create_app(run_dir: Path | str,
               labels_dir: Path | str | None = None) -> Flask:
    """The Workbench over one run's artifacts. Verdicts persist under
    labels_dir — by default labels/ inside the run directory, keeping
    labeled examples outside git alongside the artifacts (ADR-0001).
    Labeled examples are keyed by Convention Profile and Sheet number, so
    an overridden labels_dir must not be shared across runs of different
    Documents: their Sheet numbers would collide."""
    run_dir = Path(run_dir)
    store = LabelStore(run_dir / "labels" if labels_dir is None
                       else labels_dir)
    app = Flask(__name__)

    def load_record(number: int) -> dict:
        record = _sheet_record(run_dir, number)
        if record is None:
            abort(404, f"no run artifacts for Sheet {number}")
        return record

    @app.get("/")
    def index():
        return render_template_string(
            _INDEX_HTML, numbers=_sheet_numbers(run_dir))

    @app.get("/sheet/<int:number>")
    def sheet_html(number: int):
        record = load_record(number)
        examples = store.sheet_labels(record["profile"],
                                      number)["examples"]
        return render_template_string(
            _SHEET_HTML, number=number,
            overlay=_overlay_svg(record, examples))

    @app.get("/sheet/<int:number>/raster.png")
    def sheet_raster(number: int):
        path = run_dir / "sheets" / f"sheet_{number}.png"
        if not path.is_file():
            return f"no raster artifact for Sheet {number}", 404
        return send_file(path, mimetype="image/png")

    @app.get("/sheet/<int:number>/verdicts")
    def sheet_verdicts(number: int):
        record = load_record(number)
        return store.sheet_labels(record["profile"], number)

    @app.post("/sheet/<int:number>/verdicts")
    def post_verdict(number: int):
        record = load_record(number)
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return "expected a JSON object", 400
        found = _find_detection(record, payload.get("detection_id"))
        if found is None:
            return (f"Sheet {number} has no detection"
                    f" {payload.get('detection_id')!r}", 400)
        kind, detection = found
        try:
            example = make_example(kind, detection,
                                   payload.get("verdict"),
                                   payload.get("correction"))
        except ValueError as error:
            return str(error), 400
        return store.record(record["profile"], number, example)

    return app


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m pidgraph.workbench",
        description="Serve the Review Workbench over one run's artifacts.")
    parser.add_argument("run_dir", type=Path,
                        help="run directory holding the artifacts")
    parser.add_argument("--labels-dir", type=Path, default=None,
                        help="where verdicts persist"
                             " (default: <run_dir>/labels)")
    parser.add_argument("--port", type=int, default=5088)
    args = parser.parse_args(argv)
    # local-only, like every part of pidgraph that touches drawing
    # content (ADR-0001)
    create_app(args.run_dir, args.labels_dir).run(host="127.0.0.1",
                                                  port=args.port)


if __name__ == "__main__":
    main()
