"""Review Workbench (ticket 10): the local web application where a
reviewer sees the digitized P&ID overlaid on the original Sheet and gives
each detection one verdict — pass, reject, or edit (supplying the
correction). Every verdict persists as a labeled example keyed to
Convention Profile and Sheet, so reviewing is simultaneously
training-data creation.

Review flow (ticket 11): a reviewer's minutes go to the elements most
likely to be wrong, so each Sheet's undecided detections queue lowest
confidence first — scoped per Sheet and, on request, per detection kind —
and giving a verdict advances the queue. The Document-level index shows
each Sheet's review state (untouched, in progress, reviewed), derived on
every request from the persisted verdicts, so a 400-Sheet review splits
across days and restarts without losing its place.

Product KPI (ticket 19): the Workbench logs its own activity — every
Sheet opened, every verdict saved, at the server's clock — beside the
verdicts, and shows the operator the KPI (corrections per 100 symbols,
reviewer minutes per accepted Sheet) recomputed from the persisted
verdicts and activity, with the measurement basis stated.

The Workbench reads run artifacts only — detections/sheet_N.json and
sheets/sheet_N.png — and has no path to invoke extraction: nothing from
the engine (pipeline, seams, batch) is imported here.
"""

from __future__ import annotations

from html import escape
from pathlib import Path

from flask import Flask, abort, render_template_string, request, send_file

from .artifacts import (
    KINDS,
    RecordSummaries,
    iter_detections,
    load_record,
    load_sheets,
    review_state,
)
from .kpi import (
    DEFAULT_IDLE_MINUTES,
    ActivityLog,
    Clock,
    display_kpi,
    kpi_report,
    run_kpi,
)
from .labels import LabelStore, make_example

_INDEX_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8"><title>Review Workbench</title>
<style>
  body { font-family: sans-serif; margin: 1rem; }
  [data-state="reviewed"] { color: #282; }
  [data-state="in progress"] { color: #a60; }
  [data-state="unreadable"] { color: #c00; }
  [data-state] a { color: inherit; }
  #kpi { border: 1px solid #ccc; padding: 0.5rem 1rem; margin-top: 1rem;
         max-width: 60em; }
  #kpi dt { font-weight: bold; margin-top: 0.5rem; }
  #kpi .basis { color: #555; font-size: 0.9em; }
</style>
</head>
<body>
<h1>Review Workbench</h1>
<ul>
{% for row in sheets %}
  <li data-state="{{ row.state }}"><a href="/sheet/{{ row.sheet }}">Sheet
 {{ row.sheet }}</a> — {{ row.state }},
 {{ row.decided }}/{{ row.total }} reviewed</li>
{% endfor %}
</ul>
<section id="kpi">
<h2>Product KPI</h2>
<dl>
  <dt>Corrections per 100 symbols</dt>
  <dd><span data-kpi="corrections_per_100_symbols">{{
      display_kpi(kpi.corrections_per_100_symbols) }}</span> —
      {{ kpi.by_kind.symbol.reviewed }} symbols reviewed,
      {{ kpi.by_kind.symbol.corrections }} corrections (reject + edit)
      <div class="basis">{{ basis.corrections_per_100_symbols }}</div></dd>
  <dt>Reviewer minutes per accepted Sheet</dt>
  <dd><span data-kpi="reviewer_minutes_per_accepted_sheet">{{
      display_kpi(kpi.reviewer_minutes_per_accepted_sheet) }}</span> —
      {{ kpi.sheets_summary.accepted }} accepted of {{
      kpi.sheets_summary.total }} Sheets, {{
      kpi.sheets_summary.accepted_timed }} timed by the Workbench log
      <div class="basis">{{ basis.reviewer_minutes_per_accepted_sheet }}
      </div></dd>
</dl>
<p><a href="/kpi">full KPI report (JSON)</a> — recomputed from the
persisted verdicts and activity on every request</p>
</section>
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
<div class="panel">
  <label>queue scope
    <select id="scope">
      <option value="">all kinds</option>
      <option value="symbol">symbols</option>
      <option value="line">lines</option>
      <option value="text">texts</option>
    </select>
  </label>
  <button id="next">Next in queue</button>
  <span id="queue-status"></span>
</div>
<div class="stage">
  <img src="/sheet/{{ number }}/raster.png"
       alt="Sheet {{ number }} original raster">
  {{ overlay | safe }}
</div>
<script>
const VERDICTS_URL = "/sheet/{{ number }}/verdicts";
const QUEUE_URL = "/sheet/{{ number }}/queue";
const buttons = document.querySelectorAll("button[data-action]");
const info = document.getElementById("selection");
const status = document.getElementById("status");
const scope = document.getElementById("scope");
const queueStatus = document.getElementById("queue-status");
let selected = null;
let advanceToken = 0;

function select(el) {
  if (selected) selected.classList.remove("selected");
  selected = el;
  if (!el) {
    info.textContent = "click a detection";
    buttons.forEach((b) => { b.disabled = true; });
    return;
  }
  el.classList.add("selected");
  el.scrollIntoView({block: "nearest", inline: "nearest"});
  info.textContent = el.querySelector("title").textContent;
  buttons.forEach((b) => { b.disabled = false; });
}

async function advance() {
  const token = ++advanceToken;
  const url = scope.value
    ? QUEUE_URL + "?kind=" + scope.value : QUEUE_URL;
  let data;
  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error("HTTP " + response.status);
    data = await response.json();
  } catch (error) {
    if (token === advanceToken) {
      queueStatus.textContent = "queue unavailable";
    }
    return;
  }
  if (token !== advanceToken) return;  // superseded by a newer advance
  queueStatus.textContent =
    data.remaining + " of " + data.total + " awaiting review";
  if (!data.queue.length) {
    select(null);
    info.textContent = "queue empty — scope reviewed";
    return;
  }
  select(document.querySelector(
    '[data-id="' + CSS.escape(data.queue[0].id) + '"]'));
}

document.getElementById("next").addEventListener("click", advance);
scope.addEventListener("change", advance);

document.querySelector(".stage svg").addEventListener("click", (event) => {
  const el = event.target.closest("[data-id]");
  if (el) select(el);
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
  const el = selected;  // the save targets this element, even if the
                        // reviewer clicks elsewhere while it is in flight
  const payload = {detection_id: el.getAttribute("data-id"),
                   verdict: action};
  if (action === "edit") {
    const correction = promptCorrection(el);
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
    el.setAttribute("data-verdict", action);
    status.textContent = "saved";
    if (selected === el) {
      await advance();  // the verdict decided it: the queue moves on
    }
  } else {
    status.textContent = "refused: " + await response.text();
  }
}

buttons.forEach((b) => b.addEventListener(
  "click", () => give(b.getAttribute("data-action"))));

// the queue greets the reviewer: status and lowest-confidence-first
// selection are presented on load, not after a first click
advance();
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


def _find_detection(record: dict,
                    detection_id: object) -> tuple[str, dict] | None:
    for kind, detection in iter_detections(record):
        if detection["id"] == detection_id:
            return kind, detection
    return None


def _queue_payload(sheet: int, record: dict, examples: dict,
                   kind: str | None) -> dict:
    """The Sheet's undecided detections, lowest confidence first, so
    reviewer minutes go to the elements most likely to be wrong. Ties
    keep record order (the sort is stable), so the queue is deterministic
    across requests and restarts."""
    scoped = [(k, d) for k, d in iter_detections(record)
              if kind is None or k == kind]
    pending = [{"id": d["id"], "kind": k, "confidence": d["confidence"]}
               for k, d in scoped if d["id"] not in examples]
    pending.sort(key=lambda entry: entry["confidence"])
    return {"sheet": sheet, "kind": kind,
            "total": len(scoped), "remaining": len(pending),
            "queue": pending}


def _review_state(sheet: int, ids: frozenset[str],
                  examples: dict) -> dict:
    """One Sheet's review state from verdict coverage. Only verdicts on
    detections this Sheet actually contains count — labels persisted
    against another run's artifacts never inflate coverage."""
    decided = len(ids & examples.keys())
    return {"sheet": sheet, "state": review_state(decided, len(ids)),
            "decided": decided, "total": len(ids)}


def create_app(run_dir: Path | str,
               labels_dir: Path | str | None = None,
               clock: Clock | None = None,
               idle_minutes: float = DEFAULT_IDLE_MINUTES) -> Flask:
    """The Workbench over one run's artifacts. Verdicts persist under
    labels_dir — by default labels/ inside the run directory, keeping
    labeled examples outside git alongside the artifacts (ADR-0001) — and
    the activity log lives beside them. Labeled examples are keyed by
    Convention Profile and Sheet number, so an overridden labels_dir must
    not be shared across runs of different Documents: their Sheet numbers
    would collide. clock stamps activity (default: the system's UTC
    clock); idle_minutes is the KPI's break threshold."""
    run_dir = Path(run_dir)
    store = LabelStore(run_dir / "labels" if labels_dir is None
                       else labels_dir)
    activity = ActivityLog(store.root, clock)
    # Detection records never change while a run is under review, so the
    # Document-level views share one summary cache keyed on each record
    # file's stat. Labels and activity are never cached: review state and
    # KPI are recomputed from the persisted store on every request, never
    # from session memory.
    summaries = RecordSummaries(run_dir)
    app = Flask(__name__)

    def record_for(number: int) -> dict:
        record = load_record(run_dir, number)
        if record is None:
            abort(404, f"no run artifacts for Sheet {number}")
        return record

    # A Sheet's identity is the number its artifact file is served
    # under — the same number in the URL — so labels, queue, progress and
    # KPI all key one way even if a record's internal "sheet" field
    # disagrees with its filename.
    def load_examples(record: dict, number: int) -> dict:
        return store.sheet_labels(record["profile"], number)["examples"]

    def review_states() -> list[dict]:
        rows = []
        for number, summary in load_sheets(run_dir, summaries):
            if summary is None:
                # a run killed mid-write leaves a truncated record; one
                # bad Sheet must not take down the whole Document's view
                rows.append({"sheet": number, "state": "unreadable",
                             "decided": 0, "total": 0})
                continue
            profile, ids = summary
            examples = store.sheet_labels(profile, number)["examples"]
            rows.append(_review_state(number, ids, examples))
        return rows

    def kpi() -> dict:
        return kpi_report([run_kpi(run_dir, store.root, idle_minutes,
                                   summaries)])

    @app.get("/")
    def index():
        report = kpi()
        return render_template_string(_INDEX_HTML, sheets=review_states(),
                                      kpi=report["overall"],
                                      basis=report["basis"],
                                      display_kpi=display_kpi)

    @app.get("/progress")
    def progress():
        return {"sheets": review_states()}

    @app.get("/kpi")
    def kpi_json():
        return kpi()

    @app.get("/sheet/<int:number>/queue")
    def sheet_queue(number: int):
        record = record_for(number)
        kind = request.args.get("kind")
        if kind is not None and kind not in dict(KINDS):
            return (f"kind must be one of {sorted(dict(KINDS))},"
                    f" got {kind!r}", 400)
        return _queue_payload(number, record,
                              load_examples(record, number), kind)

    @app.get("/sheet/<int:number>")
    def sheet_html(number: int):
        record = record_for(number)
        # opening a Sheet for review is activity the KPI times by
        activity.record("open", number)
        return render_template_string(
            _SHEET_HTML, number=number,
            overlay=_overlay_svg(record, load_examples(record, number)))

    @app.get("/sheet/<int:number>/raster.png")
    def sheet_raster(number: int):
        path = run_dir / "sheets" / f"sheet_{number}.png"
        if not path.is_file():
            return f"no raster artifact for Sheet {number}", 404
        return send_file(path, mimetype="image/png")

    @app.get("/sheet/<int:number>/verdicts")
    def sheet_verdicts(number: int):
        record = record_for(number)
        return store.sheet_labels(record["profile"], number)

    @app.post("/sheet/<int:number>/verdicts")
    def post_verdict(number: int):
        record = record_for(number)
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
        store.record(record["profile"], number, example)
        # a saved verdict is activity: identifier and verdict only, the
        # detection and any correction live in the verdict store
        activity.record("verdict", number, detection_id=detection["id"],
                        verdict=example["verdict"])
        return example

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
    parser.add_argument("--idle-minutes", type=float,
                        default=DEFAULT_IDLE_MINUTES,
                        help="KPI: an interval between Workbench events"
                             " longer than this is a break, not review"
                             f" time (default {DEFAULT_IDLE_MINUTES:g})")
    parser.add_argument("--port", type=int, default=5088)
    args = parser.parse_args(argv)
    # local-only, like every part of pidgraph that touches drawing
    # content (ADR-0001)
    create_app(args.run_dir, args.labels_dir,
               idle_minutes=args.idle_minutes).run(host="127.0.0.1",
                                                   port=args.port)


if __name__ == "__main__":
    main()
