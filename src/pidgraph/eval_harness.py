"""Eval harness with pinned component gates (ticket 15).

Runs a configured pipeline — whatever SymbolDetector / TextRecognizer
implementations `PipelineConfig` selects, stubs included — against a
labeled eval set (the label factory's output, ticket 13) and reports the
three gate scores: symbol F1, tag exact-match, connectivity F1. Two
configurations score side by side in one comparison report, the
mechanism by which per-company fine-tuning is judged against Legend
Dictionary nearest-neighbor classification and the OCR engine is chosen
empirically.

The harness is a separate concern from the test suite and never runs in
CI; runs are fully local and touch no network. Only the metric math and
the harness mechanics are unit-tested (offline, deterministic); gate
runs against real models happen on demand.

The gates are pinned here and versioned with the harness: a change to a
threshold or to the metric definitions is a new HARNESS_VERSION, so two
reports compare only when their versions agree. Scores without ground
truth stay None and fail their gate — an empty eval set never
greenlights a model.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Mapping, Sequence

from .assemble import SheetAssembly, build_plant_graph
from .labels import LabelStore, supervised_label
from .model import (
    Bbox,
    ConventionProfile,
    LineAnnotation,
    Sheet,
    SheetAnnotations,
    SymbolAnnotation,
    TextAnnotation,
)
from .pipeline import detections_from_record, digitize_sheet
from .pngio import decode_gray_png
from .profile import load_profile
from .seams import PipelineConfig, build_components

HARNESS_VERSION = "1"

MATCH_IOU = 0.5   # pinned with the harness: how much a predicted box
                  # must overlap a truth box to be reporting it


@dataclass(frozen=True)
class Gate:
    metric: str
    threshold: float


# Provisional thresholds (spec), refined when the real corpus lands.
GATES = (
    Gate("symbol_f1", 0.90),
    Gate("tag_exact_match", 0.98),
    Gate("connectivity_f1", 0.70),
)


# --------------------------------------------------------------------------
# Metric math: matching predicted to ground-truth boxes/strings/connections

def bbox_iou(a: Bbox, b: Bbox) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    inter_w = min(ax1, bx1) - max(ax0, bx0)
    inter_h = min(ay1, by1) - max(ay0, by0)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    intersection = inter_w * inter_h
    union = ((ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0)
             - intersection)
    return intersection / union if union > 0 else 0.0


Boxed = tuple[Bbox, "str | None"]  # a box with its class; None matches any


def match_boxes(predicted: Sequence[Boxed], truth: Sequence[Boxed],
                iou_threshold: float) -> list[tuple[int, int]]:
    """Greedy one-to-one matching of predicted to ground-truth boxes:
    candidate pairs need compatible classes and IoU at or above the
    threshold, and the highest IoU pairs match first (ties broken by
    index, so the matching is deterministic). Returns (predicted index,
    truth index) pairs."""
    candidates = []
    for pi, (pred_box, pred_class) in enumerate(predicted):
        for ti, (truth_box, truth_class) in enumerate(truth):
            if (pred_class is not None and truth_class is not None
                    and pred_class != truth_class):
                continue
            iou = bbox_iou(pred_box, truth_box)
            if iou >= iou_threshold:
                candidates.append((-iou, pi, ti))
    candidates.sort()
    taken_predicted: set[int] = set()
    taken_truth: set[int] = set()
    matches = []
    for _, pi, ti in candidates:
        if pi in taken_predicted or ti in taken_truth:
            continue
        matches.append((pi, ti))
        taken_predicted.add(pi)
        taken_truth.add(ti)
    return matches


def prf(tp: int, fp: int, fn: int) -> dict:
    """Precision / recall / F1 from match counts. With no evidence at
    all every score is None — undefined, never a silent perfect — and a
    gate on a None score fails closed."""
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    if tp + fp + fn == 0:
        f1 = None
    else:
        f1 = 2 * tp / (2 * tp + fp + fn)
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


Tagged = tuple[Bbox, str]  # a text box with its string


def score_tags(predicted: Sequence[Tagged], truth: Sequence[Tagged],
               iou_threshold: float) -> dict:
    """Tag exact-match: each ground-truth tag is read correctly only if
    a predicted text box overlaps it (best IoU, one-to-one, class-blind)
    and carries exactly its string. Unread tags count against the rate;
    extra predicted text does not — precision lives with the review
    workflow, the gate is about reading the plant's tags right."""
    matches = match_boxes([(box, None) for box, _ in predicted],
                          [(box, None) for box, _ in truth],
                          iou_threshold)
    exact = sum(predicted[pi][1] == truth[ti][1] for pi, ti in matches)
    return {"truth": len(truth), "exact": exact,
            "exact_match": exact / len(truth) if truth else None}


def _pair(link: Mapping) -> tuple[str, str]:
    low, high = sorted((link["source"], link["target"]))
    return low, high


def score_links(predicted: Sequence[Mapping],
                truth: Sequence[Mapping]) -> dict:
    """Connectivity F1 over links as a multiset of unordered terminal
    pairs — the s2_pml edge shape both sides speak (the label factory
    emits ground truth with build_plant_graph's edge multiplicity, so
    this compares like for like). Direction is reported beside the gate,
    never inside it: a link predicted with an honest "unknown" is still
    the right connection."""
    predicted_pairs = Counter(_pair(link) for link in predicted)
    truth_pairs = Counter(_pair(link) for link in truth)
    tp = sum((predicted_pairs & truth_pairs).values())
    scores = prf(tp,
                 fp=sum((predicted_pairs - truth_pairs).values()),
                 fn=sum((truth_pairs - predicted_pairs).values()))

    known = [link for link in truth
             if link["attributes"]["direction"] == "known"]
    pools: dict[tuple[str, str], list[Mapping]] = {}
    for link in predicted:
        pools.setdefault(_pair(link), []).append(link)
    agreeing = conflicting = 0
    for link in known:
        # each predicted copy vouches for one truth link, so parallel
        # truth links consume distinct copies — one directed prediction
        # never satisfies a whole bundle
        copies = pools.get(_pair(link), [])
        agree = next(
            (i for i, p in enumerate(copies)
             if p["attributes"]["direction"] == "known"
             and (p["source"], p["target"])
             == (link["source"], link["target"])), None)
        if agree is not None:
            copies.pop(agree)
            agreeing += 1
            continue
        asserted = next(
            (i for i, p in enumerate(copies)
             if p["attributes"]["direction"] in ("known", "conflict")),
            None)
        if asserted is not None:
            copies.pop(asserted)
            conflicting += 1
    scores["direction"] = {
        "truth_known": len(known), "agreeing": agreeing,
        "conflicting": conflicting,
        "unasserted": len(known) - agreeing - conflicting}
    return scores


def apply_gates(scores: Mapping[str, "float | None"]) -> list[dict]:
    """Pass/fail per pinned gate. A None score (nothing to score) fails:
    the absence of evidence never opens a gate."""
    results = []
    for gate in GATES:
        score = scores[gate.metric]
        results.append({"metric": gate.metric, "threshold": gate.threshold,
                        "score": score,
                        "passed": score is not None
                        and score >= gate.threshold})
    return results


# --------------------------------------------------------------------------
# The labeled eval set: label-factory output loaded as scorable Sheets

@dataclass(frozen=True)
class EvalSheet:
    """One labeled Sheet: the pipeline input (raster plus ground-truth
    annotations, so stub seams are scorable too) beside the truth the
    predictions are matched against."""
    sheet: Sheet
    symbols: tuple[dict, ...]   # {"id", "symbol_class", "bbox"}
    tags: tuple[dict, ...]      # {"id", "text_class", "string", "bbox"}
    links: tuple[dict, ...]     # s2_pml edges between symbol ids


def _load_eval_sheet(root: Path, store: LabelStore, identity: dict,
                     number: int, tag_classes: set[str]) -> EvalSheet:
    labels = store.sheet_labels(identity, number)
    if labels["profile"] != identity:
        raise ValueError(
            f"labels for Sheet {number} were recorded for profile"
            f" {labels['profile']!r}, not {identity!r} — refusing to"
            " score against them")

    symbols: list[dict] = []
    tags: list[dict] = []
    annotations: dict[str, list] = {"symbols": [], "lines": [], "texts": []}
    for _, stored in sorted(labels["examples"].items()):
        truth = supervised_label(number, stored)
        if truth is None:  # a reject leaves no ground truth behind
            continue
        kind = stored["kind"]
        if kind == "symbol":
            bbox = tuple(truth["bbox"])
            symbols.append({"id": stored["detection"]["id"],
                            "symbol_class": truth["symbol_class"],
                            "bbox": bbox})
            direction = stored["detection"].get("direction")
            annotations["symbols"].append(SymbolAnnotation(
                symbol_class=truth["symbol_class"], bbox=bbox,
                direction=None if direction is None else tuple(direction)))
        elif kind == "text":
            bbox = tuple(truth["bbox"])
            if truth["text_class"] in tag_classes:
                tags.append({"id": stored["detection"]["id"],
                             "text_class": truth["text_class"],
                             "string": truth["string"], "bbox": bbox})
            annotations["texts"].append(TextAnnotation(
                string=truth["string"], text_class=truth["text_class"],
                bbox=bbox))
        else:
            annotations["lines"].append(LineAnnotation(
                polyline=tuple(tuple(p) for p in truth["polyline"]),
                line_class=truth["line_class"]))

    png_path = root / "sheets" / f"sheet_{number}.png"
    if not png_path.is_file():
        raise ValueError(
            f"Sheet {number} carries no raster: the eval set is missing"
            f" {png_path} and real implementations read pixels")
    width, height, pixels = decode_gray_png(png_path.read_bytes())

    connectivity_path = root / "connectivity" / f"sheet_{number}.json"
    if not connectivity_path.is_file():
        raise ValueError(
            f"Sheet {number} carries no connectivity ground truth: the"
            f" eval set is missing {connectivity_path}")
    connectivity = json.loads(connectivity_path.read_text(encoding="utf-8"))
    if connectivity["profile"] != identity:
        raise ValueError(
            f"connectivity for Sheet {number} was recorded for profile"
            f" {connectivity['profile']!r}, not {identity!r} — refusing"
            " to score against it")

    return EvalSheet(
        sheet=Sheet(number=number, width=width, height=height,
                    raster=pixels,
                    annotations=SheetAnnotations(
                        symbols=tuple(annotations["symbols"]),
                        lines=tuple(annotations["lines"]),
                        texts=tuple(annotations["texts"]))),
        symbols=tuple(symbols),
        tags=tuple(tags),
        links=tuple(connectivity["links"]))


def load_eval_set(root: Path | str, profile: ConventionProfile,
                  sheets: Sequence[int] | None = None) -> list[EvalSheet]:
    """Load a labeled eval set — a label factory output directory
    (labels/, connectivity/, sheets/) — for one Convention Profile.
    Only the tags the profile's grammar names enter the tag gate; free
    text stays out of it. Fails closed: an empty set, or a labeled
    Sheet missing its raster or its connectivity file, is refused
    rather than silently narrowing what the gates cover."""
    root = Path(root)
    store = LabelStore(root / "labels")
    identity = profile.identity_record()
    available = store.sheets(identity)
    if not available:
        raise ValueError(
            f"no labeled Sheets for profile {profile.name!r} version"
            f" {profile.version!r} under {root / 'labels'}; the store"
            f" holds {store.profiles() or 'no profiles'}")
    if sheets is None:
        numbers = available
    else:
        numbers = sorted(set(sheets))
        unknown = sorted(set(numbers) - set(available))
        if unknown:
            raise ValueError(
                f"Sheet(s) {unknown} are not in the eval set; it labels"
                f" Sheets {available}")

    return [_load_eval_sheet(root, store, identity, number,
                             set(profile.tag_grammar))
            for number in numbers]


# --------------------------------------------------------------------------
# Scoring a configured pipeline against the eval set

def _predicted_links(assembly: SheetAssembly,
                     truth_id_by_prediction: Mapping[str, str]) -> list[dict]:
    """The pipeline's plant-graph edges translated into ground-truth
    symbol ids through the symbol matching, so connectivity compares
    like for like. A terminal whose detection matched no truth symbol
    keeps a sentinel id no ground truth uses — its links stay in the
    count as false positives, never silently dropped."""
    truth_id_by_tag: dict[str, str] = {}
    for terminal in assembly.terminals:
        tag = terminal.node_tag()
        if tag not in truth_id_by_tag:  # first wins under duplicate tags
            truth_id_by_tag[tag] = truth_id_by_prediction.get(
                terminal.detection.id, f"?{terminal.detection.id}")
    edges = build_plant_graph([assembly])["edges"]
    return [{"source": truth_id_by_tag.get(edge["source"],
                                           f"?{edge['source']}"),
             "target": truth_id_by_tag.get(edge["target"],
                                           f"?{edge['target']}"),
             "attributes": {
                 "direction": edge["attributes"]["direction"]}}
            for edge in edges]


def _score_sheet(eval_sheet: EvalSheet, profile: ConventionProfile,
                 detector, recognizer) -> dict:
    record, assembly = digitize_sheet(eval_sheet.sheet, profile,
                                      detector, recognizer)
    predicted_symbols, _, predicted_texts = detections_from_record(record)

    matches = match_boxes(
        [(s.bbox, s.symbol_class) for s in predicted_symbols],
        [(s["bbox"], s["symbol_class"]) for s in eval_sheet.symbols],
        MATCH_IOU)
    symbol = prf(tp=len(matches),
                 fp=len(predicted_symbols) - len(matches),
                 fn=len(eval_sheet.symbols) - len(matches))

    tag = score_tags([(t.bbox, t.string) for t in predicted_texts],
                     [(t["bbox"], t["string"]) for t in eval_sheet.tags],
                     MATCH_IOU)

    truth_id_by_prediction = {
        predicted_symbols[pi].id: eval_sheet.symbols[ti]["id"]
        for pi, ti in matches}
    connectivity = score_links(
        _predicted_links(assembly, truth_id_by_prediction),
        eval_sheet.links)

    return {"sheet": eval_sheet.sheet.number, "symbol": symbol,
            "tag": tag, "connectivity": connectivity}


def _missed_sheet(eval_sheet: EvalSheet) -> dict:
    """A Sheet the pipeline could not digitize scores as all misses —
    its ground truth stays in the denominators, so a failure can never
    inflate a gate."""
    return {"sheet": eval_sheet.sheet.number,
            "symbol": prf(tp=0, fp=0, fn=len(eval_sheet.symbols)),
            "tag": score_tags([], [(t["bbox"], t["string"])
                                   for t in eval_sheet.tags], MATCH_IOU),
            "connectivity": score_links([], eval_sheet.links)}


def _aggregate(per_sheet: Sequence[dict]) -> dict:
    symbol = prf(tp=sum(s["symbol"]["tp"] for s in per_sheet),
                 fp=sum(s["symbol"]["fp"] for s in per_sheet),
                 fn=sum(s["symbol"]["fn"] for s in per_sheet))
    truth = sum(s["tag"]["truth"] for s in per_sheet)
    exact = sum(s["tag"]["exact"] for s in per_sheet)
    tag = {"truth": truth, "exact": exact,
           "exact_match": exact / truth if truth else None}
    connectivity = prf(
        tp=sum(s["connectivity"]["tp"] for s in per_sheet),
        fp=sum(s["connectivity"]["fp"] for s in per_sheet),
        fn=sum(s["connectivity"]["fn"] for s in per_sheet))
    connectivity["direction"] = {
        key: sum(s["connectivity"]["direction"][key] for s in per_sheet)
        for key in ("truth_known", "agreeing", "conflicting", "unasserted")}
    return {"symbol": symbol, "tag": tag, "connectivity": connectivity}


def evaluate(eval_sheets: Sequence[EvalSheet], profile: ConventionProfile,
             config: PipelineConfig | None = None) -> dict:
    """Score one configured pipeline against the eval set: gate scores
    with pass/fail per pinned gate, per-Sheet detail, and honest
    coverage — a Sheet that fails to digitize is listed and its truth
    counted as missed."""
    config = config or PipelineConfig()
    detector, recognizer, _ = build_components(config)  # the GraphStore
    # seam is not scored: gates end at the assembled topology

    per_sheet = []
    failures = []
    for eval_sheet in eval_sheets:
        try:
            per_sheet.append(_score_sheet(eval_sheet, profile,
                                          detector, recognizer))
        except Exception as error:
            failures.append({
                "sheet": eval_sheet.sheet.number,
                "reason": f"{type(error).__name__}: {error}"})
            per_sheet.append(_missed_sheet(eval_sheet))

    metrics = _aggregate(per_sheet)
    gates = apply_gates({
        "symbol_f1": metrics["symbol"]["f1"],
        "tag_exact_match": metrics["tag"]["exact_match"],
        "connectivity_f1": metrics["connectivity"]["f1"]})
    return {
        "harness_version": HARNESS_VERSION,
        "profile": profile.identity_record(),
        "config": asdict(config),
        "coverage": {"sheets": len(eval_sheets),
                     "scored": len(eval_sheets) - len(failures),
                     "failed": len(failures)},
        "failures": failures,
        "per_sheet": per_sheet,
        "metrics": metrics,
        "gates": gates,
        "passed": all(gate["passed"] for gate in gates),
    }


def compare(eval_sheets: Sequence[EvalSheet], profile: ConventionProfile,
            configs: Mapping[str, PipelineConfig]) -> dict:
    """Two or more configurations against the same eval set in one
    report — fine-tuning beside nearest-neighbor classification,
    candidate OCR engines beside each other."""
    if len(configs) < 2:
        raise ValueError(
            f"a comparison puts at least two configurations side by"
            f" side; got {sorted(configs) or 'none'}")
    return {
        "harness_version": HARNESS_VERSION,
        "profile": profile.identity_record(),
        "configurations": [
            {"name": name, **evaluate(eval_sheets, profile, config)}
            for name, config in configs.items()],
    }


# --------------------------------------------------------------------------
# CLI — on-demand, fully local; never wired into CI

def _parse_config(spec: str) -> tuple[str, PipelineConfig]:
    name, _, rest = spec.partition(":")
    if not name:
        raise ValueError(
            f"configuration {spec!r} is not"
            f" NAME[:seam=implementation,...]")
    selections = {}
    for part in rest.split(",") if rest else []:
        key, separator, value = part.partition("=")
        if not separator or not value:
            raise ValueError(
                f"configuration {spec!r} is not"
                f" NAME[:seam=implementation,...]")
        selections[key] = value
    try:
        return name, PipelineConfig(**selections)
    except TypeError:
        raise ValueError(
            f"configuration {spec!r} selects an unknown seam; the seams"
            f" are {[f.name for f in fields(PipelineConfig)]}") from None


def _print_report(name: str, report: dict) -> None:
    print(f"{name}: {'PASS' if report['passed'] else 'FAIL'}"
          f" ({report['coverage']['scored']}/"
          f"{report['coverage']['sheets']} Sheets scored)")
    for gate in report["gates"]:
        score = ("unscored" if gate["score"] is None
                 else f"{gate['score']:.3f}")
        print(f"  {gate['metric']}: {score}"
              f" (gate >= {gate['threshold']})"
              f" {'PASS' if gate['passed'] else 'FAIL'}")
    for failure in report["failures"]:
        print(f"  Sheet {failure['sheet']} failed: {failure['reason']}")


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m pidgraph.eval_harness",
        description="Score configured pipelines against a labeled eval"
                    " set: symbol F1, tag exact-match, connectivity F1,"
                    " pass/fail per pinned gate. On-demand and fully"
                    " local; never run in CI.",
        epilog="Exit status: 0 when every configuration passes its"
               " gates, 1 when any gate fails.")
    parser.add_argument("eval_root", type=Path,
                        help="a label factory output directory"
                             " (labels/, connectivity/, sheets/)")
    parser.add_argument("profile_bundle", type=Path,
                        help="the Convention Profile bundle directory")
    parser.add_argument("--sheets", type=int, nargs="+", default=None,
                        help="score only these Sheet numbers")
    parser.add_argument("--config", action="append", default=None,
                        metavar="NAME[:SEAM=IMPL,...]",
                        help='a configuration to score, e.g.'
                             ' "nn:symbol_detector=legend-nn"; repeat'
                             ' for a side-by-side comparison (default:'
                             ' the stub defaults)')
    parser.add_argument("--out", type=Path, default=None,
                        help="write the JSON report here")
    args = parser.parse_args(argv)

    configs: dict[str, PipelineConfig] = {}
    for spec in args.config or ["default"]:
        try:
            name, config = _parse_config(spec)
        except ValueError as error:
            parser.error(str(error))
        if name in configs:
            parser.error(f"configuration name {name!r} appears twice")
        configs[name] = config

    profile = load_profile(args.profile_bundle)
    eval_sheets = load_eval_set(args.eval_root, profile, args.sheets)

    if len(configs) == 1:
        ((name, config),) = configs.items()
        report = evaluate(eval_sheets, profile, config)
        named = [(name, report)]
    else:
        report = compare(eval_sheets, profile, configs)
        named = [(c["name"], c) for c in report["configurations"]]

    print(f"eval harness v{HARNESS_VERSION}: profile {profile.name!r}"
          f" version {profile.version!r},"
          f" Sheets {[s.sheet.number for s in eval_sheets]}")
    for name, config_report in named:
        _print_report(name, config_report)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.out.with_name(args.out.name + ".tmp")
        tmp.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        os.replace(tmp, args.out)
        print(f"report -> {args.out}")

    if not all(config_report["passed"] for _, config_report in named):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
