"""Metric computation for the eval harness (ticket 15): matching
predicted to ground-truth boxes/strings/connections and the F1 math,
unit-tested deterministically under the offline invariant."""

import pytest

from pidgraph.eval_harness import (
    GATES,
    apply_gates,
    bbox_iou,
    match_boxes,
    prf,
    score_links,
    score_tags,
)


# --------------------------------------------------------------------------
# IoU

def test_iou_of_identical_boxes_is_one():
    assert bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_of_disjoint_boxes_is_zero():
    assert bbox_iou((0, 0, 10, 10), (20, 0, 30, 10)) == 0.0


def test_iou_of_half_overlapping_boxes():
    # intersection 50, union 150
    assert bbox_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_iou_of_degenerate_boxes_is_zero():
    assert bbox_iou((5, 5, 5, 5), (5, 5, 5, 5)) == 0.0


# --------------------------------------------------------------------------
# Greedy one-to-one matching

def test_matching_requires_the_same_class():
    matches = match_boxes([((0, 0, 10, 10), "tank")],
                          [((0, 0, 10, 10), "gate_valve")], 0.5)
    assert matches == []


def test_matching_requires_the_iou_threshold():
    barely = ((0, 0, 10, 10), "tank")
    shifted = ((8, 0, 18, 10), "tank")  # IoU 2/18 < 0.5
    assert match_boxes([barely], [shifted], 0.5) == []


def test_matching_is_one_to_one():
    truth = [((0, 0, 10, 10), "tank")]
    predicted = [((0, 0, 10, 10), "tank"), ((1, 0, 11, 10), "tank")]
    assert match_boxes(predicted, truth, 0.5) == [(0, 0)]


def test_matching_prefers_the_higher_iou():
    truth = [((0, 0, 10, 10), "tank"), ((30, 0, 40, 10), "tank")]
    predicted = [((1, 0, 11, 10), "tank"),   # good fit for truth 0
                 ((0, 0, 10, 10), "tank")]   # perfect fit for truth 0
    matches = match_boxes(predicted, truth, 0.5)
    assert (1, 0) in matches
    assert (0, 0) not in matches


def test_matching_is_deterministic_under_input_order():
    truth = [((0, 0, 10, 10), "tank"), ((5, 0, 15, 10), "tank")]
    predicted = [((0, 0, 10, 10), "tank"), ((5, 0, 15, 10), "tank")]
    forward = match_boxes(predicted, truth, 0.5)
    backward = match_boxes(list(reversed(predicted)), truth, 0.5)
    assert {(predicted[p][0], truth[t][0]) for p, t in forward} == {
        (list(reversed(predicted))[p][0], truth[t][0])
        for p, t in backward}


def test_a_none_class_matches_any_class():
    matches = match_boxes([((0, 0, 10, 10), None)],
                          [((0, 0, 10, 10), "tank")], 0.5)
    assert matches == [(0, 0)]


# --------------------------------------------------------------------------
# Precision / recall / F1 — fail closed on no evidence

def test_prf_math():
    scores = prf(tp=2, fp=1, fn=1)
    assert scores == {"tp": 2, "fp": 1, "fn": 1,
                      "precision": pytest.approx(2 / 3),
                      "recall": pytest.approx(2 / 3),
                      "f1": pytest.approx(2 / 3)}


def test_prf_without_evidence_is_undefined_not_perfect():
    scores = prf(tp=0, fp=0, fn=0)
    assert (scores["precision"], scores["recall"], scores["f1"]) == (
        None, None, None)


def test_prf_with_only_false_positives():
    scores = prf(tp=0, fp=3, fn=0)
    assert scores["precision"] == 0.0
    assert scores["recall"] is None
    assert scores["f1"] == 0.0


def test_prf_with_only_false_negatives():
    scores = prf(tp=0, fp=0, fn=2)
    assert scores["precision"] is None
    assert scores["recall"] == 0.0
    assert scores["f1"] == 0.0


# --------------------------------------------------------------------------
# Tag exact-match

def test_tag_exact_match_counts_matched_equal_strings():
    truth = [((0, 0, 30, 10), "T-101"), ((0, 20, 30, 30), "V-101")]
    predicted = [((0, 0, 30, 10), "T-101"), ((0, 20, 30, 30), "V-1O1")]
    scores = score_tags(predicted, truth, 0.5)
    assert scores == {"truth": 2, "exact": 1,
                      "exact_match": pytest.approx(0.5)}


def test_an_unread_tag_counts_against_exact_match():
    truth = [((0, 0, 30, 10), "T-101")]
    scores = score_tags([], truth, 0.5)
    assert scores == {"truth": 1, "exact": 0, "exact_match": 0.0}


def test_extra_text_predictions_do_not_change_exact_match():
    truth = [((0, 0, 30, 10), "T-101")]
    predicted = [((0, 0, 30, 10), "T-101"), ((50, 50, 80, 60), "noise")]
    assert score_tags(predicted, truth, 0.5)["exact_match"] == 1.0


def test_tag_exact_match_without_truth_is_undefined():
    assert score_tags([((0, 0, 30, 10), "T-101")], [], 0.5) == {
        "truth": 0, "exact": 0, "exact_match": None}


# --------------------------------------------------------------------------
# Connectivity — multiset of unordered terminal pairs

def _link(source, target, direction="unknown"):
    return {"source": source, "target": target,
            "attributes": {"direction": direction}}


def test_links_match_regardless_of_endpoint_order():
    scores = score_links([_link("b", "a")], [_link("a", "b")])
    assert (scores["tp"], scores["fp"], scores["fn"]) == (1, 0, 0)
    assert scores["f1"] == 1.0


def test_link_multiplicity_is_respected():
    truth = [_link("a", "b"), _link("a", "b"), _link("a", "c")]
    predicted = [_link("a", "b"), _link("c", "d")]
    scores = score_links(predicted, truth)
    assert (scores["tp"], scores["fp"], scores["fn"]) == (1, 1, 2)


def test_direction_stats_over_known_truth_links():
    truth = [_link("a", "b", "known"),   # asserted, agreeing
             _link("c", "d", "known"),   # asserted the wrong way
             _link("e", "f", "known"),   # left unasserted
             _link("g", "h", "known"),   # predicted as a conflict
             _link("i", "j")]            # unknown truth: not counted
    predicted = [_link("a", "b", "known"),
                 _link("d", "c", "known"),
                 _link("e", "f"),
                 _link("g", "h", "conflict"),
                 _link("i", "j")]
    scores = score_links(predicted, truth)
    assert scores["direction"] == {"truth_known": 4, "agreeing": 1,
                                   "conflicting": 2, "unasserted": 1}


def test_a_missing_predicted_link_leaves_direction_unasserted():
    scores = score_links([], [_link("a", "b", "known")])
    assert scores["direction"] == {"truth_known": 1, "agreeing": 0,
                                   "conflicting": 0, "unasserted": 1}


def test_one_directed_prediction_cannot_vouch_for_parallel_links():
    truth = [_link("a", "b", "known"), _link("a", "b", "known")]
    predicted = [_link("a", "b", "known")]
    scores = score_links(predicted, truth)
    assert scores["direction"] == {"truth_known": 2, "agreeing": 1,
                                   "conflicting": 0, "unasserted": 1}


# --------------------------------------------------------------------------
# Pinned gates

def test_the_three_gates_are_pinned():
    assert [(g.metric, g.threshold) for g in GATES] == [
        ("symbol_f1", 0.90), ("tag_exact_match", 0.98),
        ("connectivity_f1", 0.70)]


def test_gates_state_pass_and_fail_per_gate():
    results = apply_gates({"symbol_f1": 0.95, "tag_exact_match": 0.98,
                           "connectivity_f1": 0.42})
    assert [(r["metric"], r["passed"]) for r in results] == [
        ("symbol_f1", True), ("tag_exact_match", True),
        ("connectivity_f1", False)]
    assert results[0] == {"metric": "symbol_f1", "threshold": 0.90,
                          "score": 0.95, "passed": True}


def test_an_unscorable_gate_fails_closed():
    results = apply_gates({"symbol_f1": None, "tag_exact_match": 1.0,
                           "connectivity_f1": 1.0})
    assert results[0]["passed"] is False
    assert results[0]["score"] is None
