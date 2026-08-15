"""The three component seams are selected by configuration — swapping an
implementation changes behavior at the digitize() seam without touching
the pipeline."""

import pytest

from pidgraph.model import SymbolDetection
from pidgraph.pipeline import digitize
from pidgraph.seams import SYMBOL_DETECTORS, PipelineConfig, StubSymbolDetector


def test_unknown_component_name_fails_with_available_choices(
        synthetic_document, synthetic_profile):
    config = PipelineConfig(symbol_detector="trained-yolo")

    with pytest.raises(KeyError, match="trained-yolo.*stub"):
        digitize(synthetic_document, synthetic_profile, config=config)


def test_configured_detector_is_the_one_that_runs(synthetic_document,
                                                  synthetic_profile):
    class ArrowBlindDetector(StubSymbolDetector):
        COMPONENT = "symbol_detector:arrow-blind"

        def detect(self, sheet, profile) -> list[SymbolDetection]:
            return [d for d in super().detect(sheet, profile)
                    if d.symbol_class != "flow_arrow"]

    SYMBOL_DETECTORS["arrow-blind"] = ArrowBlindDetector
    try:
        artifacts = digitize(
            synthetic_document, synthetic_profile,
            config=PipelineConfig(symbol_detector="arrow-blind"))
    finally:
        del SYMBOL_DETECTORS["arrow-blind"]

    # No arrow detected -> no direction evidence -> no FLOWS_TO anywhere.
    assert all(e["attributes"]["direction"] == "unknown"
               for e in artifacts.plant_graph["edges"])
    assert "FLOWS_TO" not in "".join(
        l for l in artifacts.cypher_script.splitlines()
        if l.startswith("MATCH"))
