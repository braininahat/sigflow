"""Elicitation results sink node.

Receives phoneme recognition / scoring results from the phoneme_recognizer
node and forwards them to the ElicitationService orchestrator via callback.
"""
from __future__ import annotations

import logging

from sigflow.node import sink_node
from sigflow.types import Event, Port

log = logging.getLogger(__name__)


@sink_node(
    name="elicitation_results",
    inputs=[Port("phonemes", Event)],
    category="elicitation",
)
def elicit_results(item, *, state, config):
    result_data = item.data
    if not isinstance(result_data, dict):
        log.warning("elicit_results: unexpected data type: %s", type(result_data))
        return

    log.info(
        "elicit result: word=%s score=%.0f%% expected=[%s] recognized=[%s]",
        result_data.get("word", "?"),
        result_data.get("score", 0) * 100,
        " ".join(result_data.get("expected", [])),
        " ".join(result_data.get("phonemes", [])),
    )

    on_result = state.get("on_result")
    if on_result is not None:
        on_result(result_data)
