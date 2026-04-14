"""Audio to spectrogram process node with rolling accumulation."""
import logging
from collections import deque

import numpy as np

from sigflow.node import process_node, Param
from sigflow.types import Port, AudioSignal, TimeSeries2D

log = logging.getLogger(__name__)


@process_node(
    name="spectrogram",
    inputs=[Port("audio", AudioSignal)],
    outputs=[Port("spectrogram", TimeSeries2D)],
    category="transform",
    params=[
        Param("nperseg", "int", 256, label="FFT Window", min=64, max=4096),
        Param("history_length", "int", 100, label="History Columns", min=10, max=1000),
    ],
)
def spectrogram(item, *, state, config):
    sample_rate = item.metadata.get("sample_rate", 44100)
    nperseg = config["nperseg"]
    history = config["history_length"]

    from scipy import signal as sig
    n = min(nperseg, len(item.data))
    f, t, Sxx = sig.spectrogram(item.data, fs=sample_rate, nperseg=n)

    # Initialize rolling buffer on first call
    if "columns" not in state:
        log.debug("initializing spectrogram buffer: nperseg=%d, history=%d", nperseg, history)
        state["columns"] = deque(maxlen=history)

    # Append new time columns
    for col_idx in range(Sxx.shape[1]):
        state["columns"].append(Sxx[:, col_idx])

    accumulated = np.column_stack(list(state["columns"]))

    return {"spectrogram": item.replace(data=accumulated, port_type=TimeSeries2D)}
