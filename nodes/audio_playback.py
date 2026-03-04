"""Audio playback sink node — plays TTS audio via sounddevice.

Plays received audio samples through the default output device and
calls optional callbacks for orchestrator signaling:
- on_start(item): called before playback begins
- on_complete(item): called after playback finishes
"""
from __future__ import annotations

import logging

import numpy as np
import sounddevice as sd

from sigflow.node import sink_node
from sigflow.types import Port, TimeSeries1D

log = logging.getLogger(__name__)


@sink_node(
    name="audio_playback",
    inputs=[Port("audio", TimeSeries1D)],
    category="output",
)
def audio_playback(item, *, state, config):
    samples = item.data
    if not isinstance(samples, np.ndarray) or len(samples) == 0:
        return

    sr = item.metadata.get("sample_rate", 24000)
    duration_s = len(samples) / sr
    log.info("playing audio: %.1fs @ %dHz", duration_s, sr)

    on_start = state.get("on_start")
    if on_start is not None:
        on_start(item)

    sd.play(samples, samplerate=sr)
    sd.wait()

    log.info("audio playback complete")

    on_complete = state.get("on_complete")
    if on_complete is not None:
        on_complete(item)
