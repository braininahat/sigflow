"""TTS synthesis process node — Kokoro ONNX.

Receives text events, generates speech audio via Kokoro TTS,
and emits audio samples with timing metadata.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from sigflow.node import process_node, Param
from sigflow.paths import resolve_data_path
from sigflow.types import Event, Port, Sample, TimeSeries1D

log = logging.getLogger(__name__)


def _load_kokoro(state, config):
    """Lazy-load Kokoro TTS model into state."""
    model_path = str(resolve_data_path(config.get("model_path", "weights/kokoro-v1.0.onnx")))
    voices_path = str(resolve_data_path(config.get("voices_path", "weights/kokoro-voices-v1.0.bin")))

    missing = []
    if not Path(model_path).exists():
        missing.append(Path(model_path).name)
    if not Path(voices_path).exists():
        missing.append(Path(voices_path).name)
    if missing:
        raise FileNotFoundError(
            f"TTS weights missing: {', '.join(missing)}. "
            "Run: uv run python tools/download_weights.py --tts"
        )

    # Use system espeak-ng — the bundled espeakng_loader has a hardcoded
    # build-time data path that doesn't exist at runtime.
    import ctypes.util
    import os
    if not os.environ.get("PHONEMIZER_ESPEAK_LIBRARY"):
        sys_lib = ctypes.util.find_library("espeak-ng") or ctypes.util.find_library("espeak")
        if sys_lib:
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = sys_lib

    # kokoro-onnx 0.5.0 calls EspeakWrapper.set_data_path() which was
    # removed in phonemizer 3.3.0 (now a class attribute, not a method).
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    if not hasattr(EspeakWrapper, "set_data_path"):
        EspeakWrapper.set_data_path = staticmethod(
            lambda p: setattr(EspeakWrapper, "data_path", p)
        )

    from kokoro_onnx import Kokoro
    state["kokoro"] = Kokoro(model_path, voices_path)
    log.info("TTS loaded: %s", Path(model_path).name)


@process_node(
    name="tts_synthesis",
    inputs=[Port("text", Event)],
    outputs=[Port("audio", TimeSeries1D)],
    category="inference",
    params=[
        Param("model_path", "str", "weights/kokoro-v1.0.onnx", label="Model Path"),
        Param("voices_path", "str", "weights/kokoro-voices-v1.0.bin", label="Voices Path"),
        Param("voice", "str", "af_heart", label="Voice"),
        Param("speed", "float", 1.0, label="Speed", min=0.5, max=2.0),
    ],
)
def tts_synthesis(item, *, state, config):
    if "kokoro" not in state:
        _load_kokoro(state, config)

    text_data = item.data  # {"text": str, ...}
    text = text_data["text"] if isinstance(text_data, dict) else str(text_data)
    voice = config.get("voice", "af_heart")
    speed = config.get("speed", 1.0)

    t0 = time.perf_counter()
    samples, sr = state["kokoro"].create(text, voice=voice, speed=speed, lang="en-us")
    latency_ms = (time.perf_counter() - t0) * 1000

    duration_s = len(samples) / sr
    realtime_factor = duration_s / (latency_ms / 1000) if latency_ms > 0 else 0

    log.info("TTS: %s (%.0fms, %.1fs audio, %.1fx RT)", repr(text[:60]), latency_ms, duration_s, realtime_factor)

    return {"audio": item.replace(
        data=samples,
        port_type=TimeSeries1D,
        metadata={
            **item.metadata,
            "sample_rate": sr,
            "duration_s": duration_s,
            "tts_latency_ms": latency_ms,
            "text": text,
        },
    )}
