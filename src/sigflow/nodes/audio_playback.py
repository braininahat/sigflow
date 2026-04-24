"""Audio playback sink node — plays TTS audio via sounddevice.

Plays received audio samples through the default output device and
calls optional callbacks for orchestrator signaling:
- on_start(item): called before playback begins
- on_complete(item): called after playback finishes
"""
from __future__ import annotations

import logging

import numpy as np

from sigflow.node import sink_node
from sigflow.types import Port, TimeSeries1D

log = logging.getLogger(__name__)


@sink_node(
    name="audio_playback",
    inputs=[Port("audio", TimeSeries1D)],
    category="output",
)
def audio_playback(item, *, state, config):
    import sounddevice as sd

    samples = item.data
    if not isinstance(samples, np.ndarray) or len(samples) == 0:
        return

    sr = int(item.metadata.get("sample_rate", 24000))
    duration_s = len(samples) / sr
    log.info("playing audio: %.1fs @ %dHz", duration_s, sr)

    on_start = state.get("on_start")
    if on_start is not None:
        on_start(item)

    # kokoro-onnx outputs 24 kHz but many host audio APIs (esp. ALSA/PulseAudio
    # default devices) only advertise 44.1/48 kHz and reject anything else with
    # "Invalid sample rate [PaErrorCode -9997]".  Try native on first call; on
    # mismatch switch to the device's default rate and cache the decision so
    # subsequent clips skip the failing sd.play() call.
    target_sr = state.get("_playback_sr", sr)
    if target_sr != sr:
        from scipy.signal import resample_poly
        g = np.gcd(sr, target_sr)
        samples = resample_poly(samples, target_sr // g, sr // g).astype(np.float32)

    try:
        sd.play(samples, samplerate=target_sr)
        sd.wait()
    except sd.PortAudioError as err:
        if "Invalid sample rate" not in str(err) or target_sr != sr:
            raise
        info = sd.query_devices(kind="output")
        device_sr = int(info["default_samplerate"])
        state["_playback_sr"] = device_sr
        log.info("audio device rejected %d Hz; resampling to %d Hz", sr, device_sr)
        from scipy.signal import resample_poly
        g = np.gcd(sr, device_sr)
        resampled = resample_poly(samples, device_sr // g, sr // g).astype(np.float32)
        sd.play(resampled, samplerate=device_sr)
        sd.wait()

    log.info("audio playback complete")

    on_complete = state.get("on_complete")
    if on_complete is not None:
        on_complete(item)
