"""Audio playback sink node — plays TTS audio via sounddevice.

Plays received audio samples through the default output device and
calls optional callbacks for orchestrator signaling:
- on_start(item): called before playback begins
- on_complete(item): called after playback finishes
"""
from __future__ import annotations

import logging
import time

import numpy as np

from sigflow.node import Param, sink_node
from sigflow.types import Port, TimeSeries1D

log = logging.getLogger(__name__)


def _negotiate_playback_sr(sd, source_sr: int) -> tuple[int, dict]:
    """Pick a samplerate the default output device accepts.

    Returns (target_sr, device_info). Tries source_sr first; falls back to the
    device's default rate if the host API rejects it (typical with ALSA /
    PulseAudio default sinks that only advertise 44.1/48 kHz).
    """
    try:
        device_info = sd.query_devices(kind="output")
    except Exception as err:  # pragma: no cover — defensive
        log.warning("sd.query_devices(output) failed: %s", err)
        return source_sr, {}

    candidates = [source_sr, int(device_info.get("default_samplerate", 0))]
    for cand in candidates:
        if cand <= 0:
            continue
        try:
            sd.check_output_settings(samplerate=cand)
            return cand, device_info
        except Exception:
            continue
    log.warning(
        "no supported output samplerate among %s — falling back to source rate %d",
        candidates, source_sr,
    )
    return source_sr, device_info


@sink_node(
    name="audio_playback",
    inputs=[Port("audio", TimeSeries1D)],
    category="output",
    params=[
        Param("min_rms", "float", 1e-4, label="Min RMS warning threshold"),
    ],
)
def audio_playback(item, *, state, config):
    import sounddevice as sd

    samples = item.data
    if not isinstance(samples, np.ndarray) or samples.size == 0:
        log.warning("audio_playback: dropping empty/non-array payload (type=%s)", type(samples).__name__)
        return

    sr = int(item.metadata.get("sample_rate", 24000))

    # Negotiate output samplerate proactively — once per session — so we don't
    # rely on PortAudioError to discover that the device can't take 24 kHz.
    if "_playback_sr" not in state:
        target_sr, device_info = _negotiate_playback_sr(sd, sr)
        state["_playback_sr"] = target_sr
        state["_device_name"] = device_info.get("name", "?")
        state["_device_index"] = device_info.get("index", -1)
        log.info(
            "audio output: device=%r index=%s source_sr=%d target_sr=%d",
            state["_device_name"], state["_device_index"], sr, target_sr,
        )
    target_sr = int(state["_playback_sr"])

    abs_mean = float(np.abs(samples).mean())
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    state["_last_rms"] = rms
    duration_s = len(samples) / sr

    log.info(
        "audio_playback: n=%d dtype=%s sr=%d->%d dur=%.2fs |mean|=%.4f rms=%.4f device=%r",
        samples.size, samples.dtype, sr, target_sr, duration_s, abs_mean, rms,
        state.get("_device_name", "?"),
    )

    min_rms = float(config.get("min_rms", 1e-4))
    if rms < min_rms and not state.get("_low_rms_warned"):
        log.warning(
            "audio_playback: payload rms=%.6f below threshold %.6f — "
            "kokoro voice config may be wrong, or model output is silent",
            rms, min_rms,
        )
        state["_low_rms_warned"] = True

    on_start = state.get("on_start")
    if on_start is not None:
        on_start(item)

    play_samples = samples
    if target_sr != sr:
        from scipy.signal import resample_poly
        g = np.gcd(sr, target_sr)
        play_samples = resample_poly(samples, target_sr // g, sr // g).astype(np.float32)

    t0 = time.perf_counter()
    try:
        sd.play(play_samples, samplerate=target_sr)
        sd.wait()
    except sd.PortAudioError as err:
        # Device may have changed since negotiation (headphone hot-plug, etc.).
        # Re-negotiate and retry once.
        log.warning("audio_playback: PortAudioError %s — re-negotiating", err)
        new_sr, info = _negotiate_playback_sr(sd, sr)
        state["_playback_sr"] = new_sr
        state["_device_name"] = info.get("name", "?")
        state["_device_index"] = info.get("index", -1)
        if new_sr != sr:
            from scipy.signal import resample_poly
            g = np.gcd(sr, new_sr)
            play_samples = resample_poly(samples, new_sr // g, sr // g).astype(np.float32)
        else:
            play_samples = samples
        sd.play(play_samples, samplerate=new_sr)
        sd.wait()
        target_sr = new_sr
    elapsed_ms = (time.perf_counter() - t0) * 1000

    log.info(
        "audio_playback: complete (%d samples @ %d Hz in %.0f ms)",
        play_samples.size, target_sr, elapsed_ms,
    )

    on_complete = state.get("on_complete")
    if on_complete is not None:
        on_complete(item)
