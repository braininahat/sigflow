"""Microphone capture source node (sounddevice)."""
import logging

import numpy as np
import sounddevice as sd

from sigflow.node import source_node, Param
from sigflow.types import Port, Sample, AudioSignal

log = logging.getLogger(__name__)

_COMMON_RATES = [8000, 16000, 22050, 44100, 48000, 96000]


def _discover_audio_inputs():
    """Return {name: index} for available audio input devices.

    Filters to the default host API to avoid duplicates from
    ALSA + PulseAudio + JACK all reporting the same hardware.
    """
    try:
        default_api = sd.default.hostapi
        return {d["name"]: i for i, d in enumerate(sd.query_devices())
                if d["max_input_channels"] > 0 and d["hostapi"] == default_api}
    except Exception as e:
        log.warning("audio device discovery failed: %s", e)
        return {}


def _discover_supported_rates():
    """Probe the default input device for supported sample rates."""
    supported = []
    for rate in _COMMON_RATES:
        try:
            sd.check_input_settings(samplerate=float(rate))
            supported.append(str(rate))
        except Exception:
            pass
    return supported if supported else [str(_COMMON_RATES[3])]  # fallback to 44100


_AUDIO_DEVICES = _discover_audio_inputs()
_AUDIO_CHOICES = ["default"] + list(_AUDIO_DEVICES.keys())
_RATE_CHOICES = _discover_supported_rates()


@source_node(
    name="microphone",
    outputs=[Port("audio", AudioSignal)],
    params=[
        Param("source_id", "str", "mic", label="Source ID"),
        Param("chunk_size", "int", 1024, label="Chunk Size", min=128, max=8192),
        Param("sample_rate", "choice", "48000", label="Sample Rate",
              choices=_RATE_CHOICES),
        Param("device", "choice", "default", label="Audio Device",
              choices=_AUDIO_CHOICES),
    ],
)
def microphone(*, state, config, clock):
    sample_rate = int(config["sample_rate"])
    chunk_size = config["chunk_size"]
    device = config["device"]

    if "stream" not in state:
        sd_device = None if device == "default" else _AUDIO_DEVICES.get(device, None)
        log.info("opening audio stream: device=%s, rate=%d, chunk=%d", sd_device, sample_rate, chunk_size)
        state["stream"] = sd.InputStream(
            device=sd_device,
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=chunk_size,
        )
        state["stream"].start()
        state["sample_rate"] = sample_rate

    data, overflowed = state["stream"].read(chunk_size)
    if overflowed:
        log.warning("audio overflow (device=%s)", device)
    return {"audio": Sample(
        source_id=config["source_id"],
        lsl_timestamp=clock.lsl_now(),
        session_time_ms=clock.session_time_ms(),
        data=data.flatten(),
        metadata={"sample_rate": state["sample_rate"]},
        port_type=AudioSignal,
    )}


@microphone.cleanup
def microphone_cleanup(state, config):
    if "stream" in state:
        log.info("closing audio stream")
        state["stream"].stop()
        state["stream"].close()
