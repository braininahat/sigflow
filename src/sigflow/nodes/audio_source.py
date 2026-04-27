"""Microphone capture source node (sounddevice).

Includes a shared circular audio buffer that other services can read
via get_audio_segment(t_start, t_end) to extract audio by LSL timestamp
range. This enables ElicitationService to use the pipeline mic instead
of running its own sounddevice InputStream.
"""
from __future__ import annotations

import logging
import threading

import numpy as np

from sigflow.node import source_node, Param
from sigflow.types import Port, Sample, AudioSignal

log = logging.getLogger(__name__)

_COMMON_RATES = [8000, 16000, 22050, 44100, 48000, 96000]


class SharedAudioBuffer:
    """Thread-safe circular buffer of timestamped audio chunks.

    Stores the last `max_seconds` of audio from the pipeline mic,
    indexed by LSL timestamp for extraction by time range.
    """

    def __init__(self, max_seconds: float = 30.0):
        self._lock = threading.Lock()
        self._chunks: list[tuple[float, np.ndarray]] = []  # (lsl_ts, data)
        self._sample_rate: int = 48000
        self._max_seconds = max_seconds

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, rate: int):
        self._sample_rate = rate

    def push(self, lsl_timestamp: float, data: np.ndarray) -> None:
        """Append a chunk with its LSL timestamp."""
        with self._lock:
            self._chunks.append((lsl_timestamp, data))
            # Evict old chunks beyond max_seconds
            if self._chunks:
                cutoff = self._chunks[-1][0] - self._max_seconds
                while self._chunks and self._chunks[0][0] < cutoff:
                    self._chunks.pop(0)

    def get_segment(self, t_start: float, t_end: float) -> np.ndarray | None:
        """Extract audio between LSL timestamps, concatenated."""
        with self._lock:
            segments = []
            for ts, data in self._chunks:
                chunk_end = ts + len(data) / self._sample_rate
                if chunk_end >= t_start and ts <= t_end:
                    segments.append(data)
            if not segments:
                return None
            return np.concatenate(segments)


# Module-level shared buffer instance
_shared_buffer: SharedAudioBuffer | None = None


def get_shared_buffer() -> SharedAudioBuffer | None:
    """Get the shared audio buffer (set by the microphone node)."""
    return _shared_buffer


def _discover_audio_inputs():
    """Return {name: index} for available audio input devices.

    Filters to the default host API to avoid duplicates from
    ALSA + PulseAudio + JACK all reporting the same hardware.
    """
    try:
        import sounddevice as sd
        default_api = sd.default.hostapi
        return {d["name"]: i for i, d in enumerate(sd.query_devices())
                if d["max_input_channels"] > 0 and d["hostapi"] == default_api}
    except Exception as e:
        log.warning("audio device discovery failed: %s", e)
        return {}


# Use a static rate list rather than probing the device with sd.check_input_settings.
# Why: probing each rate can segfault PortAudio on Pipewire/PulseAudio systems,
# which crashes the host process at import time (a C-level segfault can't be caught).
_RATE_CHOICES = [str(r) for r in _COMMON_RATES]

try:
    _AUDIO_DEVICES = _discover_audio_inputs()
except Exception:
    log.warning("audio device discovery raised", exc_info=True)
    _AUDIO_DEVICES = {}
_AUDIO_CHOICES = ["default"] + list(_AUDIO_DEVICES.keys())


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
    global _shared_buffer
    sample_rate = int(config["sample_rate"])
    chunk_size = config["chunk_size"]
    device = config["device"]

    if "stream" not in state:
        import sounddevice as sd
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
        # Initialize shared audio buffer for other services
        _shared_buffer = SharedAudioBuffer(max_seconds=30.0)
        _shared_buffer.sample_rate = sample_rate

    data, overflowed = state["stream"].read(chunk_size)
    if overflowed:
        log.warning("audio overflow (device=%s)", device)

    flat = data.flatten()
    ts = clock.lsl_now()

    # Push to shared buffer for ElicitationService
    if _shared_buffer is not None:
        _shared_buffer.push(ts, flat)

    return {"audio": Sample(
        source_id=config["source_id"],
        lsl_timestamp=ts,
        session_time_ms=clock.session_time_ms(),
        data=flat,
        metadata={"sample_rate": state["sample_rate"]},
        port_type=AudioSignal,
    )}


@microphone.cleanup
def microphone_cleanup(state, config):
    if "stream" in state:
        log.info("closing audio stream")
        state["stream"].stop()
        state["stream"].close()
