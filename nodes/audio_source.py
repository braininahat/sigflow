"""Microphone capture source node (sounddevice)."""
import numpy as np
import sounddevice as sd

from sigflow.node import source_node
from sigflow.types import Port, Sample, AudioSignal


@source_node(
    name="microphone",
    outputs=[Port("audio", AudioSignal)],
    category="source",
)
def microphone(*, state, config, clock):
    sample_rate = config.get("sample_rate", 44100)
    chunk_size = config.get("chunk_size", 1024)

    if "stream" not in state:
        state["stream"] = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=chunk_size,
        )
        state["stream"].start()

    data, overflowed = state["stream"].read(chunk_size)
    return {"audio": Sample(
        source_id=config.get("source_id", "mic"),
        lsl_timestamp=clock.lsl_now(),
        session_time_ms=clock.session_time_ms(),
        data=data.flatten(),
        metadata={"sample_rate": sample_rate},
        port_type=AudioSignal,
    )}


@microphone.cleanup
def microphone_cleanup(state, config):
    if "stream" in state:
        state["stream"].stop()
        state["stream"].close()
