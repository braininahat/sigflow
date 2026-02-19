"""Android microphone capture source via scrcpy audio recording.

Launches scrcpy with --no-video and --audio-codec=raw, recording
to a named FIFO pipe.  Reads raw PCM chunks from the pipe after
skipping the 44-byte WAV header.
"""
import logging
import os
import struct
import tempfile

import numpy as np

from sigflow.node import source_node, Param
from sigflow.types import Port, Sample, AudioSignal
from sigflow.nodes._scrcpy import launch_scrcpy, drain_output, kill_scrcpy

log = logging.getLogger(__name__)

_WAV_HEADER_SIZE = 44


def _parse_wav_header(fd) -> dict:
    """Read and parse a 44-byte WAV header from a file descriptor.

    Returns dict with sample_rate, channels, bits_per_sample.
    """
    header = fd.read(_WAV_HEADER_SIZE)
    if len(header) < _WAV_HEADER_SIZE:
        raise IOError("incomplete WAV header from scrcpy audio pipe")

    channels = struct.unpack_from("<H", header, 22)[0]
    sample_rate = struct.unpack_from("<I", header, 24)[0]
    bits_per_sample = struct.unpack_from("<H", header, 34)[0]

    log.info(
        "scrcpy audio: %d Hz, %d ch, %d-bit",
        sample_rate, channels, bits_per_sample,
    )
    return {
        "sample_rate": sample_rate,
        "channels": channels,
        "bits_per_sample": bits_per_sample,
    }


@source_node(
    name="scrcpy_mic",
    outputs=[Port("audio", AudioSignal)],
    params=[
        Param("serial", "str", "", label="Device Serial"),
        Param("audio_source", "choice", "mic", label="Audio Source",
              choices=["mic", "mic-unprocessed", "mic-camcorder",
                       "mic-voice-recognition"]),
        Param("chunk_size", "int", 1024, label="Chunk Size",
              min=128, max=8192),
        Param("source_id", "str", "android_mic", label="Source ID"),
        Param("scrcpy_path", "str", "scrcpy", label="scrcpy Path"),
    ],
)
def scrcpy_mic(*, state, config, clock):
    if "fd" not in state:
        return None

    chunk_size = config["chunk_size"]
    channels = state["channels"]
    bytes_per_sample = state["bits_per_sample"] // 8
    nbytes = chunk_size * channels * bytes_per_sample

    raw = state["fd"].read(nbytes)
    if not raw:
        return None

    if bytes_per_sample == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        audio = np.frombuffer(raw, dtype=np.float32)

    return {"audio": Sample(
        source_id=config["source_id"],
        lsl_timestamp=clock.lsl_now(),
        session_time_ms=clock.session_time_ms(),
        data=audio,
        metadata={"sample_rate": state["sample_rate"]},
        port_type=AudioSignal,
    )}


@scrcpy_mic.init
def scrcpy_mic_init(state, config):
    fifo_path = os.path.join(
        tempfile.gettempdir(),
        f"sigflow_scrcpy_audio_{os.getpid()}.wav",
    )
    os.mkfifo(fifo_path)
    state["fifo_path"] = fifo_path

    args = [
        "--no-video",
        "--audio-source", config["audio_source"],
        "--audio-codec", "raw",
        "--record", fifo_path,
    ]
    if config["serial"]:
        args += ["--serial", config["serial"]]

    state["proc"] = launch_scrcpy(args, config["scrcpy_path"])

    # Opening the FIFO blocks until scrcpy opens it for writing
    try:
        state["fd"] = open(fifo_path, "rb")
        wav = _parse_wav_header(state["fd"])
    except Exception:
        output = drain_output(state["proc"])
        log.error("scrcpy audio pipe failed. scrcpy output:\n%s", output or "(no output)")
        kill_scrcpy(state["proc"])
        return

    state["sample_rate"] = wav["sample_rate"]
    state["channels"] = wav["channels"]
    state["bits_per_sample"] = wav["bits_per_sample"]


@scrcpy_mic.cleanup
def scrcpy_mic_cleanup(state, config):
    if "fd" in state:
        state["fd"].close()
    if "proc" in state:
        kill_scrcpy(state["proc"])
    fifo_path = state.get("fifo_path")
    if fifo_path and os.path.exists(fifo_path):
        os.unlink(fifo_path)
        log.info("removed audio FIFO %s", fifo_path)
