"""Pipeline-level session recorder — captures node outputs to XDF + MP4.

Produces:
  session_YYYYMMDD_HHMMSS/
  ├── streams.xdf      (audio, keypoints, landmarks, events, video timestamps)
  ├── <source_id>.mp4  (one per video stream)
  └── metadata.json    (session config, stream registry, timing)
"""
import json
import logging
import threading
from datetime import datetime
from pathlib import Path

import cv2

from sigflow.types import (
    PortType, TimeSeries2D, AudioSignal, Keypoints, FaceLandmarks,
    Scalar, Event, ROI,
)
from sigflow.xdf_writer import (
    open_xdf_raw, close_xdf, add_stream, push_numeric_sample,
    push_numeric_samples, push_string_sample,
)

log = logging.getLogger(__name__)


def _ensure_session(state, config):
    """Lazy-create session directory and XDF file on first sample."""
    if "session_dir" in state:
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(config["output_dir"]) / f"session_{ts}"
    session_dir.mkdir(parents=True, exist_ok=True)
    log.info("session recording → %s", session_dir)

    state["session_dir"] = session_dir
    state["xdf"] = open_xdf_raw(session_dir / "streams.xdf")
    state["video_writers"] = {}
    state["xdf_streams"] = {}
    state["metadata"] = {"start_time": ts, "streams": []}


def _finalize_session(state, config):
    """Close video writers, XDF, write metadata. Clears state for next session.
    Returns the session directory Path, or None if no session was active."""
    if "session_dir" not in state:
        return None

    for sid, vw in state.get("video_writers", {}).items():
        vw["writer"].release()
        log.info("video %s: %d frames", vw["filename"], vw["frame_count"])
        for entry in state.get("metadata", {}).get("streams", []):
            if entry.get("filename") == vw["filename"]:
                entry["frame_count"] = vw["frame_count"]

    if "xdf" in state:
        close_xdf(state["xdf"])
        log.info("XDF closed: %d streams",
                 len(state["xdf"].get("streams", {})))

    session_dir = state["session_dir"]

    if "metadata" in state:
        meta_path = session_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(state["metadata"], f, indent=2)
        log.info("metadata → %s", meta_path)

    for key in ("session_dir", "xdf", "video_writers", "xdf_streams", "metadata"):
        state.pop(key, None)

    return session_dir


def _get_or_create_xdf_stream(state, source_id, port_type, **kwargs):
    """Lazy-create an XDF stream for a (source_id, port_type) pair."""
    key = (source_id, port_type.__name__)
    if key not in state["xdf_streams"]:
        stream_id = add_stream(
            state["xdf"],
            name=f"{source_id}_{port_type.__name__}",
            source_id=source_id,
            stream_type=port_type.__name__,
            **kwargs,
        )
        state["xdf_streams"][key] = stream_id
        state["metadata"]["streams"].append({
            "source_id": source_id,
            "port_type": port_type.__name__,
            "xdf_stream_id": stream_id,
            **kwargs,
        })
        log.info("XDF stream %d: %s/%s (%s ch=%d)",
                 stream_id, source_id, port_type.__name__,
                 kwargs.get("channel_format", "?"),
                 kwargs.get("channel_count", 0))
    return state["xdf_streams"][key]


def _record_video(sample, state, config):
    """Write video frame to MP4, timestamp to XDF."""
    sid = sample.source_id
    frame = sample.data

    if sid not in state["video_writers"]:
        h, w = frame.shape[:2]
        filename = f"{sid}.mp4"
        filepath = state["session_dir"] / filename
        fourcc = cv2.VideoWriter.fourcc(*config["video_codec"])
        writer = cv2.VideoWriter(str(filepath), fourcc,
                                 config["video_fps"], (w, h))
        ts_stream_id = add_stream(
            state["xdf"],
            name=f"{sid}_timestamps",
            channel_format="double64",
            channel_count=1,
            nominal_srate=0,
            source_id=sid,
            stream_type="VideoTimestamps",
        )
        state["video_writers"][sid] = {
            "writer": writer,
            "xdf_stream_id": ts_stream_id,
            "frame_count": 0,
            "filename": filename,
            "width": w,
            "height": h,
        }
        state["metadata"]["streams"].append({
            "source_id": sid,
            "port_type": sample.port_type.__name__,
            "format": "mp4",
            "filename": filename,
            "width": w, "height": h,
            "declared_fps": config["video_fps"],
            "xdf_timestamp_stream_id": ts_stream_id,
        })
        log.info("video writer: %s (%dx%d @ %dfps)", filepath, w, h,
                 config["video_fps"])

    vw = state["video_writers"][sid]
    vw["writer"].write(frame)
    vw["frame_count"] += 1
    push_numeric_sample(state["xdf"], vw["xdf_stream_id"],
                        sample.lsl_timestamp, [sample.lsl_timestamp])


def _record_audio(sample, state):
    """Write audio chunk to XDF."""
    sr = sample.metadata.get("sample_rate", 48000)
    audio = sample.data
    n = len(audio)

    stream_id = _get_or_create_xdf_stream(
        state, sample.source_id, sample.port_type,
        channel_format="float32", channel_count=1,
        nominal_srate=float(sr),
    )

    dt = 1.0 / sr
    timestamps = [sample.lsl_timestamp + i * dt for i in range(n)]
    values = [[float(v)] for v in audio]
    push_numeric_samples(state["xdf"], stream_id, timestamps, values)


def _record_keypoints(sample, state):
    """Write keypoints/landmarks as flattened double64."""
    flat = sample.data.flatten().tolist()
    stream_id = _get_or_create_xdf_stream(
        state, sample.source_id, sample.port_type,
        channel_format="double64", channel_count=len(flat),
        nominal_srate=0,
    )
    push_numeric_sample(state["xdf"], stream_id, sample.lsl_timestamp, flat)


def _record_roi(sample, state):
    """Write ROI (x, y, w, h) as double64."""
    vals = sample.data.flatten().tolist()
    stream_id = _get_or_create_xdf_stream(
        state, sample.source_id, sample.port_type,
        channel_format="double64", channel_count=len(vals),
        nominal_srate=0,
    )
    push_numeric_sample(state["xdf"], stream_id, sample.lsl_timestamp, vals)


def _record_scalar(sample, state):
    """Write scalar value as double64."""
    val = float(sample.data)
    stream_id = _get_or_create_xdf_stream(
        state, sample.source_id, sample.port_type,
        channel_format="double64", channel_count=1,
        nominal_srate=0,
    )
    push_numeric_sample(state["xdf"], stream_id, sample.lsl_timestamp, [val])


def _record_event(sample, state):
    """Write event as string."""
    stream_id = _get_or_create_xdf_stream(
        state, sample.source_id, sample.port_type,
        channel_format="string", channel_count=1,
        nominal_srate=0,
    )
    push_string_sample(state["xdf"], stream_id, sample.lsl_timestamp,
                       [str(sample.data)])


def _route_sample(sample, state, config):
    """Route a sample to the appropriate recorder by port_type."""
    pt = sample.port_type
    if issubclass(pt, TimeSeries2D) and sample.data.ndim == 3:
        _record_video(sample, state, config)
    elif issubclass(pt, AudioSignal):
        _record_audio(sample, state)
    elif issubclass(pt, (Keypoints, FaceLandmarks)):
        _record_keypoints(sample, state)
    elif issubclass(pt, ROI):
        _record_roi(sample, state)
    elif issubclass(pt, Scalar):
        _record_scalar(sample, state)
    elif issubclass(pt, Event):
        _record_event(sample, state)
    else:
        log.warning("recorder: unhandled port_type %s", pt.__name__)


class SessionRecorder:
    """Pipeline-level session recorder. Thread-safe."""

    def __init__(self, output_dir="recordings", video_fps=30, video_codec="mp4v"):
        self._lock = threading.Lock()
        self._config = {
            "output_dir": output_dir,
            "video_fps": video_fps,
            "video_codec": video_codec,
        }
        self._state: dict = {}

    def write(self, sample):
        """Thread-safe: write a sample to the session."""
        with self._lock:
            _ensure_session(self._state, self._config)
            _route_sample(sample, self._state, self._config)

    def finalize(self):
        """Close video writers, XDF, write metadata. Returns session dir Path or None."""
        with self._lock:
            return _finalize_session(self._state, self._config)
