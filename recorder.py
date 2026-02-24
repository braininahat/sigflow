"""Pipeline-level session recorder — captures node outputs to XDF + MP4.

Produces:
  session_YYYYMMDD_HHMMSS/
  ├── streams.xdf      (audio, keypoints, landmarks, events, video timestamps)
  ├── <node_id>.mp4    (one per producing node — avoids resolution collisions)
  └── metadata.json    (session config, stream registry, timing)
"""
import json
import logging
import queue
import subprocess
import threading
from datetime import datetime
from pathlib import Path

import imageio_ffmpeg

from sigflow.types import (
    PortType, TimeSeries2D, AudioSignal, Keypoints, FaceLandmarks,
    Scalar, Event, ROI,
)
from sigflow.xdf_writer import (
    open_xdf_raw, close_xdf, add_stream, push_numeric_sample,
    push_numeric_samples, push_string_sample,
)

log = logging.getLogger(__name__)

_SENTINEL = object()  # signals writer thread to shut down


def _open_ffmpeg_writer(filepath, w, h, fps):
    """Open an ffmpeg subprocess that accepts raw BGR24 frames on stdin."""
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "pipe:0",
        "-an",
        "-vcodec", "libx264",
        "-preset", "ultrafast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        str(filepath),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)


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
        vw["proc"].stdin.close()
        vw["proc"].wait(timeout=10)
        log.info("video %s: %d frames", vw["filename"], vw["frame_count"])
        for entry in state.get("metadata", {}).get("streams", []):
            if entry.get("filename") == vw["filename"]:
                entry["frame_count"] = vw["frame_count"]

    if "xdf" in state:
        close_xdf(state["xdf"])
        xdf_streams = state["xdf"].get("streams", {})
        type_counts: dict[str, int] = {}
        for sid, sinfo in xdf_streams.items():
            stype = sinfo.get("type", "unknown")
            type_counts[stype] = type_counts.get(stype, 0) + 1
        log.info("XDF closed: %d streams (%s)",
                 len(xdf_streams),
                 ", ".join(f"{t}={c}" for t, c in sorted(type_counts.items())))

    session_dir = state["session_dir"]

    if "metadata" in state:
        meta_path = session_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(state["metadata"], f, indent=2)
        log.info("metadata → %s", meta_path)

    for key in ("session_dir", "xdf", "video_writers", "xdf_streams", "metadata"):
        state.pop(key, None)

    return session_dir


def _get_or_create_xdf_stream(state, source_id, port_type, node_id=None, **kwargs):
    """Lazy-create an XDF stream for a (source_id, port_type) pair."""
    key = (node_id or source_id, port_type.__name__)
    if key not in state["xdf_streams"]:
        stream_id = add_stream(
            state["xdf"],
            name=f"{source_id}_{port_type.__name__}",
            source_id=source_id,
            stream_type=port_type.__name__,
            **kwargs,
        )
        state["xdf_streams"][key] = stream_id
        entry = {
            "source_id": source_id,
            "port_type": port_type.__name__,
            "xdf_stream_id": stream_id,
            **kwargs,
        }
        if node_id:
            entry["node_id"] = node_id
        state["metadata"]["streams"].append(entry)
        log.info("XDF stream %d: %s/%s (%s ch=%d)",
                 stream_id, source_id, port_type.__name__,
                 kwargs.get("channel_format", "?"),
                 kwargs.get("channel_count", 0))
    return state["xdf_streams"][key]


def _record_video(sample, state, config, node_id=None):
    """Write video frame to MP4, timestamp to XDF."""
    vid_key = node_id or sample.source_id
    frame = sample.data

    if vid_key not in state["video_writers"]:
        h, w = frame.shape[:2]
        safe_name = vid_key.lower().replace(" ", "_")
        filename = f"{safe_name}.mp4"
        filepath = state["session_dir"] / filename
        proc = _open_ffmpeg_writer(filepath, w, h, config["video_fps"])
        ts_stream_id = add_stream(
            state["xdf"],
            name=f"{vid_key}_timestamps",
            channel_format="double64",
            channel_count=1,
            nominal_srate=0,
            source_id=vid_key,
            stream_type="VideoTimestamps",
        )
        state["video_writers"][vid_key] = {
            "proc": proc,
            "xdf_stream_id": ts_stream_id,
            "frame_count": 0,
            "filename": filename,
            "width": w,
            "height": h,
        }
        state["metadata"]["streams"].append({
            "source_id": sample.source_id,
            "node_id": vid_key,
            "port_type": sample.port_type.__name__,
            "format": "mp4",
            "filename": filename,
            "width": w, "height": h,
            "declared_fps": config["video_fps"],
            "xdf_timestamp_stream_id": ts_stream_id,
        })
        log.info("video writer: %s (%dx%d @ %dfps)", filepath, w, h,
                 config["video_fps"])

    vw = state["video_writers"][vid_key]
    vw["proc"].stdin.write(frame.tobytes())
    vw["frame_count"] += 1
    if vw["frame_count"] % 100 == 0:
        log.debug("video %s: %d frames", vid_key, vw["frame_count"])
    push_numeric_sample(state["xdf"], vw["xdf_stream_id"],
                        sample.lsl_timestamp, [sample.lsl_timestamp])


def _record_audio(sample, state, node_id=None):
    """Write audio chunk to XDF."""
    sr = sample.metadata.get("sample_rate", 48000)
    audio = sample.data
    n = len(audio)

    key = (node_id or sample.source_id, AudioSignal.__name__)
    is_first = key not in state["xdf_streams"]
    stream_id = _get_or_create_xdf_stream(
        state, sample.source_id, sample.port_type, node_id=node_id,
        channel_format="float32", channel_count=1,
        nominal_srate=float(sr),
    )

    if is_first:
        log.debug("audio stream %s: sr=%d, chunk=%d", sample.source_id, sr, n)

    dt = 1.0 / sr
    timestamps = [sample.lsl_timestamp + i * dt for i in range(n)]
    values = [[float(v)] for v in audio]
    push_numeric_samples(state["xdf"], stream_id, timestamps, values)


def _record_keypoints(sample, state, node_id=None):
    """Write keypoints/landmarks as flattened double64."""
    flat = sample.data.flatten().tolist()
    stream_id = _get_or_create_xdf_stream(
        state, sample.source_id, sample.port_type, node_id=node_id,
        channel_format="double64", channel_count=len(flat),
        nominal_srate=0,
    )
    push_numeric_sample(state["xdf"], stream_id, sample.lsl_timestamp, flat)


def _record_roi(sample, state, node_id=None):
    """Write ROI (x, y, w, h) as double64."""
    vals = sample.data.flatten().tolist()
    stream_id = _get_or_create_xdf_stream(
        state, sample.source_id, sample.port_type, node_id=node_id,
        channel_format="double64", channel_count=len(vals),
        nominal_srate=0,
    )
    push_numeric_sample(state["xdf"], stream_id, sample.lsl_timestamp, vals)


def _record_scalar(sample, state, node_id=None):
    """Write scalar value as double64."""
    val = float(sample.data)
    stream_id = _get_or_create_xdf_stream(
        state, sample.source_id, sample.port_type, node_id=node_id,
        channel_format="double64", channel_count=1,
        nominal_srate=0,
    )
    push_numeric_sample(state["xdf"], stream_id, sample.lsl_timestamp, [val])


def _record_event(sample, state, node_id=None):
    """Write event as string."""
    stream_id = _get_or_create_xdf_stream(
        state, sample.source_id, sample.port_type, node_id=node_id,
        channel_format="string", channel_count=1,
        nominal_srate=0,
    )
    push_string_sample(state["xdf"], stream_id, sample.lsl_timestamp,
                       [str(sample.data)])


def _route_sample(sample, state, config, node_id=None):
    """Route a sample to the appropriate recorder by port_type."""
    pt = sample.port_type
    log.debug("record: %s → %s (node=%s)", sample.source_id, pt.__name__, node_id)
    if issubclass(pt, TimeSeries2D) and sample.data.ndim == 3:
        _record_video(sample, state, config, node_id=node_id)
    elif issubclass(pt, AudioSignal):
        _record_audio(sample, state, node_id=node_id)
    elif issubclass(pt, (Keypoints, FaceLandmarks)):
        _record_keypoints(sample, state, node_id=node_id)
    elif issubclass(pt, ROI):
        _record_roi(sample, state, node_id=node_id)
    elif issubclass(pt, Scalar):
        _record_scalar(sample, state, node_id=node_id)
    elif issubclass(pt, Event):
        _record_event(sample, state, node_id=node_id)
    else:
        log.warning("recorder: unhandled port_type %s", pt.__name__)


def _writer_loop(q, state, config):
    """Drain queue on a dedicated thread — sole consumer of state."""
    while True:
        item = q.get()
        if item is _SENTINEL:
            return
        sample, node_id = item
        _ensure_session(state, config)
        _route_sample(sample, state, config, node_id=node_id)


class SessionRecorder:
    """Pipeline-level session recorder.

    Pool threads enqueue samples via write() and return immediately.
    A dedicated writer thread drains the queue and does all I/O —
    no lock contention, no FFmpeg buffer overflows.
    """

    def __init__(self, output_dir="recordings", video_fps=30, on_sample=None):
        self._config = {
            "output_dir": output_dir,
            "video_fps": video_fps,
        }
        self._state: dict = {}
        self._on_sample = on_sample
        self._first_sample_logged = False

        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(
            target=_writer_loop,
            args=(self._queue, self._state, self._config),
            daemon=True,
        )
        self._thread.start()

        log.info("SessionRecorder created: output_dir=%s, on_sample=%s",
                 output_dir, on_sample is not None)

    def write(self, sample, node_id=None):
        """Enqueue a sample for the writer thread. Returns immediately."""
        if not self._first_sample_logged:
            log.info("recorder: first sample received: %s/%s",
                     sample.source_id, sample.port_type.__name__)
            self._first_sample_logged = True
        self._queue.put((sample, node_id))
        if self._on_sample:
            self._on_sample(sample, node_id)

    def finalize(self):
        """Drain queue, close video writers, XDF, write metadata.
        Returns session dir Path or None."""
        self._queue.put(_SENTINEL)
        self._thread.join(timeout=10.0)
        if self._thread.is_alive():
            log.warning("writer thread did not shut down within timeout")
        if "session_dir" not in self._state:
            log.warning("finalize: no session was created (no samples received)")
        return _finalize_session(self._state, self._config)
