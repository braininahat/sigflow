"""Session reader — loads recorded sessions for playback.

Reads:
  session_YYYYMMDD_HHMMSS/
  ├── streams.xdf      (audio, keypoints, landmarks, events, video timestamps)
  ├── <source_id>.mp4  (one per video stream)
  └── metadata.json    (session config, stream registry, timing)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pyxdf

log = logging.getLogger(__name__)


@dataclass
class StreamInfo:
    """Metadata for a single stream in a recorded session."""
    stream_id: int
    source_id: str
    port_type: str
    channel_format: str
    channel_count: int
    nominal_srate: float
    node_id: str | None = None
    # Video-specific
    filename: str | None = None
    width: int | None = None
    height: int | None = None
    frame_count: int | None = None


class SessionReader:
    """Loads a recorded session directory for playback."""

    def __init__(self, session_dir: Path):
        self._dir = Path(session_dir)
        self._metadata: dict = {}
        self._xdf_streams: list[dict] = []
        self._video_caps: dict[str, cv2.VideoCapture] = {}
        self._stream_infos: list[StreamInfo] = []
        self._load()

    def _load(self):
        meta_path = self._dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self._metadata = json.load(f)

        xdf_path = self._dir / "streams.xdf"
        if xdf_path.exists():
            self._xdf_streams, _ = pyxdf.load_xdf(str(xdf_path))
            for s in self._xdf_streams:
                info = s["info"]
                name = info["name"][0] if isinstance(info["name"], list) else info["name"]
                n_samples = len(s.get("time_stamps", []))
                stype = info.get("type", ["?"])
                stype = stype[0] if isinstance(stype, list) else stype
                log.debug("stream: %s (%s, %d samples)", name, stype, n_samples)

        # Build stream inventory from XDF + metadata
        xdf_by_name = {s["info"]["name"][0]: s for s in self._xdf_streams}

        for meta_entry in self._metadata.get("streams", []):
            source_id = meta_entry["source_id"]
            port_type = meta_entry["port_type"]

            if meta_entry.get("format") == "mp4":
                # Video stream — timestamps are in XDF, frames in MP4
                ts_stream_id = meta_entry.get("xdf_timestamp_stream_id")
                ts_stream = next(
                    (s for s in self._xdf_streams
                     if int(s["info"]["stream_id"]) == ts_stream_id),
                    None
                )
                frame_count = meta_entry.get("frame_count")
                if ts_stream is None:
                    # Find by name pattern
                    ts_stream = xdf_by_name.get(f"{source_id}_timestamps")

                self._stream_infos.append(StreamInfo(
                    stream_id=ts_stream_id or 0,
                    source_id=source_id,
                    port_type=port_type,
                    channel_format="double64",
                    channel_count=1,
                    nominal_srate=0,
                    node_id=meta_entry.get("node_id"),
                    filename=meta_entry.get("filename"),
                    width=meta_entry.get("width"),
                    height=meta_entry.get("height"),
                    frame_count=frame_count,
                ))
            else:
                # XDF-only stream
                xdf_stream_id = meta_entry.get("xdf_stream_id")
                xdf_stream = next(
                    (s for s in self._xdf_streams
                     if int(s["info"]["stream_id"]) == xdf_stream_id),
                    None
                )
                if xdf_stream is None:
                    continue

                info = xdf_stream["info"]
                ch_format = info["channel_format"][0] if isinstance(info["channel_format"], list) else info["channel_format"]
                ch_count = int(info["channel_count"][0]) if isinstance(info["channel_count"], list) else int(info["channel_count"])
                nom_srate = float(info["nominal_srate"][0]) if isinstance(info["nominal_srate"], list) else float(info["nominal_srate"])

                self._stream_infos.append(StreamInfo(
                    stream_id=xdf_stream_id,
                    source_id=source_id,
                    port_type=port_type,
                    channel_format=ch_format,
                    channel_count=ch_count,
                    nominal_srate=nom_srate,
                    node_id=meta_entry.get("node_id"),
                ))

        log.info("loaded session: %s (%d streams)", self._dir, len(self._stream_infos))

    @property
    def streams(self) -> list[StreamInfo]:
        return self._stream_infos

    @property
    def duration_s(self) -> float:
        t0, t1 = self.time_range
        return t1 - t0

    @property
    def time_range(self) -> tuple[float, float]:
        """Returns (first_timestamp, last_timestamp) across all XDF streams."""
        first = float("inf")
        last = float("-inf")
        for s in self._xdf_streams:
            ts = s["time_stamps"]
            if len(ts) > 0:
                first = min(first, ts[0])
                last = max(last, ts[-1])
        if first == float("inf"):
            return (0.0, 0.0)
        return (float(first), float(last))

    def get_time_series(self, stream_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns (timestamps, data) for an XDF stream by stream_id."""
        for s in self._xdf_streams:
            if int(s["info"]["stream_id"]) == stream_id:
                return s["time_stamps"], s["time_series"]
        raise KeyError(f"No XDF stream with id {stream_id}")

    def get_video_frame(self, source_id: str, frame_index: int) -> np.ndarray | None:
        """Decode a single frame from the MP4 for a video source."""
        cap = self._get_video_cap(source_id)
        if cap is None:
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            log.warning("video frame read failed: %s frame=%d", source_id, frame_index)
        return frame if ok else None

    def get_video_frame_at_time(self, source_id: str, timestamp: float) -> np.ndarray | None:
        """Find the nearest frame index from XDF timestamps, then decode."""
        info = next((s for s in self._stream_infos
                     if (s.node_id == source_id or s.source_id == source_id) and s.filename), None)
        if info is None:
            return None
        try:
            ts, _ = self.get_time_series(info.stream_id)
        except KeyError:
            log.warning("video seek failed: no timestamp stream for %s (stream_id=%s)",
                        source_id, info.stream_id)
            return None
        if len(ts) == 0:
            return None
        idx = int(np.argmin(np.abs(ts - timestamp)))
        return self.get_video_frame(source_id, idx)

    def _get_video_cap(self, source_id: str) -> cv2.VideoCapture | None:
        if source_id in self._video_caps:
            return self._video_caps[source_id]
        info = next((s for s in self._stream_infos
                     if (s.node_id == source_id or s.source_id == source_id) and s.filename), None)
        if info is None:
            return None
        path = self._dir / info.filename
        if not path.exists():
            log.warning("cannot open video: %s (file not found)", path)
            return None
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            log.warning("cannot open video: %s (cv2 open failed)", path)
            return None
        self._video_caps[source_id] = cap
        return cap

    def close(self):
        """Release video capture resources."""
        for cap in self._video_caps.values():
            cap.release()
        self._video_caps.clear()

    def __del__(self):
        self.close()
