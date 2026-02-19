"""Minimal XDF (Extensible Data Format) writer for sigflow.

Writes valid XDF files readable by pyxdf.load_xdf().  Supports float32,
double64, and string channel formats — covering audio, keypoints,
landmarks, timestamps, events, and markers.

XDF spec: https://github.com/sccn/xdf/wiki/Specifications
"""
from __future__ import annotations

import struct
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Sequence
from xml.etree.ElementTree import Element, SubElement, tostring


# -- Binary primitives -------------------------------------------------------

_FORMAT_STRUCT = {
    "float32": "<f",
    "double64": "<d",
    "int8": "<b",
    "int16": "<h",
    "int32": "<i",
    "int64": "<q",
}


def _write_varlen(f: IO[bytes], value: int) -> None:
    """Write a variable-length integer (1, 4, or 8 byte encoding)."""
    if value < 256:
        f.write(b"\x01")
        f.write(struct.pack("<B", value))
    elif value < 2**32:
        f.write(b"\x04")
        f.write(struct.pack("<I", value))
    else:
        f.write(b"\x08")
        f.write(struct.pack("<Q", value))


def _varlen_bytes(value: int) -> bytes:
    """Return variable-length integer as bytes (for building buffers)."""
    if value < 256:
        return b"\x01" + struct.pack("<B", value)
    elif value < 2**32:
        return b"\x04" + struct.pack("<I", value)
    else:
        return b"\x08" + struct.pack("<Q", value)


def _write_chunk(f: IO[bytes], tag: int, content: bytes) -> None:
    """Write a complete XDF chunk: [varlen_length][tag:u16][content]."""
    length = 2 + len(content)  # 2 bytes for tag
    _write_varlen(f, length)
    f.write(struct.pack("<H", tag))
    f.write(content)


# -- XDF state dict -----------------------------------------------------------

def _make_stream_info(
    *,
    name: str,
    channel_format: str,
    channel_count: int,
    nominal_srate: float,
    source_id: str,
    stream_type: str,
) -> dict:
    return {
        "name": name,
        "channel_format": channel_format,
        "channel_count": channel_count,
        "nominal_srate": nominal_srate,
        "source_id": source_id,
        "type": stream_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "first_timestamp": None,
        "last_timestamp": None,
        "sample_count": 0,
    }


def _build_header_xml(info: dict) -> bytes:
    root = Element("info")
    for key in ("name", "type", "channel_count", "nominal_srate",
                "channel_format", "source_id", "created_at"):
        el = SubElement(root, key)
        el.text = str(info.get(key, ""))
    return tostring(root, encoding="unicode").encode("utf-8")


def _build_footer_xml(info: dict) -> bytes:
    root = Element("info")
    SubElement(root, "first_timestamp").text = str(info["first_timestamp"] or 0)
    SubElement(root, "last_timestamp").text = str(info["last_timestamp"] or 0)
    SubElement(root, "sample_count").text = str(info["sample_count"])
    return tostring(root, encoding="unicode").encode("utf-8")


# -- Public API ---------------------------------------------------------------

def open_xdf_raw(path: str | Path) -> dict:
    """Open a new XDF file for writing.  Returns writer state dict.

    Caller must call close_xdf() when done (writes stream footers).
    For automatic cleanup use the open_xdf() context manager instead.
    """
    f = open(path, "wb")
    f.write(b"XDF:")
    xml = b'<?xml version="1.0"?><info><version>1.0</version></info>'
    _write_chunk(f, tag=1, content=xml)
    return {"file": f, "streams": {}, "next_stream_id": 1}


@contextmanager
def open_xdf(path: str | Path):
    """Context manager that yields an XDF writer state dict."""
    state = open_xdf_raw(path)
    try:
        yield state
    finally:
        close_xdf(state)


def close_xdf(xdf: dict) -> None:
    """Write stream footers and close the XDF file."""
    f = xdf["file"]
    for stream_id, info in xdf["streams"].items():
        footer_xml = _build_footer_xml(info)
        content = struct.pack("<I", stream_id) + footer_xml
        _write_chunk(f, tag=6, content=content)
    f.close()


def add_stream(
    xdf: dict,
    *,
    name: str,
    channel_format: str,
    channel_count: int,
    nominal_srate: float = 0.0,
    source_id: str = "",
    stream_type: str = "",
) -> int:
    """Register a new stream.  Writes StreamHeader chunk.  Returns stream_id."""
    stream_id = xdf["next_stream_id"]
    xdf["next_stream_id"] += 1

    info = _make_stream_info(
        name=name,
        channel_format=channel_format,
        channel_count=channel_count,
        nominal_srate=nominal_srate,
        source_id=source_id,
        stream_type=stream_type,
    )
    xdf["streams"][stream_id] = info

    header_xml = _build_header_xml(info)
    content = struct.pack("<I", stream_id) + header_xml
    _write_chunk(xdf["file"], tag=2, content=content)
    return stream_id


def push_numeric_samples(
    xdf: dict,
    stream_id: int,
    timestamps: Sequence[float],
    values: Sequence[Sequence[float]],
) -> None:
    """Write a batch of numeric samples as a single Samples chunk.

    timestamps: N floats (LSL seconds)
    values:     N sequences, each with channel_count numbers
    """
    info = xdf["streams"][stream_id]
    fmt = _FORMAT_STRUCT[info["channel_format"]]
    nchns = info["channel_count"]
    n = len(timestamps)

    buf = bytearray()
    # StreamID
    buf.extend(struct.pack("<I", stream_id))
    # NumSamples (variable-length int)
    buf.extend(_varlen_bytes(n))

    for i in range(n):
        ts = timestamps[i]
        # Timestamp present flag (8 = 8-byte double follows)
        buf.append(8)
        buf.extend(struct.pack("<d", ts))
        # Channel values
        row = values[i]
        for j in range(nchns):
            buf.extend(struct.pack(fmt, row[j]))
        # Track first/last
        if info["first_timestamp"] is None:
            info["first_timestamp"] = ts
        info["last_timestamp"] = ts

    info["sample_count"] += n
    _write_chunk(xdf["file"], tag=3, content=bytes(buf))


def push_numeric_sample(
    xdf: dict,
    stream_id: int,
    timestamp: float,
    values: Sequence[float],
) -> None:
    """Write a single numeric sample."""
    push_numeric_samples(xdf, stream_id, [timestamp], [values])


def push_string_sample(
    xdf: dict,
    stream_id: int,
    timestamp: float,
    strings: Sequence[str],
) -> None:
    """Write a single string-format sample (one string per channel)."""
    info = xdf["streams"][stream_id]

    buf = bytearray()
    buf.extend(struct.pack("<I", stream_id))
    buf.extend(_varlen_bytes(1))  # 1 sample
    buf.append(8)  # timestamp present
    buf.extend(struct.pack("<d", timestamp))
    for s in strings:
        encoded = s.encode("utf-8")
        buf.extend(_varlen_bytes(len(encoded)))
        buf.extend(encoded)

    if info["first_timestamp"] is None:
        info["first_timestamp"] = timestamp
    info["last_timestamp"] = timestamp
    info["sample_count"] += 1
    _write_chunk(xdf["file"], tag=3, content=bytes(buf))
