"""Core types for sigflow: Sample, Port, PortType hierarchy."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, NamedTuple


class PortType:
    """Base class for all port types."""

class TimeSeries(PortType):
    """Any time-series data."""

class TimeSeries2D(TimeSeries):
    """Any 2D video-like data (H x W x C)."""

class TimeSeries1D(TimeSeries):
    """Any 1D signal data."""

class UltrasoundFrame(TimeSeries2D):
    """Ultrasound probe / scrcpy screen capture."""

class CameraFrame(TimeSeries2D):
    """USB webcam, front camera."""

class IRFrame(TimeSeries2D):
    """SWIR / GigE camera."""

class DepthFrame(TimeSeries2D):
    """Depth sensor."""

class AudioSignal(TimeSeries1D):
    """Microphone audio (has sample_rate in metadata)."""

class PoseSequence(TimeSeries1D):
    """Keypoints over time."""

class IMUSignal(TimeSeries1D):
    """Accelerometer / gyroscope."""

class BioSignal(TimeSeries1D):
    """EEG / EMG / OpenBCI."""

class Keypoints(PortType):
    """Spatial keypoint data."""

class TongueKeypoints(Keypoints):
    """16 DLC tongue points."""

class LipKeypoints(Keypoints):
    """11 DLC lip points."""

class FaceLandmarks(Keypoints):
    """468 MediaPipe face points."""

class Scalar(PortType):
    """Single numeric value."""

class Score(Scalar):
    """Quality score (0-1)."""

class Confidence(Scalar):
    """Model confidence."""

class Event(PortType):
    """Discrete event."""

class PhonemeEvent(Event):
    """Phoneme label + timestamp."""

class MarkerEvent(Event):
    """User annotation."""

class ROI(PortType):
    """Region of interest."""

class MouthROI(ROI):
    """Mouth crop region from face landmarks."""


class Port(NamedTuple):
    """A named, typed port on a node."""
    name: str
    type: type[PortType]


def compatible(output_type: type[PortType], input_type: type[PortType]) -> bool:
    """Check if an output port type can connect to an input port type."""
    return issubclass(output_type, input_type)


@dataclass(frozen=False)
class Sample:
    """One timestamped observation flowing through the DAG."""
    source_id: str
    lsl_timestamp: float
    session_time_ms: int
    data: Any
    metadata: dict = field(default_factory=dict)
    port_type: type[PortType] = PortType
    frame_id: int = 0

    def with_metadata(self, **kw) -> Sample:
        """Shallow copy with merged metadata. Data is shared (zero-copy)."""
        new_meta = {**self.metadata, **kw}
        return dataclasses.replace(self, metadata=new_meta)

    def replace(self, **kw) -> Sample:
        """Shallow copy with replaced fields."""
        return dataclasses.replace(self, **kw)
