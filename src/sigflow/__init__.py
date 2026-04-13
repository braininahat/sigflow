"""sigflow — reactive DAG pipeline framework for real-time signal processing."""

from sigflow.types import (
    PortType, TimeSeries, TimeSeries2D, TimeSeries1D,
    UltrasoundFrame, CameraFrame, IRFrame, DepthFrame,
    AudioSignal, PoseSequence, IMUSignal, BioSignal,
    Keypoints, TongueKeypoints, LipKeypoints, FaceLandmarks,
    Scalar, Score, Confidence,
    Event, PhonemeEvent, MarkerEvent,
    ROI, MouthROI,
    Port, Sample, compatible,
)
from sigflow.node import source_node, process_node, sink_node, NodeSpec, Param
from sigflow.registry import register, get as get_node, all_nodes, scan_plugins, clear
from sigflow.graph import Graph, NodeDef, Connection

# Runtime and recording have heavy deps (imageio_ffmpeg, pylsl, opencv).
# Import lazily so types/nodes are usable without them.
try:
    from sigflow.runtime import Pipeline, PipelineMode, IncompatiblePortError, MasterClock
    from sigflow.metrics import NodeMetrics, MetricsTracker, MetricsCollector
except ImportError:
    pass
