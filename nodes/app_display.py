"""Application display sinks — push pipeline output to the host app's UI.

Three sink nodes that call a module-level callback to route processed data
to the application layer (e.g. PipelineBridge). Unlike the implicit
Pipeline.on_sample approach, these appear explicitly in the protocol YAML
so the data flow from processing to display is visible and configurable.

The callback signature is: (display_id: str, kind: str, data: Any) -> None
  - kind="frame"     → numpy array (H x W x C)
  - kind="keypoints" → flat list [x0, y0, x1, y1, ...]
  - kind="waveform"  → numpy array (1D)
"""
from sigflow.node import sink_node, Param
from sigflow.types import Port, TimeSeries2D, Keypoints, AudioSignal

_display_callback = None  # (display_id, kind, data) -> None


def set_display_callback(cb):
    global _display_callback
    _display_callback = cb


@sink_node(
    name="app_display",
    inputs=[Port("frame", TimeSeries2D)],
    params=[Param("display_id", "str", "video", label="Display Target")],
)
def app_display(item, *, state, config):
    if _display_callback:
        _display_callback(config["display_id"], "frame", item.data)


@sink_node(
    name="app_keypoints",
    inputs=[Port("keypoints", Keypoints)],
    params=[Param("display_id", "str", "video", label="Display Target")],
)
def app_keypoints(item, *, state, config):
    data = item.data
    flat = data.flatten().tolist() if hasattr(data, "flatten") else data if isinstance(data, list) else []
    if _display_callback:
        _display_callback(config["display_id"], "keypoints", flat)


@sink_node(
    name="app_waveform",
    inputs=[Port("audio", AudioSignal)],
    params=[Param("display_id", "str", "audio", label="Display Target")],
)
def app_waveform(item, *, state, config):
    if _display_callback:
        _display_callback(config["display_id"], "waveform", item.data)
