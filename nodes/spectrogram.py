"""Audio to spectrogram process node (scipy)."""
from scipy import signal as sig

from sigflow.node import process_node
from sigflow.types import Port, AudioSignal, TimeSeries2D


@process_node(
    name="spectrogram",
    inputs=[Port("audio", AudioSignal)],
    outputs=[Port("spectrogram", TimeSeries2D)],
    category="transform",
)
def spectrogram(item, *, state, config):
    sample_rate = item.metadata.get("sample_rate", 44100)
    nperseg = config.get("nperseg", min(256, len(item.data)))
    f, t, Sxx = sig.spectrogram(item.data, fs=sample_rate, nperseg=nperseg)
    return {"spectrogram": item.replace(data=Sxx, port_type=TimeSeries2D)}
