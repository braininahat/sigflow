"""ONNX Runtime execution provider selection.

Fixed to CPU.  Consumers (e.g. `nodes/dlc_inference.py`) import
`get_providers()` so a future re-introduction of CUDA / TensorRT is a
one-line change here rather than a change to every `InferenceSession`
call site.
"""
from __future__ import annotations


def get_providers() -> list[str]:
    """Return the ordered provider list for ``ort.InferenceSession``."""
    return ["CPUExecutionProvider"]
