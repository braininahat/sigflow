"""ONNX Runtime execution provider selection.

Module-level state that ONNX model loaders consult when creating
InferenceSession instances.  Call ``set_tensorrt(True)`` to enable
TensorRT; loaders that call ``get_providers()`` will pick up the
change on next session creation.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

_use_tensorrt: bool = False


def set_tensorrt(enabled: bool) -> None:
    """Enable or disable TensorRT for future ONNX sessions."""
    global _use_tensorrt
    _use_tensorrt = enabled
    log.info("ONNX TensorRT %s", "enabled" if enabled else "disabled")


def get_providers() -> list[str]:
    """Return the ordered provider list for ``ort.InferenceSession``."""
    if _use_tensorrt:
        return [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
    return ["CPUExecutionProvider"]
