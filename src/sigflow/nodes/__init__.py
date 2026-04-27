"""Built-in sigflow nodes. Import this package to auto-register all nodes."""
import importlib
import logging
import time

log = logging.getLogger(__name__)

_NODE_MODULES = [
    "webcam_source",
    "cv2_display",
    "crop",
    "flip",
    "audio_source",
    "spectrogram",
    "canvas_display",
    "dlc_inference",
    "keypoints_overlay",
    "scrcpy_screen",
    "scrcpy_camera",
    "scrcpy_mic",
    "sonostar_source",
    "face_mesh",
    "mesh_overlay",
    "app_display",
    "skinned_tongue_display",
    "tongue_model_display",
    "tts_synthesis",
    "audio_playback",
]

_t_total = time.perf_counter()
for _mod in _NODE_MODULES:
    _t0 = time.perf_counter()
    try:
        importlib.import_module(f"sigflow.nodes.{_mod}")
        log.info("    [sigflow.nodes] loaded %-28s (%.2fs)", _mod, time.perf_counter() - _t0)
    except Exception:
        log.warning("    [sigflow.nodes] FAILED %-28s (%.2fs)", _mod, time.perf_counter() - _t0,
                    exc_info=True)
log.info("    [sigflow.nodes] all modules loaded in %.2fs", time.perf_counter() - _t_total)
