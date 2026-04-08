"""Built-in sigflow nodes. Import this package to auto-register all nodes."""

from sigflow.nodes import webcam_source  # noqa: F401
from sigflow.nodes import cv2_display  # noqa: F401
from sigflow.nodes import crop  # noqa: F401
from sigflow.nodes import flip  # noqa: F401
from sigflow.nodes import audio_source  # noqa: F401
from sigflow.nodes import spectrogram  # noqa: F401
from sigflow.nodes import canvas_display  # noqa: F401
from sigflow.nodes import dlc_inference  # noqa: F401
from sigflow.nodes import keypoints_overlay  # noqa: F401
from sigflow.nodes import scrcpy_screen  # noqa: F401
from sigflow.nodes import scrcpy_camera  # noqa: F401
from sigflow.nodes import scrcpy_mic  # noqa: F401
from sigflow.nodes import sonostar_source  # noqa: F401
from sigflow.nodes import face_mesh  # noqa: F401
from sigflow.nodes import mesh_overlay  # noqa: F401
from sigflow.nodes import app_display  # noqa: F401
from sigflow.nodes import tongue_model_display  # noqa: F401
from sigflow.nodes import llm_inference  # noqa: F401
from sigflow.nodes import tts_synthesis  # noqa: F401
from sigflow.nodes import audio_playback  # noqa: F401
