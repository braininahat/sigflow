"""Sonostar wireless ultrasound source node — uses ProbeClient for transport."""
import logging

import cv2

from sigflow.node import source_node, Param
from sigflow.types import Port, Sample, UltrasoundFrame

log = logging.getLogger(__name__)

_RENDER_PARAMS = ("gray_map", "persistence", "median", "coherence", "morph_close")
_ZOOM_MM_PER_SAMPLE = {0: 0.039, 1: 0.078, 2: 0.117, 3: 0.195}


@source_node(
    name="sonostar",
    outputs=[Port("frame", UltrasoundFrame)],
    params=[
        # Connection
        Param("source_id", "str", "ultrasound", label="Source ID"),
        Param("host", "str", "192.168.1.1", label="Probe IP"),
        # Protocol: BD command
        Param("gain", "int", 105, label="Gain", min=30, max=105),
        Param("dynamic_range", "int", 110, label="Dynamic Range", min=40, max=110),
        Param("enhance", "int", 2, label="Enhance", min=0, max=4),
        Param("focus", "int", 0, label="Focus", min=0, max=3),
        Param("zoom", "int", 3, label="Zoom", min=0, max=3),
        Param("harmonic", "bool", False, label="Harmonic"),
        # Render: local post-processing
        Param("gray_map", "int", 8, label="Gray Map", min=0, max=15),
        Param("persistence", "float", 0.0, label="Persistence", min=0.0, max=1.0),
        Param("median", "int", 0, label="Median Filter", min=-1, max=2),
        Param("coherence", "int", 0, label="Coherence", min=0, max=3),
        Param("morph_close", "bool", False, label="Morph Close"),
    ],
    category="",
)
def sonostar(*, state, config, clock):
    client = state.get("client")
    if client is None:
        return None

    renderer = state["renderer"]
    prev = state["_prev_params"]

    # Live probe parameter updates via ProbeClient methods
    if prev.get("gain") != config["gain"]:
        prev["gain"] = config["gain"]
        client.set_gain(config["gain"])

    if prev.get("zoom") != config["zoom"]:
        prev["zoom"] = config["zoom"]
        client.set_zoom(config["zoom"])
        state["renderer"] = renderer = _build_renderer(config)
        state["_frame_metadata"] = _compute_frame_metadata(renderer)

    bd_dirty = False
    for name in ("dynamic_range", "enhance", "focus", "harmonic"):
        if prev.get(name) != config[name]:
            prev[name] = config[name]
            bd_dirty = True
    if bd_dirty:
        client.send_bd(
            enhance=config["enhance"],
            dynamic_range=config["dynamic_range"],
            focus=config["focus"],
            harmonic=config["harmonic"],
        )

    # Render parameter updates (local post-processing)
    render_dirty = False
    for name in _RENDER_PARAMS:
        if prev.get(name) != config[name]:
            prev[name] = config[name]
            render_dirty = True
    if render_dirty:
        state["renderer"] = renderer = _build_renderer(config)
        state["_frame_metadata"] = _compute_frame_metadata(renderer)

    # Read next frame from ProbeClient queue
    probe_frame = client.read_frame(timeout=0.001)
    if probe_frame is None:
        return None

    image = renderer.render(probe_frame.to_numpy())
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return {"frame": Sample(
        source_id=config["source_id"],
        lsl_timestamp=clock.lsl_now(),
        session_time_ms=clock.session_time_ms(),
        data=bgr,
        metadata=state["_frame_metadata"],
        port_type=UltrasoundFrame,
    )}


def _build_renderer(config):
    from sonospy.render import BmodeRenderer
    mm_per_sample = _ZOOM_MM_PER_SAMPLE.get(config["zoom"], 0.195)
    return BmodeRenderer.for_probe(
        "microconvex",
        mm_per_sample=mm_per_sample,
        gray_map_index=config["gray_map"],
        persistence=config["persistence"],
        median_level=config["median"],
        coherence_threshold=config["coherence"],
        morph_close=config["morph_close"],
        double_samples=True,
        double_lines=True,
    )


def _compute_frame_metadata(renderer):
    geo = renderer.geometry
    n_eff = geo.samples_per_line
    if renderer.double_samples:
        n_eff = 2 * n_eff - 1
    r_end = geo.dead_radius_mm + n_eff * geo.mm_per_sample
    scale = r_end / (renderer.output_size * 0.9)
    return {
        "mm_per_pixel": scale,
        "depth_mm": n_eff * geo.mm_per_sample,
        "dead_radius_mm": geo.dead_radius_mm,
        "scan_angle_deg": geo.scan_angle_deg,
    }


@sonostar.init
def sonostar_init(state, config):
    from sonospy import ProbeClient

    host = config["host"]
    log.info("connecting to Sonostar probe at %s (ProbeClient)", host)

    client = ProbeClient(host=host, lines_per_frame=160, samples_per_line=512)
    client.connect()
    log.info("ProbeClient connected to %s", host)

    # Set initial imaging parameters
    client.send_bd(
        enhance=config["enhance"],
        dynamic_range=config["dynamic_range"],
        focus=config["focus"],
        harmonic=config["harmonic"],
    )
    client.set_gain(config["gain"])
    client.set_zoom(config["zoom"])

    state["client"] = client
    state["renderer"] = _build_renderer(config)
    state["_frame_metadata"] = _compute_frame_metadata(state["renderer"])
    state["_prev_params"] = {
        **{name: config[name] for name in ("dynamic_range", "enhance", "focus", "harmonic")},
        "gain": config["gain"], "zoom": config["zoom"],
        **{name: config[name] for name in _RENDER_PARAMS},
    }
    log.info("Sonostar probe initialized")


@sonostar.cleanup
def sonostar_cleanup(state, config):
    log.info("closing Sonostar probe connection")
    client = state.get("client")
    if client is not None:
        try:
            client.close()
        except Exception as e:
            log.warning("error closing ProbeClient: %s", e)
