"""Sonostar wireless ultrasound source node (raw socket, live_view pattern)."""
import logging
import socket
import time

import cv2

from sigflow.node import source_node, Param
from sigflow.types import Port, Sample, UltrasoundFrame

log = logging.getLogger(__name__)

# --- Param groups for change detection ---
_BD_PARAMS = ("dynamic_range", "enhance", "focus", "harmonic")
_HB_PARAMS = ("gain", "zoom")
_RENDER_PARAMS = ("gray_map", "persistence", "median", "coherence", "morph_close")


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
    data_sock = state.get("data_sock")
    if data_sock is None:
        return None

    ctrl_sock = state["ctrl_sock"]
    parser = state["parser"]
    assembler = state["assembler"]
    renderer = state["renderer"]

    # --- Live parameter updates: re-send commands when config changes ---
    prev = state["_prev_params"]

    # BD params → re-send BD command
    bd_dirty = False
    for name in _BD_PARAMS:
        if prev.get(name) != config[name]:
            prev[name] = config[name]
            bd_dirty = True
    if bd_dirty:
        _send_bd(ctrl_sock, config)

    # Heartbeat params → set countdowns
    if prev.get("gain") != config["gain"]:
        prev["gain"] = config["gain"]
        state["gain_countdown"] = 8

    if prev.get("zoom") != config["zoom"]:
        prev["zoom"] = config["zoom"]
        state["zoom_countdown"] = 8

    # Render params → rebuild renderer
    render_dirty = False
    for name in _RENDER_PARAMS:
        if prev.get(name) != config[name]:
            prev[name] = config[name]
            render_dirty = True
    if render_dirty:
        state["renderer"] = renderer = _build_renderer(config)

    # --- Timer: keepalive every 50ms, heartbeat every 100ms ---
    now = time.monotonic()
    if now - state["last_tick"] >= 0.050:
        state["last_tick"] = now
        state["tick"] += 1
        tick = state["tick"]

        data_sock.sendall(b'\x00\x00\x00\x00')

        if tick % 2 == 0:
            _send_heartbeat(ctrl_sock, state, config)
            if tick % 4 == 0:
                _send_bd(ctrl_sock, config)

    # --- Read data, assemble frames ---
    try:
        chunk = data_sock.recv(16384)
        if chunk:
            latest_frame = None
            for block in parser.feed(chunk):
                frame = assembler.push_block(block)
                if frame is not None:
                    latest_frame = frame
            if latest_frame is not None:
                image = renderer.render(latest_frame)
                bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                return {"frame": Sample(
                    source_id=config["source_id"],
                    lsl_timestamp=clock.lsl_now(),
                    session_time_ms=clock.session_time_ms(),
                    data=bgr,
                    metadata={},
                    port_type=UltrasoundFrame,
                )}
    except socket.timeout:
        pass

    # --- Drain control port ---
    try:
        ctrl_sock.recv(256)
    except (socket.timeout, OSError):
        pass

    return None


def _build_renderer(config):
    from sonospy.render import BmodeRenderer
    return BmodeRenderer.for_probe(
        "microconvex",
        gray_map_index=config["gray_map"],
        persistence=config["persistence"],
        median_level=config["median"],
        coherence_threshold=config["coherence"],
        morph_close=config["morph_close"],
        double_samples=True,
        double_lines=True,
    )


def _send_bd(ctrl_sock, config):
    from sonospy import protocol
    cmd = protocol.make_bd_command(
        enhance=config["enhance"],
        dynamic_range=config["dynamic_range"],
        focus=config["focus"],
        harmonic=config["harmonic"],
    )
    ctrl_sock.sendall(cmd)


def _send_heartbeat(ctrl_sock, state, config):
    from sonospy import protocol
    cmd = protocol.make_heartbeat(
        live=True,
        gain=config["gain"],
        zoom=config["zoom"],
        live_countdown=state["live_countdown"],
        zoom_countdown=state["zoom_countdown"],
        gain_countdown=state["gain_countdown"],
    )
    ctrl_sock.sendall(cmd)
    if state["live_countdown"] > 0:
        state["live_countdown"] -= 1
    if state["zoom_countdown"] > 0:
        state["zoom_countdown"] -= 1
    if state["gain_countdown"] > 0:
        state["gain_countdown"] -= 1


@sonostar.init
def sonostar_init(state, config):
    from sonospy import protocol
    from sonospy.data import FrameAssembler, StreamParser

    host = config["host"]
    log.info("connecting to Sonostar probe at %s (raw sockets)", host)

    data_sock = socket.create_connection((host, 5002), timeout=3.0)
    ctrl_sock = socket.create_connection((host, 5003), timeout=3.0)
    log.info("connected to data (5002) + control (5003)")

    # Send initial imaging commands
    bd_cmd = protocol.make_bd_command(
        enhance=config["enhance"],
        dynamic_range=config["dynamic_range"],
        focus=config["focus"],
        harmonic=config["harmonic"],
    )
    ctrl_sock.sendall(bd_cmd)
    ctrl_sock.sendall(protocol.make_vgain([64, 64, 64, 64, 64, 64, 64, 64]))

    # 1ms non-blocking timeout (matches live_view.py)
    data_sock.settimeout(0.001)
    ctrl_sock.settimeout(0.001)

    state["data_sock"] = data_sock
    state["ctrl_sock"] = ctrl_sock
    state["parser"] = StreamParser()
    state["assembler"] = FrameAssembler(lines_per_frame=160, samples_per_line=512)
    state["renderer"] = _build_renderer(config)
    state["tick"] = 0
    state["last_tick"] = 0.0
    state["live_countdown"] = 8
    state["gain_countdown"] = 8
    state["zoom_countdown"] = 8
    state["_prev_params"] = {
        **{name: config[name] for name in _BD_PARAMS},
        **{name: config[name] for name in _HB_PARAMS},
        **{name: config[name] for name in _RENDER_PARAMS},
    }
    log.info("Sonostar probe initialized")


@sonostar.cleanup
def sonostar_cleanup(state, config):
    log.info("closing Sonostar probe connection")
    for key in ("data_sock", "ctrl_sock"):
        sock = state.get(key)
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
