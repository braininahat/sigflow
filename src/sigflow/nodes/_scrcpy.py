"""Shared scrcpy process management utilities for source nodes."""
import atexit
import logging
import os
import signal
import subprocess
import time

import cv2

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Persistent elevated V4L2 helper
# ---------------------------------------------------------------------------
# A single pkexec prompt launches a bash process that stays alive for the
# session.  All modprobe / v4l2loopback-ctl operations go through it so the
# user only authenticates once.

_V4L2_HELPER = r"""
while IFS='|' read -r cmd arg1 arg2 arg3; do
  case "$cmd" in
    modprobe)  out=$(modprobe v4l2loopback 2>&1); printf '%d|%s\n' "$?" "$out" ;;
    add)       out=$(v4l2loopback-ctl add -n "$arg1" -x "$arg2" "$arg3" 2>&1); printf '%d|%s\n' "$?" "$out" ;;
    delete)    out=$(v4l2loopback-ctl delete "$arg1" 2>&1); printf '%d|%s\n' "$?" "$out" ;;
    quit)      exit 0 ;;
  esac
done
"""

_v4l2_proc: subprocess.Popen | None = None


def _v4l2_cmd(cmd: str, *args: str) -> str:
    """Send a command to the persistent elevated V4L2 helper.

    Launches the helper on first call (single pkexec prompt).
    Returns command output on success, raises CalledProcessError on failure.
    """
    global _v4l2_proc
    if _v4l2_proc is None or _v4l2_proc.poll() is not None:
        log.info("starting elevated V4L2 helper (one-time authentication)")
        _v4l2_proc = subprocess.Popen(
            ["pkexec", "/bin/bash", "-c", _V4L2_HELPER],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        atexit.register(shutdown_v4l2)

    line = "|".join([cmd, *args]) + "\n"
    log.debug("v4l2 helper << %s", line.strip())
    _v4l2_proc.stdin.write(line.encode())
    _v4l2_proc.stdin.flush()

    response = _v4l2_proc.stdout.readline().decode().strip()
    if not response:
        rc = _v4l2_proc.poll()
        _v4l2_proc = None
        raise subprocess.CalledProcessError(
            rc or 1, f"v4l2 {cmd}",
            output="V4L2 helper not running (authentication denied?)",
        )

    log.debug("v4l2 helper >> %s", response)
    rc_str, _, msg = response.partition("|")
    rc = int(rc_str)
    if rc != 0:
        raise subprocess.CalledProcessError(rc, f"v4l2 {cmd}", output=msg)
    return msg


def shutdown_v4l2() -> None:
    """Shut down the persistent elevated V4L2 helper."""
    global _v4l2_proc
    if _v4l2_proc is None or _v4l2_proc.poll() is not None:
        _v4l2_proc = None
        return
    log.info("shutting down V4L2 helper")
    try:
        _v4l2_proc.stdin.write(b"quit\n")
        _v4l2_proc.stdin.flush()
        _v4l2_proc.wait(timeout=3.0)
    except Exception:
        _v4l2_proc.kill()
        _v4l2_proc.wait()
    _v4l2_proc = None


def launch_scrcpy(args: list[str], scrcpy_path: str = "scrcpy") -> subprocess.Popen:
    """Launch a scrcpy process with the given arguments.

    Adds --no-window automatically.  Returns the Popen handle.
    """
    cmd = [scrcpy_path, "--no-window"] + args
    log.info("launching scrcpy: %s", " ".join(cmd))
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def drain_output(proc: subprocess.Popen) -> str:
    """Non-blocking read of all available stdout from a process."""
    if proc.stdout is None:
        return ""
    # If process has exited, read everything remaining
    if proc.poll() is not None:
        return proc.stdout.read().decode(errors="replace")
    # Otherwise read whatever is buffered (non-blocking via select)
    import select
    output = []
    while select.select([proc.stdout], [], [], 0)[0]:
        chunk = proc.stdout.read1(4096) if hasattr(proc.stdout, 'read1') else proc.stdout.read(4096)
        if not chunk:
            break
        output.append(chunk.decode(errors="replace"))
    return "".join(output)


_V4L2_MIN_DEVICE = 20  # start above typical hardware devices (0-9)


def device_index(path: str) -> int:
    """Parse /dev/videoN → N as integer for cv2.VideoCapture."""
    return int(path.rsplit("video", 1)[1])


def _find_next_video_device() -> int:
    """Find the next available /dev/video device number."""
    used = set()
    for name in os.listdir("/dev"):
        if name.startswith("video") and name[5:].isdigit():
            used.add(int(name[5:]))
    log.info("existing video devices: %s", sorted(used) if used else "(none)")
    n = _V4L2_MIN_DEVICE
    while n in used:
        n += 1
    log.info("next available device number: %d", n)
    return n


def create_v4l2_device(label: str) -> str:
    """Create a V4L2 loopback device with the given label.

    Ensures the v4l2loopback module is loaded, then uses
    v4l2loopback-ctl to dynamically add a device.
    Returns the device path (e.g. /dev/video20).
    """
    if not os.path.isdir("/sys/module/v4l2loopback"):
        log.info("v4l2loopback module not loaded, loading via helper...")
        _v4l2_cmd("modprobe")
        log.info("v4l2loopback module loaded")
    else:
        log.info("v4l2loopback module already loaded")

    num = _find_next_video_device()
    device_path = f"/dev/video{num}"
    log.info("creating V4L2 loopback: %s (%s)", device_path, label)
    _v4l2_cmd("add", label, "1", device_path)
    log.info("V4L2 loopback device created: %s (%s)", device_path, label)
    return device_path


def destroy_v4l2_device(device_path: str) -> None:
    """Remove a V4L2 loopback device."""
    log.info("removing V4L2 loopback: %s", device_path)
    try:
        _v4l2_cmd("delete", device_path)
        log.info("V4L2 device %s removed", device_path)
    except subprocess.CalledProcessError:
        log.warning("failed to remove V4L2 device %s (may already be gone)", device_path)


def wait_for_capture(
    device_path: str,
    proc: subprocess.Popen,
    timeout: float = 15.0,
) -> cv2.VideoCapture | None:
    """Retry cv2.VideoCapture until scrcpy starts writing to the V4L2 device.

    The loopback device is output-only until scrcpy connects, so
    cv2.VideoCapture will fail until then.  This retries with crash
    detection and verbose logging.

    Returns an opened VideoCapture on success, None on timeout or crash.
    """
    idx = device_index(device_path)
    log.info("waiting for capture on %s (device index %d, timeout=%.1fs)", device_path, idx, timeout)
    deadline = time.monotonic() + timeout
    attempt = 0

    while time.monotonic() < deadline:
        # Check if scrcpy crashed
        if proc.poll() is not None:
            output = drain_output(proc)
            log.error(
                "scrcpy exited with code %d before capture ready:\n%s",
                proc.returncode, output,
            )
            return None

        attempt += 1
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if cap.isOpened():
            log.info("capture opened on attempt %d (%s)", attempt, device_path)
            return cap
        cap.release()

        if attempt % 10 == 0:
            log.info("still waiting for capture... (attempt %d)", attempt)

        time.sleep(0.25)

    # Timeout
    output = drain_output(proc)
    log.error(
        "capture on %s not ready after %.1fs (%d attempts). scrcpy output:\n%s",
        device_path, timeout, attempt, output or "(no output)",
    )
    return None


def kill_scrcpy(proc: subprocess.Popen) -> None:
    """Gracefully terminate a scrcpy process (SIGTERM → wait → SIGKILL)."""
    if proc.poll() is not None:
        return
    log.info("terminating scrcpy (pid=%d)", proc.pid)
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        log.warning("scrcpy did not exit, sending SIGKILL")
        proc.kill()
        proc.wait()
