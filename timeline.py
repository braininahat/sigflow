"""Timeline panel — DAW-style session playback with transport and track lanes."""
from __future__ import annotations

import logging
import time

import numpy as np

from PySide6.QtCore import Qt, QTimer, QRectF, Signal
from PySide6.QtGui import QPainter, QColor, QPen, QFont, QFontMetrics, QBrush
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QSlider, QComboBox, QScrollBar,
)

log = logging.getLogger(__name__)


# Track colors by port type
_TRACK_COLORS = {
    "AudioSignal": QColor(80, 180, 80),
    "Keypoints": QColor(80, 130, 220),
    "TongueKeypoints": QColor(80, 130, 220),
    "LipKeypoints": QColor(100, 160, 220),
    "FaceLandmarks": QColor(140, 100, 220),
    "CameraFrame": QColor(200, 140, 60),
    "UltrasoundFrame": QColor(200, 140, 60),
    "Event": QColor(220, 80, 80),
    "Scalar": QColor(180, 180, 80),
}

TRACK_HEIGHT = 48
LABEL_WIDTH = 100
RULER_HEIGHT = 24
PLAYHEAD_COLOR = QColor(220, 50, 50)


def _build_waveform_overview(data: np.ndarray, target_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Downsample audio to min/max pairs per bin for fast waveform rendering."""
    n = len(data)
    if n == 0:
        return np.zeros(target_bins), np.zeros(target_bins)
    if data.ndim == 2:
        data = data[:, 0]
    bins = min(target_bins, n)
    chunk = max(1, n // bins)
    trimmed = data[:chunk * bins]
    reshaped = trimmed.reshape(bins, chunk)
    return reshaped.min(axis=1), reshaped.max(axis=1)


def _build_envelope(data: np.ndarray, target_bins: int) -> np.ndarray:
    """Compute magnitude envelope for keypoint/scalar tracks."""
    n = len(data)
    if n == 0:
        return np.zeros(target_bins)
    if data.ndim == 2:
        # RMS across channels per sample
        magnitudes = np.sqrt(np.mean(data ** 2, axis=1))
    else:
        magnitudes = np.abs(data)
    bins = min(target_bins, n)
    chunk = max(1, n // bins)
    trimmed = magnitudes[:chunk * bins]
    reshaped = trimmed.reshape(bins, chunk)
    return reshaped.max(axis=1)


def _format_time(seconds: float) -> str:
    """Format seconds as MM:SS.mmm."""
    m = int(seconds) // 60
    s = seconds - m * 60
    return f"{m:02d}:{s:06.3f}"


class _TrackData:
    """Pre-computed rendering data for one track."""
    def __init__(self, source_id: str, port_type: str, stream_id: int):
        self.source_id = source_id
        self.port_type = port_type
        self.stream_id = stream_id
        self.color = _TRACK_COLORS.get(port_type, QColor(150, 150, 150))
        # Populated after loading
        self.timestamps: np.ndarray = np.array([])
        self.waveform_min: np.ndarray | None = None
        self.waveform_max: np.ndarray | None = None
        self.envelope: np.ndarray | None = None
        self.event_labels: list[str] = []
        self.event_times: np.ndarray = np.array([])
        self.is_video = False
        self.frame_count: int = 0


class TimelineWidget(QWidget):
    """Custom-painted track lanes with time ruler, tracks, and playhead."""

    seek_requested = Signal(float)  # emits absolute timestamp

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tracks: list[_TrackData] = []
        self._t0: float = 0.0
        self._t1: float = 1.0
        self._playhead: float = 0.0
        # Zoom: visible time window
        self._view_start: float = 0.0
        self._view_duration: float = 1.0
        self.setMinimumHeight(RULER_HEIGHT + TRACK_HEIGHT)
        self.setMouseTracking(True)

    def set_tracks(self, tracks: list[_TrackData], t0: float, t1: float):
        self._tracks = tracks
        self._t0 = t0
        self._t1 = t1
        self._view_start = t0
        self._view_duration = t1 - t0
        self.setMinimumHeight(RULER_HEIGHT + max(1, len(tracks)) * TRACK_HEIGHT)
        self.update()

    def set_playhead(self, t: float):
        self._playhead = t
        self.update()

    def set_view(self, start: float, duration: float):
        self._view_start = start
        self._view_duration = max(0.01, duration)
        self.update()

    def _time_to_x(self, t: float) -> float:
        """Convert absolute timestamp to pixel x coordinate in track area."""
        track_width = self.width() - LABEL_WIDTH
        if self._view_duration <= 0 or track_width <= 0:
            return LABEL_WIDTH
        frac = (t - self._view_start) / self._view_duration
        return LABEL_WIDTH + frac * track_width

    def _x_to_time(self, x: float) -> float:
        """Convert pixel x to absolute timestamp."""
        track_width = self.width() - LABEL_WIDTH
        if track_width <= 0:
            return self._view_start
        frac = (x - LABEL_WIDTH) / track_width
        return self._view_start + frac * self._view_duration

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        track_w = w - LABEL_WIDTH

        # Background
        p.fillRect(0, 0, w, h, QColor(30, 30, 30))

        # Time ruler
        p.fillRect(LABEL_WIDTH, 0, track_w, RULER_HEIGHT, QColor(40, 40, 40))
        self._draw_ruler(p, track_w)

        # Tracks
        for i, track in enumerate(self._tracks):
            y = RULER_HEIGHT + i * TRACK_HEIGHT
            self._draw_track(p, track, y, track_w)

        # Playhead
        px = self._time_to_x(self._playhead)
        if LABEL_WIDTH <= px <= w:
            pen = QPen(PLAYHEAD_COLOR, 2)
            p.setPen(pen)
            p.drawLine(int(px), 0, int(px), h)

        p.end()

    def _draw_ruler(self, p: QPainter, track_w: int):
        """Draw time ruler with tick marks and labels."""
        p.setPen(QPen(QColor(160, 160, 160), 1))
        font = QFont("monospace", 8)
        p.setFont(font)
        fm = QFontMetrics(font)

        # Choose tick interval based on zoom level
        px_per_sec = track_w / self._view_duration if self._view_duration > 0 else 1
        # Want ticks roughly every 80px
        raw_interval = 80.0 / px_per_sec
        # Snap to nice intervals
        nice = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300]
        interval = nice[0]
        for n in nice:
            if n >= raw_interval:
                interval = n
                break

        t = (self._view_start // interval) * interval
        while t <= self._view_start + self._view_duration:
            x = self._time_to_x(t)
            if LABEL_WIDTH <= x <= LABEL_WIDTH + track_w:
                p.drawLine(int(x), RULER_HEIGHT - 6, int(x), RULER_HEIGHT)
                rel_t = t - self._t0
                label = _format_time(rel_t)
                p.drawText(int(x) + 2, RULER_HEIGHT - 4, label)
            t += interval

    def _draw_track(self, p: QPainter, track: _TrackData, y: int, track_w: int):
        """Draw a single track lane."""
        # Label area
        p.fillRect(0, y, LABEL_WIDTH, TRACK_HEIGHT, QColor(45, 45, 45))
        p.setPen(QPen(QColor(200, 200, 200), 1))
        font = QFont("sans-serif", 9)
        p.setFont(font)
        p.drawText(4, y + 4, LABEL_WIDTH - 8, TRACK_HEIGHT - 8,
                   Qt.AlignLeft | Qt.AlignVCenter, track.source_id)

        # Track lane background
        lane_rect = QRectF(LABEL_WIDTH, y, track_w, TRACK_HEIGHT)
        p.fillRect(lane_rect, QColor(35, 35, 35))

        # Separator line
        p.setPen(QPen(QColor(55, 55, 55), 1))
        p.drawLine(0, y + TRACK_HEIGHT, LABEL_WIDTH + track_w, y + TRACK_HEIGHT)

        # Draw track content
        color = track.color
        dim_color = QColor(color.red(), color.green(), color.blue(), 120)

        if track.waveform_min is not None and track.waveform_max is not None:
            self._draw_waveform(p, track, y, track_w, color)
        elif track.envelope is not None:
            self._draw_envelope(p, track, y, track_w, color)
        elif track.is_video:
            self._draw_video_bar(p, track, y, track_w, dim_color)
        elif len(track.event_times) > 0:
            self._draw_events(p, track, y, track_w, color)

    def _draw_waveform(self, p: QPainter, track: _TrackData, y: int, track_w: int, color: QColor):
        """Draw audio waveform (min/max envelope)."""
        n = len(track.waveform_min)
        if n == 0:
            return
        mid = y + TRACK_HEIGHT // 2
        half = (TRACK_HEIGHT - 4) / 2
        vmax = max(abs(track.waveform_min.min()), abs(track.waveform_max.max()), 1e-6)

        pen = QPen(color, 1)
        p.setPen(pen)

        ts = track.timestamps
        for i in range(n):
            # Map bin i to its time range in the track's timestamp domain
            frac = i / n
            bin_time = ts[0] + frac * (ts[-1] - ts[0]) if len(ts) > 1 else ts[0]
            x = self._time_to_x(bin_time)
            if x < LABEL_WIDTH or x > LABEL_WIDTH + track_w:
                continue
            ymin = mid - int(track.waveform_max[i] / vmax * half)
            ymax = mid - int(track.waveform_min[i] / vmax * half)
            p.drawLine(int(x), ymin, int(x), ymax)

    def _draw_envelope(self, p: QPainter, track: _TrackData, y: int, track_w: int, color: QColor):
        """Draw keypoint/scalar magnitude envelope."""
        n = len(track.envelope)
        if n == 0:
            return
        bottom = y + TRACK_HEIGHT - 2
        height = TRACK_HEIGHT - 4
        vmax = max(track.envelope.max(), 1e-6)

        brush = QBrush(QColor(color.red(), color.green(), color.blue(), 160))
        p.setPen(Qt.NoPen)
        p.setBrush(brush)

        ts = track.timestamps
        for i in range(n):
            frac = i / n
            bin_time = ts[0] + frac * (ts[-1] - ts[0]) if len(ts) > 1 else ts[0]
            x = self._time_to_x(bin_time)
            if x < LABEL_WIDTH or x > LABEL_WIDTH + track_w:
                continue
            bar_h = int(track.envelope[i] / vmax * height)
            p.drawRect(int(x), bottom - bar_h, max(1, track_w // n), bar_h)

    def _draw_video_bar(self, p: QPainter, track: _TrackData, y: int, track_w: int, color: QColor):
        """Draw a solid bar for video tracks."""
        if len(track.timestamps) < 2:
            return
        x0 = self._time_to_x(track.timestamps[0])
        x1 = self._time_to_x(track.timestamps[-1])
        x0 = max(LABEL_WIDTH, x0)
        x1 = min(LABEL_WIDTH + track_w, x1)
        if x1 > x0:
            p.fillRect(QRectF(x0, y + 4, x1 - x0, TRACK_HEIGHT - 8), color)
            p.setPen(QPen(QColor(255, 255, 255, 180), 1))
            font = QFont("sans-serif", 8)
            p.setFont(font)
            p.drawText(int(x0) + 4, y + 4, int(x1 - x0) - 8, TRACK_HEIGHT - 8,
                       Qt.AlignLeft | Qt.AlignVCenter,
                       f"{track.frame_count} frames")

    def _draw_events(self, p: QPainter, track: _TrackData, y: int, track_w: int, color: QColor):
        """Draw event markers as triangles."""
        pen = QPen(color, 2)
        p.setPen(pen)
        p.setBrush(QBrush(color))
        for t in track.event_times:
            x = self._time_to_x(t)
            if x < LABEL_WIDTH or x > LABEL_WIDTH + track_w:
                continue
            xi = int(x)
            # Small downward triangle
            p.drawPolygon([
                (xi - 4, y + 6),
                (xi + 4, y + 6),
                (xi, y + 14),
            ])
            # Vertical line
            p.drawLine(xi, y + 14, xi, y + TRACK_HEIGHT - 2)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and event.position().x() > LABEL_WIDTH:
            t = self._x_to_time(event.position().x())
            t = max(self._t0, min(self._t1, t))
            self.seek_requested.emit(t)

    def wheelEvent(self, event):
        """Scroll wheel zooms the time axis."""
        delta = event.angleDelta().y()
        if delta == 0:
            return
        # Zoom centered on mouse position
        mouse_t = self._x_to_time(event.position().x())
        factor = 0.8 if delta > 0 else 1.25
        new_dur = self._view_duration * factor
        # Clamp
        total = self._t1 - self._t0
        new_dur = max(0.1, min(total, new_dur))
        # Keep mouse position anchored
        mouse_frac = (event.position().x() - LABEL_WIDTH) / max(1, self.width() - LABEL_WIDTH)
        new_start = mouse_t - mouse_frac * new_dur
        new_start = max(self._t0, min(self._t1 - new_dur, new_start))
        self._view_start = new_start
        self._view_duration = new_dur
        self.update()


class TimelinePanel(QWidget):
    """Complete timeline panel: transport bar + track lanes + playback engine."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._reader = None
        self._tracks: list[_TrackData] = []
        self._t0: float = 0.0
        self._t1: float = 0.0
        self._current_time: float = 0.0
        self._playing: bool = False
        self._play_start_wall: float = 0.0
        self._play_start_session: float = 0.0
        self._speed: float = 1.0

        self._build_ui()

        # Playback timer
        self._playback_timer = QTimer(self)
        self._playback_timer.setInterval(16)  # ~60fps
        self._playback_timer.timeout.connect(self._tick)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Transport bar
        transport = QHBoxLayout()
        transport.setContentsMargins(8, 4, 8, 4)
        transport.setSpacing(8)

        self._play_btn = QPushButton("\u25b6")
        self._play_btn.setFixedSize(28, 28)
        self._play_btn.setToolTip("Play / Pause")
        self._play_btn.clicked.connect(self._toggle_play)
        transport.addWidget(self._play_btn)

        self._stop_btn = QPushButton("\u25a0")
        self._stop_btn.setFixedSize(28, 28)
        self._stop_btn.setToolTip("Stop (reset to start)")
        self._stop_btn.clicked.connect(self._stop)
        transport.addWidget(self._stop_btn)

        self._time_label = QLabel("00:00.000 / 00:00.000")
        self._time_label.setStyleSheet("font-family: monospace; color: #ccc;")
        transport.addWidget(self._time_label)

        self._seek_slider = QSlider(Qt.Horizontal)
        self._seek_slider.setRange(0, 10000)
        self._seek_slider.sliderMoved.connect(self._on_slider_seek)
        transport.addWidget(self._seek_slider, stretch=1)

        self._speed_combo = QComboBox()
        self._speed_combo.addItems(["0.25x", "0.5x", "1x", "2x", "4x"])
        self._speed_combo.setCurrentText("1x")
        self._speed_combo.currentTextChanged.connect(self._on_speed_changed)
        transport.addWidget(self._speed_combo)

        transport_widget = QWidget()
        transport_widget.setLayout(transport)
        transport_widget.setStyleSheet("background: #2a2a2a;")
        layout.addWidget(transport_widget)

        # Timeline tracks
        self._timeline = TimelineWidget()
        self._timeline.seek_requested.connect(self._seek_to)
        layout.addWidget(self._timeline, stretch=1)

    def load_session(self, reader):
        """Load a SessionReader and populate tracks."""
        self._reader = reader
        self._tracks.clear()
        self._stop()

        if not reader.streams:
            self._timeline.set_tracks([], 0, 1)
            return

        self._t0, self._t1 = reader.time_range
        self._current_time = self._t0

        for stream_info in reader.streams:
            track = _TrackData(
                source_id=stream_info.source_id,
                port_type=stream_info.port_type,
                stream_id=stream_info.stream_id,
            )

            if stream_info.filename:
                # Video track
                track.is_video = True
                track.frame_count = stream_info.frame_count or 0
                try:
                    ts, _ = reader.get_time_series(stream_info.stream_id)
                    track.timestamps = ts
                except KeyError:
                    pass
            else:
                try:
                    ts, data = reader.get_time_series(stream_info.stream_id)
                    track.timestamps = ts
                except KeyError:
                    continue

                if stream_info.port_type in ("AudioSignal",):
                    mn, mx = _build_waveform_overview(data, 2000)
                    track.waveform_min = mn
                    track.waveform_max = mx
                elif stream_info.port_type == "Event":
                    track.event_times = ts
                    if data.ndim == 2 and data.shape[1] >= 1:
                        track.event_labels = [str(row[0]) for row in data]
                    else:
                        track.event_labels = [str(d) for d in data]
                else:
                    # Keypoints, scalars, etc.
                    track.envelope = _build_envelope(data, 2000)

            self._tracks.append(track)

        self._timeline.set_tracks(self._tracks, self._t0, self._t1)
        self._update_time_display()
        log.info("timeline loaded: %d tracks, %.1fs duration",
                 len(self._tracks), self._t1 - self._t0)

    def _toggle_play(self):
        if self._playing:
            self._pause()
        else:
            self._play()

    def _play(self):
        if not self._reader:
            return
        self._playing = True
        self._play_start_wall = time.monotonic()
        self._play_start_session = self._current_time
        self._play_btn.setText("\u275a\u275a")
        self._playback_timer.start()

    def _pause(self):
        self._playing = False
        self._play_btn.setText("\u25b6")
        self._playback_timer.stop()

    def _stop(self):
        self._pause()
        self._current_time = self._t0
        self._timeline.set_playhead(self._current_time)
        self._update_time_display()

    def _tick(self):
        elapsed = (time.monotonic() - self._play_start_wall) * self._speed
        self._current_time = self._play_start_session + elapsed
        if self._current_time >= self._t1:
            self._current_time = self._t1
            self._pause()
        self._timeline.set_playhead(self._current_time)
        self._update_time_display()

    def _seek_to(self, t: float):
        was_playing = self._playing
        if was_playing:
            self._pause()
        self._current_time = max(self._t0, min(self._t1, t))
        self._timeline.set_playhead(self._current_time)
        self._update_time_display()
        if was_playing:
            self._play()

    def _on_slider_seek(self, value):
        if self._t1 <= self._t0:
            return
        frac = value / 10000.0
        t = self._t0 + frac * (self._t1 - self._t0)
        self._seek_to(t)

    def _on_speed_changed(self, text):
        self._speed = float(text.rstrip("x"))
        if self._playing:
            # Reset play anchor to current position
            self._play_start_wall = time.monotonic()
            self._play_start_session = self._current_time

    def _update_time_display(self):
        rel_current = self._current_time - self._t0
        rel_total = self._t1 - self._t0
        self._time_label.setText(f"{_format_time(rel_current)} / {_format_time(rel_total)}")
        # Update slider
        if rel_total > 0:
            frac = rel_current / rel_total
            self._seek_slider.blockSignals(True)
            self._seek_slider.setValue(int(frac * 10000))
            self._seek_slider.blockSignals(False)
