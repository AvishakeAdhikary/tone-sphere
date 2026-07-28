"""
Mixer strips and the hardware status bar.

A strip is the unit of a mixer: one source or destination, with everything that can be
done to it in a fixed vertical order. The order is not arbitrary — it matches signal flow
(trim, then pan, then mute/solo, then fader, then meter), which is the order it appears on
every console, so it can be read without learning anything.
"""

from typing import Callable, Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFontMetrics
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from tonesphere.ui.theme import METRICS, Colors, Spacing, Type, format_db
from tonesphere.ui.widgets import Fader, LevelMeter, PanKnob, StatusPill


def elide(text: str, width: int, font) -> str:
    return QFontMetrics(font).elidedText(text, Qt.TextElideMode.ElideRight, width)


class ChannelStripWidget(QFrame):
    """
    One mixer channel.

    Emits intent; it does not touch the engine. The window owns that connection, so the
    strip stays testable and reusable for both hardware devices and buses.
    """

    gain_changed = Signal(int, float)      # device_id, dB
    pan_changed = Signal(int, float)       # device_id, -1..1
    mute_toggled = Signal(int, bool)
    solo_toggled = Signal(int, bool)
    selected = Signal(int)

    def __init__(self, device_id: int, name: str, subtitle: str, channels: int = 2,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.device_id = device_id
        self.channels = channels
        self._name = name

        self.setObjectName("Panel")
        self.setFixedWidth(METRICS.STRIP_WIDTH)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(Spacing.MD, Spacing.MD, Spacing.MD, Spacing.MD)
        layout.setSpacing(Spacing.MD)

        layout.addWidget(self._build_header(name, subtitle))
        layout.addWidget(self._build_pan(), alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addLayout(self._build_buttons())
        layout.addLayout(self._build_fader_and_meter(), stretch=1)

        self.setToolTip(f"{name}\n{subtitle}")

    # --- Construction ---

    def _build_header(self, name: str, subtitle: str) -> QWidget:
        header = QWidget()
        layout = QVBoxLayout(header)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Elide against the space actually inside the strip's margins, and measure with
        # the same font the stylesheet will render — see the StripName rule in theme.py.
        self._caption_width = METRICS.STRIP_WIDTH - Spacing.MD * 2 - Spacing.SM

        self.name_label = QLabel()
        self.name_label.setObjectName("StripName")
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.name_label.setFixedWidth(self._caption_width)
        layout.addWidget(self.name_label, alignment=Qt.AlignmentFlag.AlignHCenter)

        self.subtitle_label = QLabel()
        self.subtitle_label.setObjectName("StripSub")
        self.subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle_label.setFixedWidth(self._caption_width)
        layout.addWidget(self.subtitle_label, alignment=Qt.AlignmentFlag.AlignHCenter)

        self._name_text = name
        self._subtitle_text = subtitle

        return header

    def showEvent(self, event):
        """
        Elide once the stylesheet has been applied.

        Font metrics taken before the widget is shown use the default font, not the one the
        stylesheet specifies, so the measurement would be wrong and long names would spill
        past the strip edges.
        """
        super().showEvent(event)
        self._apply_captions()

    def _apply_captions(self):
        self.name_label.setText(
            elide(self._name_text, self._caption_width, self.name_label.font())
        )
        if not self.subtitle_label.styleSheet():
            self.subtitle_label.setText(
                elide(self._subtitle_text, self._caption_width, self.subtitle_label.font())
            )

    def _build_pan(self) -> QWidget:
        self.pan_knob = PanKnob()
        self.pan_knob.value_changed.connect(
            lambda value: self.pan_changed.emit(self.device_id, value)
        )
        return self.pan_knob

    def _build_buttons(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(Spacing.SM)

        self.mute_button = QPushButton("M")
        self.mute_button.setObjectName("Mute")
        self.mute_button.setCheckable(True)
        self.mute_button.setFixedHeight(METRICS.BUTTON_HEIGHT)
        self.mute_button.setToolTip("Mute this channel")
        self.mute_button.toggled.connect(
            lambda on: self.mute_toggled.emit(self.device_id, on)
        )

        self.solo_button = QPushButton("S")
        self.solo_button.setObjectName("Solo")
        self.solo_button.setCheckable(True)
        self.solo_button.setFixedHeight(METRICS.BUTTON_HEIGHT)
        self.solo_button.setToolTip("Solo: silence everything not soloed")
        self.solo_button.toggled.connect(
            lambda on: self.solo_toggled.emit(self.device_id, on)
        )

        row.addWidget(self.mute_button)
        row.addWidget(self.solo_button)
        return row

    def _build_fader_and_meter(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(Spacing.SM)

        self.fader = Fader()
        self.fader.value_changed.connect(
            lambda db: self.gain_changed.emit(self.device_id, db)
        )

        self.meter = LevelMeter(channels=self.channels, show_scale=False)

        row.addWidget(self.fader, alignment=Qt.AlignmentFlag.AlignHCenter)
        row.addWidget(self.meter)
        return row

    # --- State ---

    def set_levels(self, peaks, rms=None, holds=None, clipped: bool = False):
        self.meter.set_levels(peaks, rms, holds, clipped)

    def set_inactive(self):
        self.meter.set_inactive()

    def set_gain_db(self, db: float):
        self.fader.set_db(db)

    def set_pan(self, pan: float):
        self.pan_knob.set_pan(pan)

    def set_muted(self, muted: bool):
        self.mute_button.blockSignals(True)
        self.mute_button.setChecked(muted)
        self.mute_button.blockSignals(False)

    def set_soloed(self, soloed: bool):
        self.solo_button.blockSignals(True)
        self.solo_button.setChecked(soloed)
        self.solo_button.blockSignals(False)

    def set_failed(self, reason: Optional[str]):
        """
        Mark a strip whose device would not open.

        Without this the strip looks normal and the user has no way to tell that its audio
        is going nowhere — which is exactly the class of silent failure this project was
        full of.
        """
        if reason:
            self.setStyleSheet(
                f"QFrame#Panel {{ border: 1px solid {Colors.ERROR.name()}; }}"
            )
            self.subtitle_label.setText("device unavailable")
            self.subtitle_label.setStyleSheet(f"color: {Colors.ERROR.name()};")
            self.setToolTip(f"{self._name}\n\nFAILED: {reason}")
        else:
            self.setStyleSheet("")
            self.subtitle_label.setStyleSheet("")

    def mousePressEvent(self, event):
        self.selected.emit(self.device_id)
        super().mousePressEvent(event)


class MasterStrip(QFrame):
    """
    The master strip: overall output level and the state of the whole engine.

    Wider than a channel strip and visually separated, because it is not one more channel
    — it applies to everything.
    """

    gain_changed = Signal(float)
    clip_cleared = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.setObjectName("Panel")
        self.setFixedWidth(METRICS.STRIP_WIDTH_WIDE)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(Spacing.LG, Spacing.MD, Spacing.LG, Spacing.MD)
        layout.setSpacing(Spacing.MD)

        title = QLabel("MASTER")
        title.setFont(Type.font(Type.SMALL, weight=700))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(f"color: {Colors.ACCENT.name()}; letter-spacing: 1px;")
        layout.addWidget(title)

        self.limiter_label = QLabel("limiter idle")
        self.limiter_label.setObjectName("Dim")
        self.limiter_label.setFont(Type.numeric(Type.TINY))
        self.limiter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.limiter_label.setToolTip(
            "Gain reduction protecting the converter from a summing over"
        )
        layout.addWidget(self.limiter_label)

        row = QHBoxLayout()
        row.setSpacing(Spacing.MD)

        self.fader = Fader()
        self.fader.value_changed.connect(self.gain_changed)

        self.meter = LevelMeter(channels=2, show_scale=True)
        self.meter.clicked.connect(self.clip_cleared)

        row.addWidget(self.fader, alignment=Qt.AlignmentFlag.AlignHCenter)
        row.addWidget(self.meter)
        layout.addLayout(row, stretch=1)

    def set_levels(self, peaks, rms=None, holds=None, clipped: bool = False):
        self.meter.set_levels(peaks, rms, holds, clipped)

    def set_inactive(self):
        self.meter.set_inactive()

    def set_limiter_reduction(self, db: Optional[float]):
        if db is None or db >= -0.05:
            self.limiter_label.setText("limiter idle")
            self.limiter_label.setStyleSheet(f"color: {Colors.TEXT_DIM.name()};")
        else:
            self.limiter_label.setText(f"limiting {db:.1f} dB")
            self.limiter_label.setStyleSheet(f"color: {Colors.WARN.name()};")


class HardwareBar(QFrame):
    """
    The status bar: what the engine is actually doing, measured.

    Every field here is either a real measurement or "--". The old UI's equivalent showed
    "CPU: 0% | Latency: 0ms" permanently, which was not a reading of anything.

    Latency shows the measured round trip with the nominal figure beside it, because those
    two numbers differ by a factor of four on shared-mode WASAPI and the difference is the
    single most useful thing a latency-sensitive user can know.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.setObjectName("Panel")
        self.setFixedHeight(METRICS.STATUSBAR_HEIGHT + Spacing.MD)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(Spacing.LG, Spacing.SM, Spacing.LG, Spacing.SM)
        layout.setSpacing(Spacing.XL)

        self.state = QLabel("Stopped")
        self.state.setFont(Type.font(Type.SMALL, weight=600))
        layout.addWidget(self.state)

        layout.addWidget(self._separator())

        self.backend = StatusPill("BACKEND")
        self.rate = StatusPill("RATE")
        self.buffer = StatusPill("BUFFER")
        self.latency = StatusPill("LATENCY")
        self.load = StatusPill("DSP")
        self.xruns = StatusPill("XRUNS")

        for pill in (self.backend, self.rate, self.buffer, self.latency,
                     self.load, self.xruns):
            layout.addWidget(pill)

        layout.addStretch()

        self.message = QLabel("")
        self.message.setObjectName("Dim")
        self.message.setFont(Type.font(Type.SMALL))
        layout.addWidget(self.message)

    def _separator(self) -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        line.setStyleSheet(f"color: {Colors.BORDER.name()};")
        return line

    def update_state(self, state: str, stats: Dict):
        """Reflect engine state and measurements. Called on a UI timer."""
        labels = {
            'running': ("Running", Colors.OK),
            'degraded': ("Degraded", Colors.ERROR),
            'idle': ("Idle — nothing patched", Colors.WARN),
            'stopped': ("Stopped", Colors.TEXT_MUTED),
        }
        text, colour = labels.get(state, ("Unknown", Colors.TEXT_MUTED))
        self.state.setText(text)
        self.state.setStyleSheet(f"color: {colour.name()};")

        self.backend.set_value(stats.get('host_api'))

        rate = stats.get('samplerate')
        self.rate.set_value(f"{rate / 1000:.1f} kHz" if rate else None)

        blocksize = stats.get('blocksize')
        exclusive = " excl" if stats.get('exclusive') else ""
        self.buffer.set_value(f"{blocksize}{exclusive}" if blocksize else None)

        self._update_latency(stats)
        self._update_load(stats)
        self._update_xruns(stats)

        failed = stats.get('failed_streams') or {}
        if failed:
            first = next(iter(failed.values()))
            self.message.setText(first[:110])
            self.message.setStyleSheet(f"color: {Colors.ERROR.name()};")
        elif stats.get('problems'):
            self.message.setText(str(stats['problems'][0])[:110])
            self.message.setStyleSheet(f"color: {Colors.WARN.name()};")
        else:
            self.message.setText("")

    def _update_latency(self, stats: Dict):
        measured = stats.get('measured_latency_ms')
        nominal = stats.get('nominal_latency_ms')

        if measured is None:
            # Never show the nominal figure alone as if it were the real one.
            self.latency.set_value(
                f"-- ({nominal:.1f} nom)" if nominal else None
            )
            return

        tone = Colors.OK if measured < 15 else (
            Colors.WARN if measured < 40 else Colors.ERROR
        )
        text = f"{measured:.1f} ms"
        if nominal:
            text += f" ({nominal:.1f} nom)"
        self.latency.set_value(text, tone)

    def _update_load(self, stats: Dict):
        load = stats.get('cpu_usage')
        if load is None:
            self.load.set_value(None)
            return

        # Above ~80% of the block period, dropouts are imminent.
        tone = Colors.OK if load < 50 else (Colors.WARN if load < 80 else Colors.ERROR)
        self.load.set_value(f"{load:.0f}%", tone)

    def _update_xruns(self, stats: Dict):
        xruns = stats.get('xruns')
        if xruns is None:
            self.xruns.set_value(None)
            return
        self.xruns.set_value(str(xruns), Colors.OK if xruns == 0 else Colors.ERROR)
