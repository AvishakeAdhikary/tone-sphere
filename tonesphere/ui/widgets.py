"""
Custom-painted mixer controls.

These are painted rather than styled because a stylesheet cannot express a dBFS scale, a
non-linear fader taper, or a peak-hold marker. That limitation is most of why the previous
Tkinter interface could not look like audio software.

Every widget here shows a real measurement or controls a real parameter. None of them is
decorative.
"""

import math
from typing import Optional

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush, QColor, QFontMetrics, QLinearGradient, QMouseEvent, QPainter, QPainterPath,
    QPen, QWheelEvent,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

from tonesphere.ui.theme import (
    METER_HIGH_DB, METER_MAX_DB, METER_MIN_DB, METER_PEAK_DB, METRICS,
    Colors, Spacing, Type, db_to_fader_position, db_to_fraction, fader_position_to_db,
    format_db, meter_color,
)


class LevelMeter(QWidget):
    """
    A dBFS level meter with peak hold and a latching clip indicator.

    Scaled in dB, not amplitude. An amplitude-linear meter puts half its length in the top
    6 dB and shows almost nothing across the range people actually work in.

    Peak hold exists because a transient shorter than one repaint is invisible otherwise,
    and a transient is exactly what makes a converter clip. The clip light latches for the
    same reason: a single over between two glances would go unnoticed.
    """

    clicked = Signal()

    # Ticks worth labelling. Enough to read the scale, few enough not to become texture.
    SCALE_TICKS = (0, -6, -12, -18, -24, -36, -48)

    def __init__(self, channels: int = 2, horizontal: bool = False,
                 show_scale: bool = True, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.channels = channels
        self.horizontal = horizontal
        self.show_scale = show_scale

        self._peaks = [METER_MIN_DB] * channels
        self._rms = [METER_MIN_DB] * channels
        self._holds = [METER_MIN_DB] * channels
        self._clipped = False
        self._active = False

        if horizontal:
            self.setMinimumHeight(channels * 6 + 4)
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        else:
            width = METRICS.METER_WIDTH * channels + 2
            if show_scale:
                width += 26
            self.setMinimumWidth(width)
            self.setMinimumHeight(80)
            self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        self.setToolTip("Click to clear the clip indicator")

    def set_levels(self, peaks, rms=None, holds=None, clipped: bool = False):
        """
        Update from the engine. Values in dBFS.

        Called at UI rate from a timer, not from the audio callback — the callback writes
        into plain slots and never touches Qt.
        """
        self._peaks = list(peaks)[:self.channels] or [METER_MIN_DB]
        self._rms = list(rms)[:self.channels] if rms else self._peaks
        self._holds = list(holds)[:self.channels] if holds else self._peaks
        self._clipped = clipped
        self._active = True
        self.update()

    def set_inactive(self):
        """
        Show "no signal path" rather than a confident silence.

        A meter pinned at the floor looks like measured silence. When nothing is running
        there is no measurement at all, and the meter greys out to say so.
        """
        self._active = False
        self._peaks = [METER_MIN_DB] * self.channels
        self._rms = list(self._peaks)
        self._holds = list(self._peaks)
        self.update()

    def clear_clip(self):
        self._clipped = False
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        self.clear_clip()
        self.clicked.emit()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        if self.horizontal:
            self._paint_horizontal(painter)
        else:
            self._paint_vertical(painter)

    def _paint_vertical(self, painter: QPainter):
        scale_width = 26 if self.show_scale else 0
        bar_area = self.width() - scale_width
        bar_width = max(4, (bar_area - 2) // self.channels - 1)

        # Reserve the top for the clip light so it cannot be confused with signal.
        clip_height = 6
        top = clip_height + 2
        height = self.height() - top - 2

        for index in range(self.channels):
            x = scale_width + index * (bar_width + 1)
            self._paint_bar(painter, QRectF(x, top, bar_width, height),
                            self._peaks[index] if index < len(self._peaks) else METER_MIN_DB,
                            self._rms[index] if index < len(self._rms) else METER_MIN_DB,
                            self._holds[index] if index < len(self._holds) else METER_MIN_DB)

        if self.show_scale:
            self._paint_scale(painter, top, height, scale_width)

        # Clip light spans the full meter width: it is about the device, not one channel.
        clip_rect = QRectF(scale_width, 0, bar_area - 1, clip_height)
        painter.fillRect(
            clip_rect,
            Colors.METER_PEAK if self._clipped else Colors.BG_RAISED,
        )

    def _paint_bar(self, painter: QPainter, rect: QRectF,
                   peak_db: float, rms_db: float, hold_db: float):
        painter.fillRect(rect, Colors.METER_BG)

        if not self._active:
            # Greyed out: nothing is being measured.
            painter.setPen(QPen(Colors.BORDER, 1))
            painter.drawRect(rect.adjusted(0, 0, -1, -1))
            return

        # RMS as the solid body, peak as a brighter cap. RMS tracks perceived loudness;
        # peak tracks what will clip. Showing both is why console meters have two.
        rms_fraction = db_to_fraction(rms_db)
        if rms_fraction > 0:
            rms_height = rect.height() * rms_fraction
            rms_rect = QRectF(rect.x(), rect.bottom() - rms_height,
                              rect.width(), rms_height)
            painter.fillRect(rms_rect, self._gradient(rect))

        peak_fraction = db_to_fraction(peak_db)
        if peak_fraction > 0:
            y = rect.bottom() - rect.height() * peak_fraction
            painter.fillRect(QRectF(rect.x(), y, rect.width(), 2), meter_color(peak_db))

        if hold_db > METER_MIN_DB:
            y = rect.bottom() - rect.height() * db_to_fraction(hold_db)
            painter.fillRect(QRectF(rect.x(), y, rect.width(), 1), Colors.METER_HOLD)

    def _gradient(self, rect: QRectF) -> QBrush:
        """
        Green through amber to red up the bar.

        The colour is a property of the level, not of the moment, so a glance at where the
        colour changes tells you the headroom without reading any number.
        """
        gradient = QLinearGradient(rect.x(), rect.bottom(), rect.x(), rect.y())
        gradient.setColorAt(0.0, Colors.METER_LOW)
        gradient.setColorAt(db_to_fraction(METER_HIGH_DB), Colors.METER_MID)
        gradient.setColorAt(db_to_fraction(METER_PEAK_DB), Colors.METER_HIGH)
        gradient.setColorAt(1.0, Colors.METER_PEAK)
        return QBrush(gradient)

    def _paint_scale(self, painter: QPainter, top: int, height: int, width: int):
        painter.setFont(Type.font(Type.TINY, mono=True))
        painter.setPen(QPen(Colors.TEXT_DIM, 1))
        metrics = QFontMetrics(painter.font())

        for db in self.SCALE_TICKS:
            if db < METER_MIN_DB:
                continue
            y = top + height - height * db_to_fraction(float(db))
            label = "0" if db == 0 else str(db)
            painter.drawText(
                QRectF(0, y - metrics.height() / 2, width - 4, metrics.height()),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                label,
            )

    def _paint_horizontal(self, painter: QPainter):
        bar_height = max(3, (self.height() - 2) // self.channels - 1)

        for index in range(self.channels):
            y = index * (bar_height + 1)
            rect = QRectF(0, y, self.width(), bar_height)
            painter.fillRect(rect, Colors.METER_BG)

            if not self._active:
                continue

            peak_db = self._peaks[index] if index < len(self._peaks) else METER_MIN_DB
            fraction = db_to_fraction(peak_db)
            if fraction > 0:
                gradient = QLinearGradient(rect.x(), 0, rect.right(), 0)
                gradient.setColorAt(0.0, Colors.METER_LOW)
                gradient.setColorAt(db_to_fraction(METER_HIGH_DB), Colors.METER_MID)
                gradient.setColorAt(1.0, Colors.METER_PEAK)
                painter.fillRect(
                    QRectF(rect.x(), rect.y(), rect.width() * fraction, rect.height()),
                    QBrush(gradient),
                )


class Fader(QWidget):
    """
    A gain fader with a console-style taper and a dB readout.

    The taper matters: linear-in-dB travel gives a couple of pixels per dB near unity,
    which is exactly where fine adjustment happens. `theme.fader_position_to_db` expands
    the top of the travel the way a real fader's law does.

    Fine mode (hold Shift) divides the movement by ten, because a mouse cannot otherwise
    resolve 0.1 dB.
    """

    value_changed = Signal(float)   # dB

    def __init__(self, initial_db: float = 0.0, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._db = initial_db
        self._dragging = False
        self._drag_origin = 0.0
        self._drag_start_db = 0.0

        self.setMinimumSize(28, 120)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setToolTip("Drag to set gain · Shift for fine · double-click for unity")

    @property
    def db(self) -> float:
        return self._db

    def set_db(self, db: float, notify: bool = False):
        clamped = min(max(db, -60.0), 12.0)
        if clamped == self._db:
            return
        self._db = clamped
        self.update()
        if notify:
            self.value_changed.emit(self._db)

    # --- Interaction ---

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag_origin = event.position().y()
            self._drag_start_db = self._db

    def mouseMoveEvent(self, event: QMouseEvent):
        if not self._dragging:
            return

        travel = self._track_rect().height()
        if travel <= 0:
            return

        delta = self._drag_origin - event.position().y()
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            delta *= 0.1

        position = db_to_fader_position(self._drag_start_db) + delta / travel
        self.set_db(fader_position_to_db(position), notify=True)

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._dragging = False

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """Unity. The single most common thing to want a fader to be."""
        self.set_db(0.0, notify=True)

    def wheelEvent(self, event: QWheelEvent):
        step = 0.1 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1.0
        self.set_db(self._db + (step if event.angleDelta().y() > 0 else -step), notify=True)

    def keyPressEvent(self, event):
        """Keyboard control, so the mixer is usable without a mouse."""
        step = 0.1 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1.0

        if event.key() == Qt.Key.Key_Up:
            self.set_db(self._db + step, notify=True)
        elif event.key() == Qt.Key.Key_Down:
            self.set_db(self._db - step, notify=True)
        elif event.key() in (Qt.Key.Key_0, Qt.Key.Key_Home):
            self.set_db(0.0, notify=True)
        else:
            super().keyPressEvent(event)

    # --- Painting ---

    def _track_rect(self) -> QRectF:
        readout = 16
        margin = 8
        return QRectF(self.width() / 2 - 3, margin,
                      6, self.height() - readout - margin * 2)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        track = self._track_rect()

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(Colors.BG_SUNKEN)
        painter.drawRoundedRect(track, 3, 3)

        position = db_to_fader_position(self._db)
        y = track.bottom() - track.height() * position

        # Fill from unity, not from the bottom: at a glance you see whether the fader is
        # cutting or boosting, which is the question you actually have.
        unity_y = track.bottom() - track.height() * db_to_fader_position(0.0)
        painter.setBrush(Colors.ACCENT if self._db > 0 else Colors.ACCENT_DIM)
        if y < unity_y:
            painter.drawRoundedRect(QRectF(track.x(), y, track.width(), unity_y - y), 3, 3)
        else:
            painter.drawRoundedRect(QRectF(track.x(), unity_y, track.width(), y - unity_y), 3, 3)

        # Unity mark, so 0 dB is findable without reading the number.
        painter.setPen(QPen(Colors.TEXT_DIM, 1))
        painter.drawLine(QPointF(track.x() - 5, unity_y), QPointF(track.right() + 5, unity_y))

        self._paint_cap(painter, y)
        self._paint_readout(painter)

    def _paint_cap(self, painter: QPainter, y: float):
        cap = QRectF(self.width() / 2 - 11, y - 6, 22, 12)

        painter.setPen(QPen(Colors.BORDER_STRONG, 1))
        painter.setBrush(Colors.BG_HOVER if self._dragging else Colors.BG_RAISED)
        painter.drawRoundedRect(cap, 3, 3)

        # A grip line down the middle of the cap, as on a physical fader.
        painter.setPen(QPen(Colors.TEXT_DIM, 1))
        painter.drawLine(QPointF(cap.x() + 4, y), QPointF(cap.right() - 4, y))

    def _paint_readout(self, painter: QPainter):
        painter.setFont(Type.numeric(Type.TINY))
        painter.setPen(Colors.TEXT if self._db != 0.0 else Colors.TEXT_MUTED)
        painter.drawText(
            QRectF(0, self.height() - 15, self.width(), 14),
            Qt.AlignmentFlag.AlignCenter,
            format_db(self._db),
        )


class PanKnob(QWidget):
    """
    A pan control.

    Drawn as an arc rather than a rotating pointer because the arc shows *how far* from
    centre at a glance, and centre is the reference position you keep returning to.
    """

    value_changed = Signal(float)   # -1..1

    def __init__(self, initial: float = 0.0, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._pan = initial
        self._dragging = False
        self._drag_origin = 0.0
        self._drag_start = 0.0

        # Height leaves room for the arc plus a readout line beneath it, so the label
        # never sits on top of the arc.
        self._label_height = 12
        self.setFixedSize(METRICS.KNOB_SIZE, METRICS.KNOB_SIZE + self._label_height)
        self.setCursor(Qt.CursorShape.SizeHorCursor)
        self.setToolTip("Drag to pan · Shift for fine · double-click to centre")

    @property
    def pan(self) -> float:
        return self._pan

    def set_pan(self, pan: float, notify: bool = False):
        clamped = min(max(pan, -1.0), 1.0)
        if clamped == self._pan:
            return
        self._pan = clamped
        self.update()
        if notify:
            self.value_changed.emit(self._pan)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag_origin = event.position().x()
            self._drag_start = self._pan

    def mouseMoveEvent(self, event: QMouseEvent):
        if not self._dragging:
            return
        delta = (event.position().x() - self._drag_origin) / 60.0
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            delta *= 0.2
        self.set_pan(self._drag_start + delta, notify=True)

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._dragging = False

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        self.set_pan(0.0, notify=True)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        inset = 4
        diameter = min(self.width(), self.height() - self._label_height) - inset * 2
        rect = QRectF((self.width() - diameter) / 2, inset, diameter, diameter)

        # 270 degrees of travel with the gap at the bottom. The gap is what makes hard
        # left and hard right visually unambiguous instead of meeting in a full circle.
        span = 270.0
        top = 90.0                          # Qt measures from 3 o'clock, counter-clockwise
        start = top + span / 2.0

        painter.setPen(QPen(Colors.BG_SUNKEN, 4, Qt.PenStyle.SolidLine,
                            Qt.PenCapStyle.RoundCap))
        painter.drawArc(rect, int(start * 16), int(-span * 16))

        if self._pan != 0.0:
            # Sweep out from top-centre, so the arc length reads directly as "how far
            # off centre" without having to interpret a pointer angle.
            painter.setPen(QPen(Colors.ACCENT, 4, Qt.PenStyle.SolidLine,
                                Qt.PenCapStyle.RoundCap))
            painter.drawArc(rect, int(top * 16), int(-self._pan * span / 2.0 * 16))

        # Centre detent notch.
        painter.setPen(QPen(Colors.TEXT_DIM, 1))
        painter.drawLine(QPointF(rect.center().x(), rect.top() - 2),
                         QPointF(rect.center().x(), rect.top() + 4))

        painter.setFont(Type.numeric(Type.TINY))
        painter.setPen(Colors.TEXT_MUTED if self._pan == 0 else Colors.TEXT)
        painter.drawText(
            QRectF(0, self.height() - self._label_height, self.width(), self._label_height),
            Qt.AlignmentFlag.AlignCenter,
            self._label(),
        )

    def _label(self) -> str:
        if self._pan == 0.0:
            return "C"
        side = "L" if self._pan < 0 else "R"
        return f"{side}{abs(self._pan) * 100:.0f}"


class StatusPill(QWidget):
    """
    A labelled readout for the hardware bar.

    Exists so that a value which is not measured can render as "--" in a dimmed colour
    rather than as a confident zero. That distinction is the whole reason this widget is
    not just a QLabel.
    """

    def __init__(self, label: str, value: str = "--", parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._label = label
        self._value = value
        self._tone: Optional[QColor] = None

        self.setFixedHeight(METRICS.STATUSBAR_HEIGHT - 6)
        self._resize_to_fit()

    def set_value(self, value: Optional[str], tone: Optional[QColor] = None):
        self._value = value if value is not None else "--"
        self._tone = tone
        self._resize_to_fit()
        self.update()

    def _resize_to_fit(self):
        metrics = QFontMetrics(Type.numeric(Type.SMALL))
        label_metrics = QFontMetrics(Type.font(Type.TINY))
        width = (label_metrics.horizontalAdvance(self._label)
                 + metrics.horizontalAdvance(self._value) + Spacing.XL)
        self.setMinimumWidth(width)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        painter.setFont(Type.font(Type.TINY))
        painter.setPen(Colors.TEXT_DIM)
        label_width = QFontMetrics(painter.font()).horizontalAdvance(self._label)
        painter.drawText(QRectF(0, 0, label_width, self.height()),
                         Qt.AlignmentFlag.AlignVCenter, self._label)

        painter.setFont(Type.numeric(Type.SMALL))
        if self._tone is not None:
            painter.setPen(self._tone)
        elif self._value == "--":
            # Dimmed, because it is the absence of a measurement rather than a value.
            painter.setPen(Colors.TEXT_DIM)
        else:
            painter.setPen(Colors.TEXT)

        painter.drawText(
            QRectF(label_width + Spacing.MD, 0, self.width() - label_width - Spacing.MD,
                   self.height()),
            Qt.AlignmentFlag.AlignVCenter, self._value,
        )
