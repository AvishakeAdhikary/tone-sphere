"""
An on-screen keyboard for an instrument: two octaves, playable with the mouse or with the
computer keyboard (A W S E D F T G Y H U J K for one octave, Z and X to shift the range).

It sends notes through the engine worker (`ui/tasks.py`), so a note is never played from
the main thread; the plugin receives it on its next block through the native MIDI queue.
"""

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from tonesphere.i18n import tr
from tonesphere.ui.tasks import EngineTasks
from tonesphere.ui.theme import Colors, Spacing

WHITE = (0, 2, 4, 5, 7, 9, 11)
BLACK = {1: 0, 3: 1, 6: 3, 8: 4, 10: 5}      # semitone -> the white key it sits after
KEYS = 'awsedftgyhujk'                         # C to C, the usual computer-keyboard layout
NAMES = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')


def note_name(note: int) -> str:
    return f"{NAMES[note % 12]}{note // 12 - 1}"


class PianoKeys(QWidget):
    """Two octaves from `low`. Emits `pressed(note)` and `released(note)`."""

    pressed = Signal(int)
    released = Signal(int)

    def __init__(self, low: int = 60, parent: QWidget | None = None):
        super().__init__(parent)
        self.low = low
        self.down: set[int] = set()
        self._mouse_note: int | None = None
        self.setMinimumSize(560, 140)

    def _geometry(self):
        whites = [self.low + 12 * o + w for o in range(2) for w in WHITE] + [self.low + 24]
        width = self.width() / len(whites)
        keys = [(n, QRectF(i * width, 0, width, self.height()), False) for i, n in enumerate(whites)]
        for i, n in enumerate(whites[:-1]):
            if (n + 1) % 12 in BLACK:
                keys.append((n + 1, QRectF((i + 0.65) * width, 0, width * 0.7, self.height() * 0.6), True))
        return keys

    def note_at(self, x: float, y: float) -> int | None:
        found = None
        for note, rect, black in self._geometry():
            if rect.contains(x, y) and (black or found is None):
                found = note
        return found

    def paintEvent(self, _event):
        painter = QPainter(self)
        for black in (False, True):
            for note, rect, is_black in self._geometry():
                if is_black != black:
                    continue
                fill = Colors.ACCENT if note in self.down else (QColor('#1b1d22') if black else QColor('#e8e8e8'))
                painter.fillRect(rect.adjusted(1, 0, -1, -1), fill)
                if not black and note % 12 == 0:
                    painter.setPen(QColor('#555'))
                    painter.drawText(rect.adjusted(0, 0, 0, -6),
                                     Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom,
                                     note_name(note))

    def mousePressEvent(self, event):
        note = self.note_at(event.position().x(), event.position().y())
        if note is not None:
            self._mouse_note = note
            self.press(note)

    def mouseReleaseEvent(self, _event):
        if self._mouse_note is not None:
            self.release(self._mouse_note)
            self._mouse_note = None

    def press(self, note: int):
        if note not in self.down:
            self.down.add(note)
            self.pressed.emit(note)
            self.update()

    def release(self, note: int):
        if note in self.down:
            self.down.discard(note)
            self.released.emit(note)
            self.update()


class KeyboardDialog(QDialog):
    """Plays the instrument on `device_id` (a bus or device side carrying one)."""

    def __init__(self, engine, device_id: int, name: str, parent: QWidget | None = None,
                 tasks: EngineTasks | None = None):
        super().__init__(parent)
        self._engine = engine
        self._device_id = device_id
        self._tasks = tasks or EngineTasks(self, name='keyboard')
        self._velocity = 100
        self.setWindowTitle(tr('keyboard.title', name=name))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(Spacing.LG, Spacing.LG, Spacing.LG, Spacing.LG)
        self.hint = QLabel(tr('keyboard.hint'))
        self.hint.setObjectName("Dim")
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)

        self.keys = PianoKeys()
        self.keys.pressed.connect(self._note_on)
        self.keys.released.connect(self._note_off)
        layout.addWidget(self.keys, stretch=1)

        row = QHBoxLayout()
        self.down_button = QPushButton(tr('keyboard.octave_down'))
        self.down_button.clicked.connect(lambda: self._shift(-12))
        self.up_button = QPushButton(tr('keyboard.octave_up'))
        self.up_button.clicked.connect(lambda: self._shift(12))
        self.range_label = QLabel()
        self.status = QLabel()
        self.status.setStyleSheet(f"color: {Colors.ERROR.name()};")
        for w in (self.down_button, self.up_button, self.range_label):
            row.addWidget(w)
        row.addStretch()
        row.addWidget(self.status)
        layout.addLayout(row)
        self._show_range()
        self.resize(640, 240)

    @property
    def tasks(self) -> EngineTasks:
        return self._tasks

    def _show_range(self):
        self.range_label.setText(tr('keyboard.range', low=note_name(self.keys.low), high=note_name(self.keys.low + 24)))

    def _shift(self, semitones: int):
        for note in list(self.keys.down):
            self.keys.release(note)
        self.keys.low = min(max(self.keys.low + semitones, 0), 103)
        self._show_range()
        self.keys.update()

    def _send(self, fn, *args):
        def done(result):
            ok, message = result
            self.status.setText('' if ok else message)
        self._tasks.submit(lambda: fn(self._device_id, *args), done)

    def _note_on(self, note: int):
        self._send(self._engine.note_on, note, self._velocity)

    def _note_off(self, note: int):
        self._send(self._engine.note_off, note)

    def keyPressEvent(self, event):
        text = event.text().lower()
        if event.isAutoRepeat():
            return
        if text in KEYS:
            self.keys.press(self.keys.low + KEYS.index(text))
        elif text in ('z', 'x'):
            self._shift(-12 if text == 'z' else 12)
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        text = event.text().lower()
        if not event.isAutoRepeat() and text in KEYS:
            self.keys.release(self.keys.low + KEYS.index(text))

    def done(self, result):
        for note in list(self.keys.down):
            self.keys.release(note)   # no note is left sounding when the keyboard closes
        super().done(result)
