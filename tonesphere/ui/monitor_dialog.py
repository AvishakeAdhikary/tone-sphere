"""
Monitor Input: hear an input — a guitar, a microphone — on an output, in one step.

The old button patched Windows' default input to its default output, muted (so a laptop's
own microphone and speakers could not howl), and left unmuting to a dialog that defaulted to
No and to an engine the user had not started: the meters moved and nothing was heard. Here
the user picks the input, the channel and the output — an audio interface is preselected
over the computer's own devices — and on OK the route is made unmuted and the engine runs.
A guitar on input 1 of a two-input interface defaults to that channel alone, in both ears.
"""

import re

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from tonesphere.i18n import tr
from tonesphere.ui.theme import Colors, Spacing

# The computer's own audio hardware, by the names its drivers give it. An input on one of
# these, monitored on another, is a microphone next to a speaker.
BUILT_IN = re.compile(r'realtek|intel|smart sound|conexant|high definition audio|microphone array|'
                      r'stereo mix|cirrus|synaptics|webcam|camera', re.IGNORECASE)


def hardware_name(name: str) -> str:
    """'Line (AI-04)' and 'Speakers (AI-04)' are one device: what the brackets say."""
    match = re.search(r'\(([^()]*(?:\([^()]*\)[^()]*)*)\)\s*$', name)
    return (match[1] if match else name).strip().lower()


def is_built_in(name: str) -> bool:
    return bool(BUILT_IN.search(name))


def choose_defaults(inputs: list[dict], outputs: list[dict], default_input: int | None,
                    default_output: int | None) -> tuple[int | None, int | None, int | None]:
    """
    (input id, output id, source channel) to preselect: an interface's input and its own
    output when there is one — input 1 alone for a two-input interface — else Windows'
    defaults, all channels.
    """
    for device in inputs:
        if is_built_in(device['name']):
            continue
        twin = next((o for o in outputs if hardware_name(o['name']) == hardware_name(device['name'])), None)
        if twin is not None:
            channel = 0 if device['channels'] == 2 else None
            return device['id'], twin['id'], channel
    return default_input, default_output, None


class MonitorDialog(QDialog):
    """Pick what to hear and where; `choice()` is (input id, output id, source channel)."""

    def __init__(self, devices: list[dict], default_input: int | None, default_output: int | None,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(tr('monitor.title'))
        physical = [d for d in devices if d['origin'] not in ('loopback', 'in_process_bus')]
        self._inputs = [d for d in physical if d['direction'] == 'input']
        self._outputs = [d for d in physical if d['direction'] == 'output']

        layout = QVBoxLayout(self)
        layout.setContentsMargins(Spacing.LG, Spacing.LG, Spacing.LG, Spacing.LG)
        layout.setSpacing(Spacing.MD)
        hint = QLabel(tr('monitor.hint'))
        hint.setWordWrap(True)
        hint.setObjectName("Dim")
        layout.addWidget(hint)

        form = QFormLayout()
        self.input_combo = QComboBox()
        for d in self._inputs:
            self.input_combo.addItem(d['name'], d['id'])
        self.channel_combo = QComboBox()
        self.output_combo = QComboBox()
        for d in self._outputs:
            self.output_combo.addItem(d['name'], d['id'])
        form.addRow(tr('monitor.input'), self.input_combo)
        form.addRow(tr('monitor.channel'), self.channel_combo)
        form.addRow(tr('monitor.output'), self.output_combo)
        layout.addLayout(form)

        self.warning = QLabel()
        self.warning.setWordWrap(True)
        self.warning.setStyleSheet(f"color: {Colors.WARN.name()};")
        layout.addWidget(self.warning)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText(tr('monitor.start'))
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        input_id, output_id, channel = choose_defaults(self._inputs, self._outputs, default_input, default_output)
        self.input_combo.currentIndexChanged.connect(self._input_changed)
        self.output_combo.currentIndexChanged.connect(self._check_feedback)
        self.channel_combo.currentIndexChanged.connect(self._check_feedback)
        self._select(self.input_combo, input_id)
        self._input_changed()
        self._select(self.output_combo, output_id)
        if channel is not None:
            self._select(self.channel_combo, channel)
        self._check_feedback()
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(bool(self._inputs and self._outputs))

    @staticmethod
    def _select(combo: QComboBox, value):
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _device(self, devices: list[dict], combo: QComboBox) -> dict | None:
        return next((d for d in devices if d['id'] == combo.currentData()), None)

    def _input_changed(self):
        device = self._device(self._inputs, self.input_combo)
        self.channel_combo.blockSignals(True)
        self.channel_combo.clear()
        if device is not None:
            if device['channels'] > 1:
                self.channel_combo.addItem(tr('monitor.channel.all'), None)
            for c in range(device['channels']):
                self.channel_combo.addItem(tr('monitor.channel.one', number=c + 1), c)
        self.channel_combo.blockSignals(False)
        self._check_feedback()

    def _check_feedback(self):
        source = self._device(self._inputs, self.input_combo)
        dest = self._device(self._outputs, self.output_combo)
        risky = source is not None and dest is not None and is_built_in(source['name']) and \
            is_built_in(dest['name']) and 'headphone' not in dest['name'].lower()
        self.warning.setText(tr('monitor.feedback') if risky else '')

    def choice(self) -> tuple[int, int, int | None]:
        return self.input_combo.currentData(), self.output_combo.currentData(), self.channel_combo.currentData()
