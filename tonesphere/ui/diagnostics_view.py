"""
Diagnostics: everything the engine measured about itself, and the one measurement a user
has to ask for, the round trip.

Latency is broken into the parts that are actually different kinds of number: ToneSphere's
own block (arithmetic), what the drivers report, what the plugins report, and the round
trip timed by sending a sweep out and hearing it back. Only the last is a measurement, and
until one has been taken at the current rate and block it reads `--`.

Like the main window, this dialog never calls the engine on the main thread: its readings
are gathered by a poller thread and the measurement runs on the engine worker
(`ui/tasks.py`).
"""

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from tonesphere.i18n import tr
from tonesphere.ui.tasks import EnginePoller, EngineTasks
from tonesphere.ui.theme import Colors, Spacing, Type
from tonesphere.utils.formatting import UNKNOWN

REFRESH_MS = 500


def ms(value: float | None, decimals: int = 2) -> str:
    return UNKNOWN if value is None else tr('diag.value_ms', ms=f"{value:.{decimals}f}")


def us(value_ms: float | None) -> str:
    return UNKNOWN if value_ms is None else tr('diag.value_us', us=f"{value_ms * 1000:.1f}")


def percent(fraction: float | None) -> str:
    return UNKNOWN if fraction is None else tr('diag.value_percent', percent=f"{fraction * 100:.1f}")


def count(value: int | None) -> str:
    return UNKNOWN if value is None else str(value)


class _Section(QGroupBox):
    """A titled grid of label/value rows."""

    def __init__(self, title: str, parent: QWidget | None = None):
        super().__init__(title, parent)
        self._grid = QGridLayout(self)
        self._grid.setHorizontalSpacing(Spacing.XL)
        self._grid.setVerticalSpacing(Spacing.SM)
        self._grid.setColumnStretch(1, 1)
        self.values: dict[str, QLabel] = {}

    def row(self, key: str, caption: str) -> QLabel:
        r = self._grid.rowCount()
        label = QLabel(caption)
        label.setObjectName("Dim")
        value = QLabel(UNKNOWN)
        value.setFont(Type.numeric(Type.SMALL))
        value.setWordWrap(True)
        self._grid.addWidget(label, r, 0)
        self._grid.addWidget(value, r, 1)
        self.values[key] = value
        return value

    def set(self, key: str, text: str, colour=None):
        label = self.values[key]
        label.setText(text)
        label.setStyleSheet(f"color: {colour.name()};" if colour is not None else "")


class DiagnosticsDialog(QDialog):
    def __init__(self, engine, parent: QWidget | None = None, tasks: EngineTasks | None = None):
        super().__init__(parent)
        self._engine = engine
        self._tasks = tasks or EngineTasks(self, name='diagnostics')
        self._measuring = False
        self._closed = False

        self.setWindowTitle(tr('diag.title'))
        self.resize(760, 780)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(Spacing.LG, Spacing.LG, Spacing.LG, Spacing.LG)
        layout.setSpacing(Spacing.MD)

        self.engine_section = _Section(tr('diag.section.engine'))
        for key, caption in (('state', tr('diag.state')), ('backend', tr('diag.backend')),
                             ('host_api', tr('diag.host_api')), ('format', tr('diag.format')),
                             ('streams', tr('diag.streams')), ('failed', tr('diag.failed'))):
            self.engine_section.row(key, caption)
        layout.addWidget(self.engine_section)

        self.timing_section = _Section(tr('diag.section.timing'))
        for key, caption in (('callback', tr('diag.callback')), ('load_worst', tr('diag.load_worst')),
                             ('load_mean', tr('diag.load_mean')), ('xruns', tr('diag.xruns')),
                             ('allocations', tr('diag.allocations')), ('rings', tr('diag.rings')),
                             ('drift', tr('diag.drift'))):
            self.timing_section.row(key, caption)
        layout.addWidget(self.timing_section)

        self.latency_section = _Section(tr('diag.section.latency'))
        for key, caption in (('nominal', tr('diag.nominal')), ('driver_in', tr('diag.driver_in')),
                             ('driver_out', tr('diag.driver_out')), ('plugins', tr('diag.plugins')),
                             ('reported', tr('diag.reported')), ('measured', tr('diag.measured')),
                             ('last', tr('diag.last_measurement'))):
            self.latency_section.row(key, caption)
        layout.addWidget(self.latency_section)
        layout.addWidget(self._build_measure())

        self.virtual_section = _Section(tr('diag.section.virtual'))
        self.virtual_section.row('status', tr('diag.virtual_status'))
        self.virtual_section.row('endpoints', tr('diag.virtual_endpoints'))
        self.virtual_note = QLabel(tr('diag.virtual_note'))
        self.virtual_note.setObjectName("Dim")
        self.virtual_note.setWordWrap(True)
        self.virtual_section.layout().addWidget(self.virtual_note, self.virtual_section.layout().rowCount(), 0, 1, 2)
        layout.addWidget(self.virtual_section)

        close = QHBoxLayout()
        close.addStretch()
        self.close_button = QPushButton(tr('dialog.close'))
        self.close_button.clicked.connect(self.accept)
        close.addWidget(self.close_button)
        layout.addLayout(close)

        self._populate_devices()
        self._poller = EnginePoller(self.gather, REFRESH_MS, self, name='diagnostics-poll')
        self._poller.polled.connect(self.show_data)

    def _build_measure(self) -> QWidget:
        box = QGroupBox(tr('diag.section.measure'))
        grid = QGridLayout(box)
        grid.setHorizontalSpacing(Spacing.MD)

        self.output_combo = QComboBox()
        self.input_combo = QComboBox()
        grid.addWidget(QLabel(tr('diag.measure_output')), 0, 0)
        grid.addWidget(self.output_combo, 0, 1)
        grid.addWidget(QLabel(tr('diag.measure_input')), 1, 0)
        grid.addWidget(self.input_combo, 1, 1)

        self.measure_warning = QLabel(tr('diag.measure_warning'))
        self.measure_warning.setWordWrap(True)
        self.measure_warning.setStyleSheet(f"color: {Colors.WARN.name()};")
        grid.addWidget(self.measure_warning, 2, 0, 1, 2)

        row = QHBoxLayout()
        self.measure_status = QLabel()
        self.measure_status.setObjectName("Dim")
        self.measure_status.setWordWrap(True)
        row.addWidget(self.measure_status, stretch=1)
        self.measure_button = QPushButton(tr('diag.measure'))
        self.measure_button.setObjectName("Primary")
        self.measure_button.clicked.connect(self.measure)
        row.addWidget(self.measure_button)
        grid.addLayout(row, 3, 0, 1, 2)
        return box

    def _populate_devices(self):
        from tonesphere.engine.devices import HostApi

        def gather():
            host = self._engine.host
            supported = getattr(host, 'backend', 'portaudio') == 'native' and host.host_api == HostApi.WASAPI
            return self._engine.get_devices(), self._engine.default_output_id(), supported

        self._tasks.submit(gather, lambda result: self._alive() and self._show_devices(*result))

    def _show_devices(self, devices: list[dict], default_out: int | None, supported: bool):
        self.output_combo.clear()
        self.input_combo.clear()
        self.input_combo.addItem(tr('diag.measure_loopback'), None)
        for d in devices:
            if d['origin'] == 'in_process_bus':
                continue
            if d['direction'] == 'output':
                self.output_combo.addItem(d['name'], d['id'])
            else:
                self.input_combo.addItem(d['name'], d['id'])
        index = self.output_combo.findData(default_out)
        if index >= 0:
            self.output_combo.setCurrentIndex(index)
        self.measure_button.setEnabled(supported and self.output_combo.count() > 0)
        if not supported:
            self.measure_status.setText(tr('diag.measure_unsupported'))

    def _alive(self) -> bool:
        """A result can arrive after the dialog closed; it is then dropped."""
        return not self._closed

    def measure(self):
        output, listen = self.output_combo.currentData(), self.input_combo.currentData()
        if self._measuring or output is None:
            return
        self._measuring = True
        self.measure_button.setEnabled(False)
        self.measure_status.setText(tr('diag.measuring'))
        self._tasks.submit(lambda: self._engine.measure_round_trip(output, listen),
                           lambda result: self._alive() and self._measured(result),
                           lambda error: self._alive() and self._measure_failed(str(error)))

    def _measured(self, result: dict):
        self._measuring = False
        self.measure_button.setEnabled(True)
        self.measure_status.setText(result['note'])

    def _measure_failed(self, message: str):
        self._measuring = False
        self.measure_button.setEnabled(True)
        self.measure_status.setText(tr('diag.measure_failed', reason=message))

    def gather(self) -> dict:
        """On the poller thread: everything shown, read under the engine's lock."""
        stats = self._engine.get_performance_stats()
        return {
            'stats': stats,
            'state': self._engine.state,
            'rings': self._engine.get_ring_statistics().get('native') if stats.get('running') else None,
            'configured_exclusive': self._engine.host.exclusive,
            'virtual': self._engine.virtual_device_status(),
        }

    def show_data(self, data: dict):
        stats, state = data['stats'], data['state']

        e = self.engine_section
        e.set('state', {
            'running': tr('status.state.running'),
            'degraded': tr('status.state.degraded'),
            'idle': tr('status.state.idle'),
            'stopped': tr('status.state.stopped'),
        }.get(state, tr('status.state.unknown')))
        e.set('backend', tr('diag.backend_native') if stats.get('backend') == 'native'
              else tr('diag.backend_portaudio'))
        e.set('host_api', stats.get('host_api') or UNKNOWN)
        rate, block = stats.get('samplerate'), stats.get('blocksize')
        # Running, the streams say what they got; stopped, the configuration is all there is.
        exclusive = stats.get('exclusive') if stats.get('running') else data['configured_exclusive']
        e.set('format', tr('diag.format_value', rate=rate or UNKNOWN, block=block or UNKNOWN,
                           mode=tr('diag.exclusive') if exclusive else tr('diag.shared')))
        e.set('streams', tr('diag.streams_value', live=stats.get('live_stream_count', 0),
                            total=stats.get('active_streams', 0)))
        failed = stats.get('failed_streams') or {}
        e.set('failed', '; '.join(f"{k}: {v}" for k, v in failed.items()) if failed else tr('diag.none'),
              Colors.ERROR if failed else None)

        t = self.timing_section
        t.set('callback', tr('diag.callback_value', min=us(stats.get('callback_min_ms')),
                             mean=us(stats.get('callback_mean_ms')), p99=us(stats.get('callback_p99_ms')),
                             max=us(stats.get('callback_max_ms'))))
        worst = stats.get('processing_load')
        t.set('load_worst', percent(worst),
              None if worst is None else (Colors.OK if worst < 0.5 else Colors.WARN if worst < 0.8 else Colors.ERROR))
        mean = stats.get('cpu_usage')
        t.set('load_mean', percent(mean / 100 if mean is not None else None))
        running = stats.get('running')
        xruns = stats.get('xruns') if running else None
        t.set('xruns', count(xruns), None if xruns is None else (Colors.OK if not xruns else Colors.ERROR))
        allocations = stats.get('audio_thread_allocations')
        t.set('allocations', count(allocations),
              None if allocations is None else (Colors.OK if allocations == 0 else Colors.ERROR))
        rings = data['rings']
        t.set('rings', tr('diag.rings_value', overruns=rings['overruns'], underruns=rings['underruns'])
              if rings else UNKNOWN)
        t.set('drift', count(stats.get('drift_corrections')) if stats.get('backend') != 'native' else UNKNOWN)

        lat = self.latency_section
        lat.set('nominal', ms(stats.get('nominal_latency_ms')))
        lat.set('driver_in', ms(stats.get('input_latency_ms')))
        lat.set('driver_out', ms(stats.get('output_latency_ms')))
        samples = stats.get('plugin_latency_samples')
        lat.set('plugins', UNKNOWN if samples is None else tr('diag.plugins_value', samples=samples,
                                                                 ms=f"{stats['plugin_latency_ms']:.2f}"))
        lat.set('reported', ms(stats.get('reported_latency_ms')))
        measured = stats.get('measured_round_trip_ms')
        lat.set('measured', ms(measured), Colors.OK if measured is not None else None)
        lat.set('last', self._describe(stats.get('round_trip')))

        self._show_virtual(data['virtual'])

    def _show_virtual(self, status: dict):
        v = self.virtual_section
        if not status['platform_supported']:
            v.set('status', tr('diag.virtual_windows_only'))
            v.set('endpoints', UNKNOWN)
        elif status['installed']:
            v.set('status', tr('diag.virtual_present'), Colors.OK)
            v.set('endpoints', f"{status['render']['name']}  ·  {status['capture']['name']}")
        else:
            v.set('status', tr('diag.virtual_absent'), Colors.TEXT_MUTED)
            v.set('endpoints', UNKNOWN)

    @staticmethod
    def _describe(trip: dict | None) -> str:
        if trip is None:
            return tr('diag.never_measured')
        path = tr('diag.path_loopback') if trip['path'] == 'loopback' else tr('diag.path_capture')
        return tr('diag.last_value', value=ms(trip['measured_ms']), path=path, output=trip['output'],
                  input=trip['input'], rate=trip['sample_rate'], block=trip['block'],
                  confidence=f"{trip['confidence']:.1f}", note=trip['note'])

    def done(self, result):
        self._closed = True
        self._poller.stop()
        super().done(result)
