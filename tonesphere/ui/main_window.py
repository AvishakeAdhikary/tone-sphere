"""
The main window.

Layout follows signal flow top to bottom: transport and device selection, then the patch,
then the mixer, then the measured state of the hardware. Nothing in this window displays a
value the engine did not measure.

All engine access happens here. The widgets emit intent and know nothing about audio, which
keeps them testable and keeps the audio layer free of Qt.

No engine call runs on the main thread (`ui/tasks.py`). An action is submitted to the
engine worker, which also gathers what the window needs to redraw - the device list, the
routing matrix - and the main thread renders that snapshot when it arrives. Meters and
statistics come from a poller thread the same way. So the window keeps repainting while a
device takes seconds to open, and the control lock is never waited on here.
"""

import json
from pathlib import Path

from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from tonesphere import i18n
from tonesphere.core.engine_factory import UnifiedAudioEngine
from tonesphere.engine.graph import db_to_linear
from tonesphere.i18n import active_locale, available_locales, locale_info, set_active_locale, tr
from tonesphere.ui.routing_view import RoutingScene, RoutingView
from tonesphere.ui.strip import ChannelStripWidget, HardwareBar, MasterStrip
from tonesphere.ui.tasks import EnginePoller, EngineTasks
from tonesphere.ui.theme import METRICS, Colors, Spacing, Type
from tonesphere.utils.config import ConfigManager
from tonesphere.utils.logger import get_logger

logger = get_logger(__name__)

# Meters at 30 Hz. Fast enough to look continuous, slow enough to leave the CPU to audio.
METER_INTERVAL_MS = 33
# Statistics change slowly and cost more to gather, so they run at a lower rate.
STATS_INTERVAL_MS = 500
STATS_EVERY = max(1, STATS_INTERVAL_MS // METER_INTERVAL_MS)


class MainWindow(QMainWindow):
    """ToneSphere's main window."""

    def __init__(self, config_manager: ConfigManager | None = None):
        super().__init__()

        self.config_manager = config_manager or ConfigManager()
        # Before anything below calls tr(): i18n's active locale is otherwise read from
        # whatever ConfigManager a previous window or test last pointed it at, not this
        # window's own settings store.
        i18n.use_config(self.config_manager)
        self.engine = UnifiedAudioEngine(self.config_manager)
        self.tasks = EngineTasks(self)

        # Keyed by side as well as id: a duplex device (an ASIO driver) is one id with an
        # input strip and an output strip, each metered on its own.
        self._strips: dict[tuple[int, str], ChannelStripWidget] = {}
        self._node_positions: dict[int, QPointF] = {}
        self._inserts_dialogs: dict[tuple[int, bool], object] = {}
        self._diagnostics = None
        # What the engine last reported, as rendered: the window draws from these, never
        # from the engine directly.
        self._view: dict = {'devices': [], 'matrix': {}, 'drivers': [], 'active_driver': None,
                            'plugin_host': False, 'insert_counts': {}, 'sample_rate': None,
                            'buffer_size': None, 'exclusive': None}
        self._state = 'stopped'
        self._running = False
        self._tick = 0
        self._stats_due = True
        self._device_serial = 0

        self.setWindowTitle(tr('app.name'))
        self.resize(1360, 880)
        self.setMinimumSize(1000, 640)

        self._build_ui()
        self._build_menu()
        self.tasks.busy.connect(self.hardware_bar.set_busy)

        # Enumeration opens every endpoint to read its format: seconds on some machines. Then
        # the last session comes back and the engine starts: an audio app that opens silent
        # until the user finds a Start button is one the user thinks is broken.
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(2000)
        self._save_timer.timeout.connect(lambda: self._run(self.engine.save_session))
        self._run(self._start_up, refresh='all', done=self._started_up, error_title=tr('dialog.engine_error'))
        self._poller = EnginePoller(self._gather_poll, METER_INTERVAL_MS, self)
        self._poller.polled.connect(self._apply_poll)

    # --- The engine worker ---

    def _run(self, fn, *args, refresh: str | None = None, done=None, error_title: str | None = None):
        """
        Run `fn(*args)` on the engine worker. `refresh` 'routing' or 'all' has the worker
        also gather what the window must redraw afterwards; `done(result)` then runs here,
        on the main thread, and a failure is shown under `error_title` (or only logged).
        """
        def job():
            result = fn(*args)
            return result, self._gather_view(refresh) if refresh else None

        def finished(value):
            result, view = value
            if view is not None:
                self._apply_view(view)
                # Whatever changed the routing or the devices is worth keeping for next time.
                self._save_timer.start()
            self._stats_due = True
            if done is not None:
                done(result)

        def failed(error):
            self._stats_due = True
            if error_title is not None:
                self._error(error_title, str(error))
            if refresh:
                self._run(lambda: None, refresh=refresh)

        self.tasks.submit(job, finished, failed)

    def _start_up(self):
        """On the worker: devices, then the last session, then the engine running."""
        self.engine.initialize()
        restored = self.engine.restore_session()
        self.engine.start_engine()
        return restored

    def _started_up(self, restored: str | None):
        if restored:
            self.hardware_bar.set_notice(tr('session.partial', detail=restored))

    def _gather_view(self, refresh: str) -> dict:
        """On the worker: everything the window draws from, read under the engine's lock."""
        view = {'matrix': self.engine.get_routing_matrix()}
        if refresh == 'all':
            devices = self.engine.get_devices()
            plugin_host = self.engine.hosts_plugins()
            counts = {}
            if plugin_host:
                for d in devices:
                    if d['origin'] != 'loopback':
                        entries = self.engine.list_inserts(d['id'], d['direction'] == 'input')
                        counts[(d['id'], d['direction'] == 'input')] = (
                            len(entries), any(e['crashed'] for e in entries))
            info = self.engine.get_driver_info()
            view.update(devices=devices, drivers=self.engine.get_available_drivers(),
                        active_driver=info.get('active_driver'), exclusive=info.get('exclusive_mode'),
                        plugin_host=plugin_host, insert_counts=counts,
                        sample_rate=self.engine.sample_rate, buffer_size=self.engine.buffer_size)
        return view

    def _apply_view(self, view: dict):
        rebuild = 'devices' in view
        self._view.update(view)
        if rebuild:
            self._rebuild_from_engine()
        else:
            self._sync_cables()

    # --- Construction ---

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setContentsMargins(Spacing.LG, Spacing.LG, Spacing.LG, Spacing.LG)
        layout.setSpacing(Spacing.LG)

        layout.addWidget(self._build_transport())

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self._build_routing_panel())
        splitter.addWidget(self._build_mixer_panel())
        splitter.setSizes([420, 380])
        splitter.setChildrenCollapsible(False)
        layout.addWidget(splitter, stretch=1)

        self.hardware_bar = HardwareBar()
        layout.addWidget(self.hardware_bar)

    def _build_transport(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("Panel")

        layout = QHBoxLayout(panel)
        layout.setContentsMargins(Spacing.LG, Spacing.MD, Spacing.LG, Spacing.MD)
        layout.setSpacing(Spacing.LG)

        self.engine_button = QPushButton(tr('transport.start_engine'))
        self.engine_button.setObjectName("Primary")
        self.engine_button.setMinimumWidth(130)
        self.engine_button.setMinimumHeight(30)
        self.engine_button.clicked.connect(self._toggle_engine)
        layout.addWidget(self.engine_button)

        self.backend_combo = QComboBox()
        self.backend_combo.setMinimumWidth(180)
        self.backend_combo.currentTextChanged.connect(self._switch_backend)
        self.backend_caption = self._labelled(
            tr('transport.caption.backend'), self.backend_combo, layout
        )

        self.exclusive_button = QPushButton(tr('transport.exclusive'))
        self.exclusive_button.setCheckable(True)
        self.exclusive_button.setChecked(True)
        self.exclusive_button.setMinimumHeight(METRICS.BUTTON_HEIGHT + 4)
        self.exclusive_button.setToolTip(tr('transport.exclusive_tooltip'))
        self.exclusive_button.toggled.connect(self._toggle_exclusive)
        self.mode_caption = self._labelled(
            tr('transport.caption.mode'), self.exclusive_button, layout
        )

        self.buffer_combo = QComboBox()
        self.buffer_combo.setMinimumWidth(90)
        for size in (64, 128, 256, 512, 1024):
            self.buffer_combo.addItem(tr('transport.buffer_frames', frames=size), size)
        self.buffer_combo.setToolTip(tr('transport.buffer_tooltip'))
        self.buffer_combo.currentIndexChanged.connect(self._change_buffer)
        self.buffer_caption = self._labelled(
            tr('transport.caption.buffer'), self.buffer_combo, layout
        )

        self.rate_combo = QComboBox()
        self.rate_combo.setMinimumWidth(90)
        self._fill_rates()
        self.rate_combo.setToolTip(tr('transport.rate_tooltip'))
        self.rate_combo.currentIndexChanged.connect(self._change_rate)
        self.rate_caption = self._labelled(
            tr('transport.caption.rate'), self.rate_combo, layout
        )

        layout.addStretch()

        self.diagnostics_button = QPushButton(tr('transport.diagnostics'))
        self.diagnostics_button.setToolTip(tr('transport.diagnostics_tooltip'))
        self.diagnostics_button.clicked.connect(self._show_diagnostics)
        layout.addWidget(self.diagnostics_button)

        self.monitor_button = QPushButton(tr('transport.monitor'))
        self.monitor_button.setToolTip(tr('transport.monitor_tooltip'))
        self.monitor_button.clicked.connect(self._create_monitor_patch)
        layout.addWidget(self.monitor_button)

        self.rescan_button = QPushButton(tr('transport.rescan'))
        self.rescan_button.setToolTip(tr('transport.rescan_tooltip'))
        self.rescan_button.clicked.connect(self._rescan)
        layout.addWidget(self.rescan_button)

        return panel

    SAMPLE_RATES = (44100, 48000, 88200, 96000)

    def _fill_rates(self):
        self.rate_combo.blockSignals(True)
        self.rate_combo.clear()
        for rate in self.SAMPLE_RATES:
            self.rate_combo.addItem(tr('transport.rate_khz', khz=f"{rate / 1000:g}"), rate)
        index = self.rate_combo.findData(self._view['sample_rate'])
        if index >= 0:
            self.rate_combo.setCurrentIndex(index)
        self.rate_combo.blockSignals(False)

    def _labelled(self, text: str, control: QWidget, parent_layout: QHBoxLayout) -> QLabel:
        """
        Stack a small caption above its control, and add the pair to `parent_layout`.

        Captions beside controls collide the moment a device name is long; above, they
        stay legible and the controls keep a single baseline across the bar. Returns the
        caption label so a language switch can retranslate it later.
        """
        column = QVBoxLayout()
        column.setSpacing(1)
        column.setContentsMargins(0, 0, 0, 0)

        caption = QLabel(text)
        caption.setObjectName("Dim")
        caption.setFont(Type.font(Type.TINY))
        caption.setStyleSheet(f"color: {Colors.TEXT_DIM.name()}; letter-spacing: 1px;")

        column.addWidget(caption)
        column.addWidget(control)
        parent_layout.addLayout(column)
        return caption

    def _build_routing_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("Panel")

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(Spacing.MD, Spacing.MD, Spacing.MD, Spacing.MD)
        layout.setSpacing(Spacing.MD)

        header = QHBoxLayout()
        self.patchbay_title = QLabel(tr('patchbay.title'))
        self.patchbay_title.setObjectName("Heading")
        header.addWidget(self.patchbay_title)

        self.patchbay_hint = QLabel(tr('patchbay.hint'))
        self.patchbay_hint.setObjectName("Dim")
        header.addWidget(self.patchbay_hint)
        header.addStretch()

        self.patchbay_buttons: dict[str, QPushButton] = {}
        for key, slot in (('patchbay.fit', lambda: self.routing_view.fit_content()),
                          ('patchbay.zoom_reset', lambda: self.routing_view.reset_zoom()),
                          ('patchbay.auto_arrange', self._auto_arrange)):
            button = QPushButton(tr(key))
            button.clicked.connect(slot)
            header.addWidget(button)
            self.patchbay_buttons[key] = button

        layout.addLayout(header)

        self.routing_scene = RoutingScene()
        self.routing_scene.connect_requested.connect(self._connect_nodes)
        self.routing_scene.disconnect_requested.connect(self._disconnect_nodes)
        self.routing_scene.gain_requested.connect(self._set_route_gain)
        self.routing_scene.mute_requested.connect(self._set_route_mute)
        self.routing_scene.connect_refused.connect(lambda text: self.hardware_bar.set_notice(text))
        self.routing_scene.source_channel_requested.connect(self._set_route_source_channel)

        self.routing_view = RoutingView(self.routing_scene)
        layout.addWidget(self.routing_view, stretch=1)

        return panel

    def _build_mixer_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("Panel")

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(Spacing.MD, Spacing.MD, Spacing.MD, Spacing.MD)
        layout.setSpacing(Spacing.MD)

        header = QHBoxLayout()
        self.mixer_title = QLabel(tr('mixer.title'))
        self.mixer_title.setObjectName("Heading")
        header.addWidget(self.mixer_title)
        header.addStretch()

        self.add_bus_button = QPushButton(tr('mixer.add_bus'))
        self.add_bus_button.setToolTip(tr('mixer.add_bus_tooltip'))
        self.add_bus_button.clicked.connect(self._add_bus)
        header.addWidget(self.add_bus_button)
        layout.addLayout(header)

        row = QHBoxLayout()
        row.setSpacing(Spacing.MD)

        self.strip_container = QWidget()
        self.strip_layout = QHBoxLayout(self.strip_container)
        self.strip_layout.setContentsMargins(0, 0, 0, 0)
        self.strip_layout.setSpacing(Spacing.MD)
        self.strip_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(self.strip_container)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        row.addWidget(scroll, stretch=1)

        self.master_strip = MasterStrip()
        self.master_strip.gain_changed.connect(self._set_master_gain)
        self.master_strip.clip_cleared.connect(self._clear_clips)
        row.addWidget(self.master_strip)

        layout.addLayout(row, stretch=1)
        return panel

    def _build_menu(self):
        """
        (Re)build the whole menu bar.

        Called once at startup and again on every language switch: a `QAction`'s text
        cannot be swapped without keeping a reference to each one, and a menu bar is cheap
        enough to throw away and rebuild fresh rather than track two dozen extra
        attributes purely to retranslate them. `_change_language` clears the menu bar and
        calls this again.
        """
        file_menu = self.menuBar().addMenu(tr('menu.file'))
        save_preset = QAction(tr('menu.file.save_preset'), self)
        save_preset.setShortcut(QKeySequence.StandardKey.Save)
        save_preset.triggered.connect(self._save_preset)
        file_menu.addAction(save_preset)
        load_preset = QAction(tr('menu.file.load_preset'), self)
        load_preset.setShortcut(QKeySequence.StandardKey.Open)
        load_preset.triggered.connect(self._load_preset)
        file_menu.addAction(load_preset)

        engine_menu = self.menuBar().addMenu(tr('menu.engine'))

        toggle = QAction(tr('menu.engine.start_stop'), self)
        toggle.setShortcut(QKeySequence("Space"))
        toggle.triggered.connect(self._toggle_engine)
        engine_menu.addAction(toggle)

        rescan = QAction(tr('menu.engine.rescan'), self)
        rescan.setShortcut(QKeySequence("F5"))
        rescan.triggered.connect(self._rescan)
        engine_menu.addAction(rescan)

        diagnostics = QAction(tr('menu.engine.diagnostics'), self)
        diagnostics.setShortcut(QKeySequence("Ctrl+D"))
        diagnostics.triggered.connect(self._show_diagnostics)
        engine_menu.addAction(diagnostics)

        cables = QAction(tr('menu.engine.cables'), self)
        cables.triggered.connect(self._show_cables)
        engine_menu.addAction(cables)

        engine_menu.addSeparator()
        quit_action = QAction(tr('menu.engine.quit'), self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        engine_menu.addAction(quit_action)

        patch_menu = self.menuBar().addMenu(tr('menu.patch'))

        monitor = QAction(tr('menu.patch.monitor'), self)
        monitor.triggered.connect(self._create_monitor_patch)
        patch_menu.addAction(monitor)

        clear = QAction(tr('menu.patch.clear'), self)
        clear.triggered.connect(self._clear_routing)
        patch_menu.addAction(clear)

        patch_menu.addSeparator()
        arrange = QAction(tr('menu.patch.auto_arrange'), self)
        arrange.triggered.connect(self._auto_arrange)
        patch_menu.addAction(arrange)

        plugins_menu = self.menuBar().addMenu(tr('menu.plugins'))
        browse = QAction(tr('menu.plugins.browse'), self)
        browse.setShortcut(QKeySequence("Ctrl+B"))
        browse.triggered.connect(self._show_plugin_browser)
        plugins_menu.addAction(browse)
        instrument = QAction(tr('menu.plugins.add_instrument'), self)
        instrument.triggered.connect(self._add_instrument)
        plugins_menu.addAction(instrument)

        language_menu = self.menuBar().addMenu(tr('menu.language'))
        group = QActionGroup(language_menu)
        group.setExclusive(True)
        current = active_locale()

        for info in available_locales():
            action = QAction(info.native_name, group)
            action.setCheckable(True)
            action.setChecked(info.code == current)
            if info.review_status != 'source':
                action.setToolTip(tr('menu.language.unreviewed'))
            action.triggered.connect(lambda checked=False, code=info.code: self._change_language(code))
            language_menu.addAction(action)

        help_menu = self.menuBar().addMenu(tr('menu.help'))
        about = QAction(tr('menu.help.about'), self)
        about.triggered.connect(self._show_about)
        help_menu.addAction(about)

    # --- Population ---

    def _rebuild_from_engine(self):
        """Rebuild strips and nodes from the engine's current device list."""
        self._populate_backends()
        self._select_format()

        devices = self._view['devices']

        self._clear_strips()
        self.routing_scene.clear_content()

        inputs = [d for d in devices if d['direction'] == 'input']
        outputs = [d for d in devices if d['direction'] == 'output']

        for device in inputs + outputs:
            self._add_strip(device)

        self._place_nodes(inputs, outputs)
        self._sync_cables()
        self.routing_view.fit_content()

    def _select_format(self):
        """Show the rate and buffer the engine reported, without that selection acting as a change."""
        index = self.buffer_combo.findData(self._view['buffer_size'])
        if index >= 0:
            self.buffer_combo.blockSignals(True)
            self.buffer_combo.setCurrentIndex(index)
            self.buffer_combo.blockSignals(False)
        self._fill_rates()
        if self._view['exclusive'] is not None:
            self.exclusive_button.blockSignals(True)
            self.exclusive_button.setChecked(bool(self._view['exclusive']))
            self.exclusive_button.blockSignals(False)
            self._label_exclusive()

    def _label_exclusive(self):
        # A checkable button looks the same either way in this theme: the text is the state.
        self.exclusive_button.setText(tr('transport.exclusive') if self.exclusive_button.isChecked()
                                      else tr('transport.shared'))

    def _populate_backends(self):
        self.backend_combo.blockSignals(True)
        self.backend_combo.clear()

        for name in self._view['drivers']:
            self.backend_combo.addItem(name)
        if self._view.get('plugin_host') and 'ASIO' not in self._view['drivers']:
            # The native engine runs ASIO when a driver is installed; with none, say so here
            # rather than leave the user wondering where it went. WASAPI does everything.
            self.backend_combo.addItem(tr('backend.asio_unavailable'))
            item = self.backend_combo.model().item(self.backend_combo.count() - 1)
            item.setEnabled(False)
            # The style sheet's popup draws a disabled item like any other; dim it by hand.
            item.setForeground(Colors.TEXT_DIM)
            item.setToolTip(tr('backend.asio_unavailable_tooltip'))

        active = self._view['active_driver']
        if active:
            index = self.backend_combo.findText(active)
            if index >= 0:
                self.backend_combo.setCurrentIndex(index)

        self.backend_combo.blockSignals(False)

    def _clear_strips(self):
        for strip in self._strips.values():
            self.strip_layout.removeWidget(strip)
            strip.deleteLater()
        self._strips.clear()

    def _hosts_plugins(self, device: dict) -> bool:
        return device['origin'] != 'loopback' and self._view['plugin_host']

    def _add_strip(self, device: dict):
        direction_key = f"device.direction.{device['direction']}"
        is_input = device['direction'] == 'input'
        strip = ChannelStripWidget(
            device_id=device['id'],
            name=device['name'],
            subtitle=tr('device.subtitle', direction=tr(direction_key), channels=device['channels']),
            channels=min(device['channels'], 2),
            is_input=is_input,
            hosts_plugins=self._hosts_plugins(device),
        )
        strip.gain_changed.connect(self._set_device_gain)
        strip.pan_changed.connect(self._set_device_pan)
        strip.mute_toggled.connect(self._set_device_mute)
        strip.solo_toggled.connect(self._set_device_solo)
        strip.inserts_requested.connect(self._show_inserts)

        # Before the stretch, so strips stay left-aligned.
        self.strip_layout.insertWidget(self.strip_layout.count() - 1, strip)
        self._strips[(device['id'], device['direction'])] = strip
        counted = self._view['insert_counts'].get((device['id'], is_input))
        if strip.inserts_button.isEnabled() and counted is not None:
            strip.set_insert_count(*counted)

    def _place_nodes(self, inputs: list[dict], outputs: list[dict]):
        """
        Sources on the left, destinations on the right.

        Signal flows left to right, so cables run in one direction and the patch reads as a
        diagram rather than a tangle.
        """
        seen = set()

        for column, group in ((0, inputs), (1, outputs)):
            x = 60 + column * 420
            for row, device in enumerate(group):
                if device['id'] in seen:
                    continue
                seen.add(device['id'])

                position = self._node_positions.get(
                    device['id'], QPointF(x, 40 + row * 92)
                )
                self._node_positions[device['id']] = position

                is_bus = 'bus' in device['host_api'].lower()
                node = self.routing_scene.add_node(
                    node_id=device['id'],
                    name=device['name'],
                    subtitle=device['host_api'],
                    can_input=(device['direction'] == 'output' or is_bus),
                    can_output=(device['direction'] == 'input' or is_bus),
                    is_bus=is_bus,
                    position=position,
                    channels=max(1, device.get('channels') or 2),
                )
                node.moved.connect(lambda n=node: self._remember_position(n))

    def _remember_position(self, node):
        self._node_positions[node.node_id] = node.pos()

    def _sync_cables(self):
        """Redraw cables from the engine's routing matrix."""
        wanted = {}
        for route in self._view['matrix'].values():
            key = (route['source_id'], route['destination_id'])
            wanted[key] = route

        for key in list(self.routing_scene.cables.keys()):
            if key not in wanted:
                self.routing_scene.remove_cable(*key)

        for (source_id, dest_id), route in wanted.items():
            self.routing_scene.add_cable(
                source_id, dest_id,
                gain_db=route.get('volume_db', 0.0),
                muted=route.get('muted', False),
                source_channel=route.get('source_channel'),
            )

    # --- Engine actions ---

    def _toggle_engine(self):
        stop = self._running or self._state == 'idle'
        self._run(self.engine.stop_engine if stop else self.engine.start_engine,
                  error_title=tr('dialog.engine_error'))

    def _switch_backend(self, name: str):
        if not name:
            return
        self._run(self.engine.switch_driver, name, refresh='all', error_title=tr('dialog.backend_error'))

    def _toggle_exclusive(self, exclusive: bool):
        self._label_exclusive()
        self._run(self.engine.set_exclusive_mode, exclusive)

    def _change_buffer(self, index: int):
        size = self.buffer_combo.itemData(index)
        if size:
            self._run(self.engine.set_buffer_size, int(size))

    def _change_rate(self, index: int):
        rate = self.rate_combo.itemData(index)
        if rate:
            self._run(self.engine.set_sample_rate, int(rate), error_title=tr('dialog.rate_error'))

    # --- Views ---

    def _show_diagnostics(self):
        from tonesphere.ui.diagnostics_view import DiagnosticsDialog

        if self._diagnostics is None:
            self._diagnostics = DiagnosticsDialog(self.engine.engine, self, tasks=self.tasks)
            self._diagnostics.finished.connect(lambda _result: setattr(self, '_diagnostics', None))
        self._diagnostics.show()
        self._diagnostics.raise_()

    def _show_cables(self):
        from tonesphere.ui.cables_view import CablesDialog

        dialog = CablesDialog(self.engine, self, tasks=self.tasks)
        dialog.finished.connect(lambda _result: self._run(lambda: None, refresh='all'))
        dialog.show()

    def _show_plugin_browser(self):
        from tonesphere.ui.plugin_views import PluginBrowser

        PluginBrowser(self.engine.engine, self.config_manager, picking=False, parent=self).exec()

    def _add_instrument(self):
        from tonesphere.ui.plugin_views import PluginBrowser

        browser = PluginBrowser(self.engine.engine, self.config_manager, picking=True, instruments=True, parent=self)

        def added(result):
            ok, message, _bus = result
            if not ok:
                self._error(tr('dialog.instrument_error'), message)

        browser.chosen.connect(lambda info: self._run(self.engine.create_instrument, info, refresh='all', done=added))
        browser.exec()

    def _show_inserts(self, device_id: int, is_input: bool):
        from tonesphere.ui.plugin_views import InsertsDialog

        key = (device_id, is_input)
        dialog = self._inserts_dialogs.get(key)
        if dialog is None:
            strip = self._strips.get((device_id, 'input' if is_input else 'output'))
            name = strip._name if strip is not None else str(device_id)
            dialog = InsertsDialog(self.engine.engine, device_id, is_input, name, self.config_manager, self,
                                   tasks=self.tasks)
            dialog.changed.connect(lambda d=device_id, i=is_input: self._refresh_insert_count(d, i))
            dialog.finished.connect(lambda _result, k=key: self._inserts_dialogs.pop(k, None))
            self._inserts_dialogs[key] = dialog
        dialog.show()
        dialog.raise_()

    def _refresh_insert_count(self, device_id: int, is_input: bool):
        def counted(entries):
            self._view['insert_counts'][(device_id, is_input)] = (
                len(entries), any(e['crashed'] for e in entries))
            strip = self._strips.get((device_id, 'input' if is_input else 'output'))
            if strip is not None:
                strip.set_insert_count(*self._view['insert_counts'][(device_id, is_input)])

        self._run(self.engine.list_inserts, device_id, is_input, done=counted)

    def _rescan(self):
        self._run(self.engine.handle_device_change, refresh='all')

    def _create_monitor_patch(self):
        from tonesphere.ui.monitor_dialog import MonitorDialog

        def chosen(defaults):
            dialog = MonitorDialog(self._view.get('devices') or [], *defaults, parent=self)
            if dialog.exec() != dialog.DialogCode.Accepted:
                return
            self._run(self.engine.monitor, *dialog.choice(), refresh='all', done=monitoring)

        def monitoring(result):
            success, message = result
            if success:
                self.hardware_bar.set_notice(tr('monitor.on', patch=message))
            else:
                self._error(tr('dialog.monitor_error'), message)

        self._run(lambda: (self.engine.default_input_id(), self.engine.default_output_id()), done=chosen)

    def _save_preset(self):
        from PySide6.QtWidgets import QFileDialog

        from tonesphere.core.presets import default_presets_dir

        default_presets_dir().mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(self, tr('menu.file.save_preset'),
                                              str(default_presets_dir() / 'My setup.yaml'), 'Presets (*.yaml)')
        if path:
            self._run(lambda: self.engine.save_preset(Path(path)), error_title=tr('dialog.preset_error'),
                      done=lambda _r: self.hardware_bar.set_notice(tr('preset.saved', name=Path(path).stem)))

    def _load_preset(self):
        from PySide6.QtWidgets import QFileDialog

        from tonesphere.core.presets import default_presets_dir

        path, _ = QFileDialog.getOpenFileName(self, tr('menu.file.load_preset'), str(default_presets_dir()),
                                              'Presets (*.yaml)')
        if path:
            self._run(lambda: self.engine.load_preset(Path(path)), refresh='all', error_title=tr('dialog.preset_error'),
                      done=lambda summary: self.hardware_bar.set_notice(summary))

    def _add_bus(self):
        def create():
            count = len(self.engine.list_virtual_devices()) + 1
            return self.engine.create_virtual_input(tr('mixer.bus_name', number=count), channels=2)

        def created(bus_id):
            if bus_id is None:
                self._error(tr('dialog.add_bus_error'), tr('dialog.bus_limit'))

        self._run(create, refresh='all', done=created)

    def _clear_routing(self):
        self._run(self.engine.clear_all_routing, refresh='routing')

    def _auto_arrange(self):
        self._node_positions.clear()
        self._rebuild_from_engine()

    # --- Routing actions ---

    def _connect_nodes(self, source_id: int, dest_id: int):
        def connected(result):
            success, message = result
            if not success:
                self._error(tr('dialog.patch_error'), message)

        def connect():
            result = self.engine.create_routing(source_id, dest_id, 1.0)
            if result[0] and not self._running:
                self.engine.start_engine()
            return result

        self._run(connect, refresh='routing', done=connected)

    def _disconnect_nodes(self, source_id: int, dest_id: int):
        self._run(self.engine.remove_routing, source_id, dest_id, refresh='routing')

    def _set_route_gain(self, source_id: int, dest_id: int, gain_db: float):
        self._run(self.engine.set_routing_volume_db, source_id, dest_id, gain_db, refresh='routing')

    def _set_route_mute(self, source_id: int, dest_id: int, muted: bool):
        self._run(self.engine.set_routing_mute, source_id, dest_id, muted, refresh='routing')

    def _set_route_source_channel(self, source_id: int, dest_id: int, channel):
        self._run(self.engine.set_routing_source_channel, source_id, dest_id, channel, refresh='routing',
                  done=lambda result: None if result[0] else self._error(tr('dialog.patch_error'), result[1]))

    # --- Strip actions ---

    def _set_device_gain(self, device_id: int, gain_db: float):
        self._run(self.engine.set_device_master_volume, device_id, db_to_linear(gain_db))

    def _set_device_pan(self, device_id: int, pan: float):
        # Channel 0 carries the strip's pan for a stereo device; per-channel pan is
        # available through the engine for anyone who needs it.
        self._run(self.engine.set_channel_pan, device_id, 0, pan)

    def _set_device_mute(self, device_id: int, muted: bool):
        self._run(self.engine.set_device_master_mute, device_id, muted)

    def _set_device_solo(self, device_id: int, soloed: bool):
        self._run(self.engine.set_channel_solo, device_id, 0, soloed)

    def _set_master_gain(self, gain_db: float):
        self._run(self.engine.set_master_volume, db_to_linear(gain_db))

    def _clear_clips(self):
        self._run(self.engine.clear_clip_indicators)

    # --- Periodic updates ---

    def _gather_poll(self) -> dict:
        """
        On the poller thread: meters every tick, statistics every STATS_EVERY ticks or
        straight after an action. Blocks on the engine's lock if it must; the main thread
        only ever sees the result.
        """
        self._tick += 1
        running = self.engine.is_running
        poll = {'running': running, 'meters': self.engine.get_meters() if running else None}
        if self._stats_due or self._tick % STATS_EVERY == 0:
            self._stats_due = False
            poll['stats'] = self.engine.get_performance_stats()
            poll['state'] = self.engine.state
            poll['failed'] = self.engine.failed_device_ids()
            poll['device_change'] = self.engine.last_device_change()
        return poll

    def _apply_poll(self, poll: dict):
        self._running = poll['running']
        self._update_meters(poll['meters'])
        if 'stats' in poll:
            self._update_stats(poll['stats'], poll['state'], poll['failed'])
            change = poll['device_change']
            if change and change['serial'] != self._device_serial:
                self._device_serial = change['serial']
                self.hardware_bar.set_notice(self._describe_change(change))
                # The device list changed under the window: redraw it from a fresh snapshot.
                self._run(lambda: None, refresh='all')

    @staticmethod
    def _describe_change(change: dict) -> str:
        parts = []
        if change['left']:
            parts.append(tr('status.device_left', names=', '.join(change['left'])))
        if change['arrived']:
            parts.append(tr('status.device_arrived', names=', '.join(change['arrived'])))
        if change['reopened']:
            parts.append(tr('status.device_reopened'))
        return ' · '.join(parts)

    def _update_meters(self, meters: dict | None):
        """Show one poll's meters. Values the audio thread measured; never computed here."""
        if meters is None:
            for strip in self._strips.values():
                strip.set_inactive()
            self.master_strip.set_inactive()
            return

        for (device_id, direction), strip in self._strips.items():
            reading = meters.get(device_id, {}).get('sides', {}).get(direction)
            if reading is None:
                strip.set_inactive()
                continue

            if len(reading['channel_peak_db']) >= strip.channels:
                peaks = reading['channel_peak_db'][:strip.channels]
                rms = reading['channel_rms_db'][:strip.channels]
                holds = reading['channel_peak_hold_db'][:strip.channels]
            else:
                # The host measured this side only as a whole: one reading, on every bar.
                peaks = [reading['peak_db']] * strip.channels
                rms = [reading['rms_db']] * strip.channels
                holds = [reading['peak_hold_db']] * strip.channels
            strip.set_levels(peaks, rms, holds, reading['clipped'])

        for device_id, reading in meters.items():
            node = self.routing_scene.nodes.get(device_id)
            if node is not None:
                node.set_level(self._db_to_bar(reading['peak_db']))

        # The master meter is what the outputs are being given — not the loudest input, which
        # made it dance with a guitar while the headphones stayed silent.
        outputs = [meters.get(d['id'], {}).get('sides', {}).get('output')
                   for d in self._view.get('devices') or [] if d['direction'] == 'output']
        outputs = [r for r in outputs if r is not None]
        if outputs:
            loudest = max(outputs, key=lambda r: r['peak_db'])
            self.master_strip.set_levels(
                [loudest['peak_db']] * 2,
                [loudest['rms_db']] * 2,
                [loudest['peak_hold_db']] * 2,
                any(r['clipped'] for r in outputs),
            )
        else:
            self.master_strip.set_inactive()

    @staticmethod
    def _db_to_bar(db: float) -> float:
        """Map dBFS onto 0..1 for the small node level strip."""
        if db <= -60.0:
            return 0.0
        return min(1.0, (db + 60.0) / 60.0)

    def _update_stats(self, stats: dict, state: str, failed: dict[int, str]):
        self._state = state
        self._last_stats = stats
        self.hardware_bar.update_state(state, stats)

        self.engine_button.setText(
            tr('transport.stop_engine') if state in ('running', 'degraded', 'idle')
            else tr('transport.start_engine')
        )
        self.engine_button.setObjectName(
            "Danger" if state in ('running', 'degraded', 'idle') else "Primary"
        )
        self.engine_button.setStyleSheet("")   # force a restyle for the new object name

        for (device_id, _direction), strip in self._strips.items():
            reason = failed.get(device_id)
            strip.set_failed(reason)

            node = self.routing_scene.nodes.get(device_id)
            if node is not None:
                node.set_failed(reason)

    # --- Misc ---

    def _error(self, title: str, message: str):
        logger.warning(f"{title}: {message}")
        QMessageBox.warning(self, title, message)

    # --- Language ---

    def _change_language(self, code: str):
        """
        Switch the active locale and retranslate the whole window immediately.

        No "restart to apply" here: everything text-bearing either gets set again below
        or is rebuilt from scratch by `_rebuild_from_engine()`, which already reconstructs
        every strip and graph node from the engine's device list on any refresh — reusing
        that path means the strips and the patchbay pick up the new language exactly the
        way they pick up a new device, with no separate retranslation logic to keep in
        sync with `_add_strip`/`_place_nodes`.
        """
        if code == active_locale():
            return

        saved = set_active_locale(code)
        if not saved:
            logger.warning(f"Language changed to '{code}' for this session, but the choice "
                           f"could not be written to settings")

        from PySide6.QtWidgets import QApplication

        from tonesphere.ui.app import apply_layout_direction
        app = QApplication.instance()
        if app is not None:
            apply_layout_direction(app)

        self._retranslate()

        if not saved:
            self._error(tr('dialog.language_error'),
                       tr('dialog.language_not_saved', reason='could not write to the settings store'))

    def _retranslate(self):
        """Refresh every piece of static text this window owns for the active locale."""
        self.setWindowTitle(tr('app.name'))

        self.engine_button.setText(
            tr('transport.stop_engine') if self._running or self._state == 'idle'
            else tr('transport.start_engine')
        )
        self.backend_caption.setText(tr('transport.caption.backend'))
        self._label_exclusive()
        self.exclusive_button.setToolTip(tr('transport.exclusive_tooltip'))
        self.mode_caption.setText(tr('transport.caption.mode'))

        current_buffer = self.buffer_combo.currentData()
        self.buffer_combo.blockSignals(True)
        self.buffer_combo.clear()
        for size in (64, 128, 256, 512, 1024):
            self.buffer_combo.addItem(tr('transport.buffer_frames', frames=size), size)
        index = self.buffer_combo.findData(current_buffer)
        if index >= 0:
            self.buffer_combo.setCurrentIndex(index)
        self.buffer_combo.blockSignals(False)
        self.buffer_combo.setToolTip(tr('transport.buffer_tooltip'))
        self.buffer_caption.setText(tr('transport.caption.buffer'))
        self._fill_rates()
        self.rate_combo.setToolTip(tr('transport.rate_tooltip'))
        self.rate_caption.setText(tr('transport.caption.rate'))
        self.diagnostics_button.setText(tr('transport.diagnostics'))
        self.diagnostics_button.setToolTip(tr('transport.diagnostics_tooltip'))

        self.monitor_button.setText(tr('transport.monitor'))
        self.monitor_button.setToolTip(tr('transport.monitor_tooltip'))
        self.rescan_button.setText(tr('transport.rescan'))
        self.rescan_button.setToolTip(tr('transport.rescan_tooltip'))

        self.patchbay_title.setText(tr('patchbay.title'))
        self.patchbay_hint.setText(tr('patchbay.hint'))
        for key, button in self.patchbay_buttons.items():
            button.setText(tr(key))

        self.mixer_title.setText(tr('mixer.title'))
        self.add_bus_button.setText(tr('mixer.add_bus'))
        self.add_bus_button.setToolTip(tr('mixer.add_bus_tooltip'))

        self.hardware_bar.retranslate()
        self.master_strip.retranslate()

        self.menuBar().clear()
        self._build_menu()

        # Strips and graph nodes are reconstructed with the engine's current device list,
        # which is also how they pick up the new language's tr() output — see
        # _change_language's docstring for why this is reused rather than duplicated.
        self._rebuild_from_engine()
        self._stats_due = True

    def _show_about(self):
        self._run(lambda: (self.engine.get_driver_info(), self.engine.get_performance_stats(),
                           self.engine.buffer_size, self.engine.sample_rate),  # on the worker
                  done=lambda result: self._about(*result))

    def _about(self, info: dict, stats: dict, buffer_size: int, sample_rate: int):
        from tonesphere import __version__

        reported = stats.get('reported_latency_ms')
        measured = stats.get('measured_round_trip_ms')
        nominal = stats.get('nominal_latency_ms')

        reported_text = (
            tr('about.latency_reported', ms=f"{reported:.1f}") if reported
            else tr('about.not_measured')
        )
        measured_text = f"{measured:.1f} ms" if measured is not None else '--'
        nominal_text = (
            tr('about.latency_nominal', ms=f"{nominal:.1f}") if nominal else '--'
        )

        lines = [
            tr('about.version', version=__version__),
            "",
            tr('about.portaudio', version=info.get('portaudio_version', tr('about.unknown'))),
            tr('about.backend', backend=info.get('active_driver') or tr('about.not_selected')),
            tr('about.exclusive', exclusive=tr('about.on') if info.get('exclusive_mode') else tr('about.off')),
            tr('about.buffer', frames=buffer_size, rate=sample_rate),
            tr('about.latency', reported=reported_text, nominal=nominal_text, measured=measured_text),
            tr('about.dropouts', xruns=stats.get('xruns', 0)),
            "",
        ]

        if not info.get('asio_available'):
            lines.append(tr('about.asio_note'))
            lines.append("")

        lines.append(tr('about.bus_note'))

        current = locale_info()
        if current.review_status != 'source':
            lines.append("")
            lines.append(tr('about.translation_note', language=current.native_name))

        QMessageBox.information(self, tr('about.title'), "\n".join(lines))

    def report_when_ready(self, path: Path, timeout_ms: int = 120_000):
        """
        For the packaging smoke test: once the window is drawing a real device list, write
        what a user would see — the backend, whether the native engine loaded, the devices —
        to `path` and quit. A start that never gets there writes the timeout instead.
        """
        from PySide6.QtWidgets import QApplication

        def write(report: dict, code: int):
            path.write_text(json.dumps(report, indent=2), encoding='utf-8')
            QApplication.exit(code)

        def gather():
            return {'driver': self.engine.get_driver_info(), 'hosts_plugins': self.engine.hosts_plugins()}

        def done(info: dict):
            write({'window_title': self.windowTitle(), 'visible': self.isVisible(),
                   'devices': len(self._view.get('devices') or []), 'drivers': self._view.get('drivers'),
                   'backend': info['driver'].get('backend', 'portaudio'), 'engine': info['driver'].get('engine'),
                   'asio_available': info['driver'].get('asio_available'), 'hosts_plugins': info['hosts_plugins'],
                   'error': info['driver'].get('error')}, 0)

        def check():
            if 'devices' in self._view:
                poll.stop()
                self._run(gather, done=done)

        poll = QTimer(self)
        poll.timeout.connect(check)
        poll.start(250)
        QTimer.singleShot(timeout_ms, lambda: write({'error': 'the window never showed a device list'}, 3))

    def closeEvent(self, event):
        self._poller.stop()
        for dialog in list(self._inserts_dialogs.values()) + ([self._diagnostics] if self._diagnostics else []):
            dialog.close()

        def cleanup():
            try:
                self.engine.save_session()
            except Exception as e:
                logger.warning(f"The session could not be saved: {e}")
            try:
                self.engine.cleanup()
            except Exception as e:
                logger.warning(f"Error during shutdown: {e}")

        # The one place the main thread waits on the worker: the window is going away, and
        # the engine must be stopped and its devices released before the process exits.
        self.tasks.close(final=cleanup)
        super().closeEvent(event)
