"""
The main window.

Layout follows signal flow top to bottom: transport and device selection, then the patch,
then the mixer, then the measured state of the hardware. Nothing in this window displays a
value the engine did not measure.

All engine access happens here. The widgets emit intent and know nothing about audio, which
keeps them testable and keeps the audio layer free of Qt.
"""


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
from tonesphere.ui.theme import METRICS, Colors, Spacing, Type
from tonesphere.utils.config import ConfigManager
from tonesphere.utils.logger import get_logger

logger = get_logger(__name__)

# Meters at 30 Hz. Fast enough to look continuous, slow enough to leave the CPU to audio.
METER_INTERVAL_MS = 33
# Statistics change slowly and cost more to gather, so they run at a lower rate.
STATS_INTERVAL_MS = 500


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
        self.engine.initialize()

        self._strips: dict[int, ChannelStripWidget] = {}
        self._node_positions: dict[int, QPointF] = {}

        self.setWindowTitle(tr('app.name'))
        self.resize(1360, 880)
        self.setMinimumSize(1000, 640)

        self._build_ui()
        self._build_menu()
        self._rebuild_from_engine()

        self._meter_timer = QTimer(self)
        self._meter_timer.timeout.connect(self._update_meters)
        self._meter_timer.start(METER_INTERVAL_MS)

        self._stats_timer = QTimer(self)
        self._stats_timer.timeout.connect(self._update_stats)
        self._stats_timer.start(STATS_INTERVAL_MS)

        self._update_stats()

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
        index = self.buffer_combo.findData(self.engine.buffer_size)
        if index >= 0:
            self.buffer_combo.setCurrentIndex(index)
        self.buffer_combo.setToolTip(tr('transport.buffer_tooltip'))
        self.buffer_combo.currentIndexChanged.connect(self._change_buffer)
        self.buffer_caption = self._labelled(
            tr('transport.caption.buffer'), self.buffer_combo, layout
        )

        layout.addStretch()

        self.monitor_button = QPushButton(tr('transport.monitor'))
        self.monitor_button.setToolTip(tr('transport.monitor_tooltip'))
        self.monitor_button.clicked.connect(self._create_monitor_patch)
        layout.addWidget(self.monitor_button)

        self.rescan_button = QPushButton(tr('transport.rescan'))
        self.rescan_button.setToolTip(tr('transport.rescan_tooltip'))
        self.rescan_button.clicked.connect(self._rescan)
        layout.addWidget(self.rescan_button)

        return panel

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
        self.master_strip.clip_cleared.connect(self.engine.clear_clip_indicators)
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
        engine_menu = self.menuBar().addMenu(tr('menu.engine'))

        toggle = QAction(tr('menu.engine.start_stop'), self)
        toggle.setShortcut(QKeySequence("Space"))
        toggle.triggered.connect(self._toggle_engine)
        engine_menu.addAction(toggle)

        rescan = QAction(tr('menu.engine.rescan'), self)
        rescan.setShortcut(QKeySequence("F5"))
        rescan.triggered.connect(self._rescan)
        engine_menu.addAction(rescan)

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

        devices = self.engine.get_devices()

        self._clear_strips()
        self.routing_scene.clear_content()

        inputs = [d for d in devices if d['direction'] == 'input']
        outputs = [d for d in devices if d['direction'] == 'output']

        for device in inputs + outputs:
            self._add_strip(device)

        self._place_nodes(inputs, outputs)
        self._sync_cables()
        self.routing_view.fit_content()

    def _populate_backends(self):
        self.backend_combo.blockSignals(True)
        self.backend_combo.clear()

        for name in self.engine.get_available_drivers():
            self.backend_combo.addItem(name)

        active = self.engine.get_driver_info().get('active_driver')
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

    def _add_strip(self, device: dict):
        direction_key = f"device.direction.{device['direction']}"
        strip = ChannelStripWidget(
            device_id=device['id'],
            name=device['name'],
            subtitle=tr('device.subtitle', direction=tr(direction_key), channels=device['channels']),
            channels=min(device['channels'], 2),
        )
        strip.gain_changed.connect(self._set_device_gain)
        strip.pan_changed.connect(self._set_device_pan)
        strip.mute_toggled.connect(self._set_device_mute)
        strip.solo_toggled.connect(self._set_device_solo)

        # Before the stretch, so strips stay left-aligned.
        self.strip_layout.insertWidget(self.strip_layout.count() - 1, strip)
        self._strips[device['id']] = strip

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
                )
                node.moved.connect(lambda n=node: self._remember_position(n))

    def _remember_position(self, node):
        self._node_positions[node.node_id] = node.pos()

    def _sync_cables(self):
        """Redraw cables from the engine's routing matrix."""
        wanted = {}
        for route in self.engine.get_routing_matrix().values():
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
            )

    # --- Engine actions ---

    def _toggle_engine(self):
        try:
            if self.engine.is_running or self.engine.state == 'idle':
                self.engine.stop_engine()
            else:
                self.engine.start_engine()
        except Exception as e:
            self._error(tr('dialog.engine_error'), str(e))

        self._update_stats()

    def _switch_backend(self, name: str):
        if not name:
            return
        try:
            if self.engine.switch_driver(name):
                self._rebuild_from_engine()
        except Exception as e:
            self._error(tr('dialog.backend_error'), str(e))

    def _toggle_exclusive(self, exclusive: bool):
        self.engine.set_exclusive_mode(exclusive)
        self._update_stats()

    def _change_buffer(self, index: int):
        size = self.buffer_combo.itemData(index)
        if size:
            self.engine.set_buffer_size(int(size))
            self._update_stats()

    def _rescan(self):
        self.engine.refresh_devices()
        self._rebuild_from_engine()

    def _create_monitor_patch(self):
        success, message = self.engine.create_monitor_patch(muted=True)

        if not success:
            self._error(tr('dialog.monitor_error'), message)
            return

        self._sync_cables()

        answer = QMessageBox.question(
            self, tr('dialog.unmute_title'),
            tr('dialog.unmute_body', patch=message),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if answer == QMessageBox.StandardButton.Yes:
            source = self.engine.default_input_id()
            dest = self.engine.default_output_id()
            self.engine.set_routing_mute(source, dest, False)
            self._sync_cables()

    def _add_bus(self):
        count = len(self.engine.list_virtual_devices()) + 1
        bus_id = self.engine.create_virtual_input(tr('mixer.bus_name', number=count), channels=2)

        if bus_id is None:
            self._error(tr('dialog.add_bus_error'), tr('dialog.bus_limit'))
            return

        self._rebuild_from_engine()

    def _clear_routing(self):
        self.engine.clear_all_routing()
        self._sync_cables()

    def _auto_arrange(self):
        self._node_positions.clear()
        self._rebuild_from_engine()

    # --- Routing actions ---

    def _connect_nodes(self, source_id: int, dest_id: int):
        success, message = self.engine.create_routing(source_id, dest_id, 1.0)

        if not success:
            self._error(tr('dialog.patch_error'), message)
            return

        self._sync_cables()
        self._update_stats()

    def _disconnect_nodes(self, source_id: int, dest_id: int):
        self.engine.remove_routing(source_id, dest_id)
        self._sync_cables()

    def _set_route_gain(self, source_id: int, dest_id: int, gain_db: float):
        self.engine.set_routing_volume_db(source_id, dest_id, gain_db)
        self._sync_cables()

    def _set_route_mute(self, source_id: int, dest_id: int, muted: bool):
        self.engine.set_routing_mute(source_id, dest_id, muted)
        self._sync_cables()

    # --- Strip actions ---

    def _set_device_gain(self, device_id: int, gain_db: float):
        self.engine.set_device_master_volume(device_id, db_to_linear(gain_db))

    def _set_device_pan(self, device_id: int, pan: float):
        # Channel 0 carries the strip's pan for a stereo device; per-channel pan is
        # available through the engine for anyone who needs it.
        self.engine.set_channel_pan(device_id, 0, pan)

    def _set_device_mute(self, device_id: int, muted: bool):
        self.engine.set_device_master_mute(device_id, muted)

    def _set_device_solo(self, device_id: int, soloed: bool):
        self.engine.set_channel_solo(device_id, 0, soloed)

    def _set_master_gain(self, gain_db: float):
        self.engine.master_volume = db_to_linear(gain_db)

    # --- Periodic updates ---

    def _update_meters(self):
        """
        Refresh meters at 30 Hz.

        Reads values the audio callback stored in plain slots. The callback never touches
        Qt, and this never blocks the callback.
        """
        if not self.engine.is_running:
            for strip in self._strips.values():
                strip.set_inactive()
            self.master_strip.set_inactive()
            return

        meters = self.engine.get_meters()

        for device_id, strip in self._strips.items():
            reading = meters.get(device_id)
            if reading is None:
                strip.set_inactive()
                continue

            peaks = [reading['peak_db']] * strip.channels
            rms = [reading['rms_db']] * strip.channels
            holds = [reading['peak_hold_db']] * strip.channels
            strip.set_levels(peaks, rms, holds, reading['clipped'])

            node = self.routing_scene.nodes.get(device_id)
            if node is not None:
                node.set_level(self._db_to_bar(reading['peak_db']))

        if meters:
            loudest = max(meters.values(), key=lambda r: r['peak_db'])
            self.master_strip.set_levels(
                [loudest['peak_db']] * 2,
                [loudest['rms_db']] * 2,
                [loudest['peak_hold_db']] * 2,
                any(r['clipped'] for r in meters.values()),
            )

    @staticmethod
    def _db_to_bar(db: float) -> float:
        """Map dBFS onto 0..1 for the small node level strip."""
        if db <= -60.0:
            return 0.0
        return min(1.0, (db + 60.0) / 60.0)

    def _update_stats(self):
        stats = self.engine.get_performance_stats()
        state = self.engine.state

        self.hardware_bar.update_state(state, stats)

        self.engine_button.setText(
            tr('transport.stop_engine') if state in ('running', 'degraded', 'idle')
            else tr('transport.start_engine')
        )
        self.engine_button.setObjectName(
            "Danger" if state in ('running', 'degraded', 'idle') else "Primary"
        )
        self.engine_button.setStyleSheet("")   # force a restyle for the new object name

        failed = stats.get('failed_streams') or {}
        for device_id, strip in self._strips.items():
            device = self.engine.engine.get_device_info(device_id)
            reason = failed.get(device.key) if device else None
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
            tr('transport.stop_engine') if self.engine.is_running or self.engine.state == 'idle'
            else tr('transport.start_engine')
        )
        self.backend_caption.setText(tr('transport.caption.backend'))
        self.exclusive_button.setText(tr('transport.exclusive'))
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
        self._update_stats()

    def _show_about(self):
        from tonesphere import __version__

        info = self.engine.get_driver_info()
        stats = self.engine.get_performance_stats()

        measured = stats.get('measured_latency_ms')
        nominal = stats.get('nominal_latency_ms')

        measured_text = (
            tr('about.latency_measured', ms=f"{measured:.1f}") if measured
            else tr('about.not_measured')
        )
        nominal_text = (
            tr('about.latency_nominal', ms=f"{nominal:.1f}") if nominal else '--'
        )

        lines = [
            tr('about.version', version=__version__),
            "",
            tr('about.portaudio', version=info.get('portaudio_version', tr('about.unknown'))),
            tr('about.backend', backend=info.get('active_driver') or tr('about.not_selected')),
            tr('about.exclusive', exclusive=tr('about.on') if info.get('exclusive_mode') else tr('about.off')),
            tr('about.buffer', frames=self.engine.buffer_size, rate=self.engine.sample_rate),
            tr('about.latency', measured=measured_text, nominal=nominal_text),
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

    def closeEvent(self, event):
        self._meter_timer.stop()
        self._stats_timer.stop()
        try:
            self.engine.cleanup()
        except Exception as e:
            logger.warning(f"Error during shutdown: {e}")
        super().closeEvent(event)
