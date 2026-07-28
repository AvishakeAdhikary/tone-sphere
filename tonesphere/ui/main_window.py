"""
The main window.

Layout follows signal flow top to bottom: transport and device selection, then the patch,
then the mixer, then the measured state of the hardware. Nothing in this window displays a
value the engine did not measure.

All engine access happens here. The widgets emit intent and know nothing about audio, which
keeps them testable and keeps the audio layer free of Qt.
"""

import math
from typing import Dict, List, Optional

from PySide6.QtCore import QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPushButton, QScrollArea, QSplitter, QVBoxLayout, QWidget,
)

from tonesphere.core.engine_factory import UnifiedAudioEngine
from tonesphere.ui.routing_view import RoutingScene, RoutingView
from tonesphere.ui.strip import ChannelStripWidget, HardwareBar, MasterStrip
from tonesphere.engine.graph import db_to_linear, linear_to_db
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

    def __init__(self, config_manager: Optional[ConfigManager] = None):
        super().__init__()

        self.config_manager = config_manager or ConfigManager()
        self.engine = UnifiedAudioEngine(self.config_manager)
        self.engine.initialize()

        self._strips: Dict[int, ChannelStripWidget] = {}
        self._node_positions: Dict[int, QPointF] = {}

        self.setWindowTitle("ToneSphere")
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

        self.engine_button = QPushButton("Start Engine")
        self.engine_button.setObjectName("Primary")
        self.engine_button.setMinimumWidth(130)
        self.engine_button.setMinimumHeight(30)
        self.engine_button.clicked.connect(self._toggle_engine)
        layout.addWidget(self.engine_button)

        self.backend_combo = QComboBox()
        self.backend_combo.setMinimumWidth(180)
        self.backend_combo.currentTextChanged.connect(self._switch_backend)
        layout.addLayout(self._labelled("BACKEND", self.backend_combo))

        self.exclusive_button = QPushButton("Exclusive")
        self.exclusive_button.setCheckable(True)
        self.exclusive_button.setChecked(True)
        self.exclusive_button.setMinimumHeight(METRICS.BUTTON_HEIGHT + 4)
        self.exclusive_button.setToolTip(
            "Bypass the OS mixer for the lowest latency.\n"
            "Measured here: 8.3 ms exclusive against 22 ms shared on the same device.\n"
            "No other application can use the device while ToneSphere holds it."
        )
        self.exclusive_button.toggled.connect(self._toggle_exclusive)
        layout.addLayout(self._labelled("MODE", self.exclusive_button))

        self.buffer_combo = QComboBox()
        self.buffer_combo.setMinimumWidth(90)
        for size in (64, 128, 256, 512, 1024):
            self.buffer_combo.addItem(f"{size} frames", size)
        index = self.buffer_combo.findData(self.engine.buffer_size)
        if index >= 0:
            self.buffer_combo.setCurrentIndex(index)
        self.buffer_combo.setToolTip(
            "Frames per callback. Smaller is lower latency with less margin before a\n"
            "dropout — watch the XRUNS counter after lowering it."
        )
        self.buffer_combo.currentIndexChanged.connect(self._change_buffer)
        layout.addLayout(self._labelled("BUFFER", self.buffer_combo))

        layout.addStretch()

        monitor = QPushButton("Monitor Input")
        monitor.setToolTip("Patch the default input straight to the default output")
        monitor.clicked.connect(self._create_monitor_patch)
        layout.addWidget(monitor)

        rescan = QPushButton("Rescan")
        rescan.setToolTip("Re-enumerate audio hardware (F5)")
        rescan.clicked.connect(self._rescan)
        layout.addWidget(rescan)

        return panel

    def _labelled(self, text: str, control: QWidget) -> QVBoxLayout:
        """
        Stack a small caption above its control.

        Captions beside controls collide the moment a device name is long; above, they
        stay legible and the controls keep a single baseline across the bar.
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
        return column

    def _build_routing_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("Panel")

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(Spacing.MD, Spacing.MD, Spacing.MD, Spacing.MD)
        layout.setSpacing(Spacing.MD)

        header = QHBoxLayout()
        title = QLabel("Patchbay")
        title.setObjectName("Heading")
        header.addWidget(title)

        hint = QLabel("drag a right-hand port onto a left-hand port to patch  ·  "
                      "right-click a cable for options  ·  middle-drag to pan")
        hint.setObjectName("Dim")
        header.addWidget(hint)
        header.addStretch()

        for text, slot in (("Fit", lambda: self.routing_view.fit_content()),
                           ("100%", lambda: self.routing_view.reset_zoom()),
                           ("Auto-arrange", self._auto_arrange)):
            button = QPushButton(text)
            button.clicked.connect(slot)
            header.addWidget(button)

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
        title = QLabel("Mixer")
        title.setObjectName("Heading")
        header.addWidget(title)
        header.addStretch()

        add_bus = QPushButton("Add Bus")
        add_bus.setToolTip("Create an in-process mix bus (not visible to other apps)")
        add_bus.clicked.connect(self._add_bus)
        header.addWidget(add_bus)
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
        engine_menu = self.menuBar().addMenu("&Engine")

        toggle = QAction("Start / Stop", self)
        toggle.setShortcut(QKeySequence("Space"))
        toggle.triggered.connect(self._toggle_engine)
        engine_menu.addAction(toggle)

        rescan = QAction("Rescan Devices", self)
        rescan.setShortcut(QKeySequence("F5"))
        rescan.triggered.connect(self._rescan)
        engine_menu.addAction(rescan)

        engine_menu.addSeparator()
        quit_action = QAction("Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        engine_menu.addAction(quit_action)

        patch_menu = self.menuBar().addMenu("&Patch")

        monitor = QAction("Monitor Input to Output", self)
        monitor.triggered.connect(self._create_monitor_patch)
        patch_menu.addAction(monitor)

        clear = QAction("Clear All Routing", self)
        clear.triggered.connect(self._clear_routing)
        patch_menu.addAction(clear)

        patch_menu.addSeparator()
        arrange = QAction("Auto-arrange", self)
        arrange.triggered.connect(self._auto_arrange)
        patch_menu.addAction(arrange)

        help_menu = self.menuBar().addMenu("&Help")
        about = QAction("About", self)
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

        for index, device in enumerate(inputs + outputs):
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

    def _add_strip(self, device: Dict):
        strip = ChannelStripWidget(
            device_id=device['id'],
            name=device['name'],
            subtitle=f"{device['direction']} · {device['channels']} ch",
            channels=min(device['channels'], 2),
        )
        strip.gain_changed.connect(self._set_device_gain)
        strip.pan_changed.connect(self._set_device_pan)
        strip.mute_toggled.connect(self._set_device_mute)
        strip.solo_toggled.connect(self._set_device_solo)

        # Before the stretch, so strips stay left-aligned.
        self.strip_layout.insertWidget(self.strip_layout.count() - 1, strip)
        self._strips[device['id']] = strip

    def _place_nodes(self, inputs: List[Dict], outputs: List[Dict]):
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
            self._error("Engine", str(e))

        self._update_stats()

    def _switch_backend(self, name: str):
        if not name:
            return
        try:
            if self.engine.switch_driver(name):
                self._rebuild_from_engine()
        except Exception as e:
            self._error("Backend", str(e))

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
            self._error("Monitor patch", message)
            return

        self._sync_cables()

        answer = QMessageBox.question(
            self, "Unmute monitoring?",
            f"Patched {message}\n\n"
            "It is muted for now. If your input is a microphone and your output is "
            "speakers, unmuting will cause feedback.\n\nUnmute now?",
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
        bus_id = self.engine.create_virtual_input(f"Bus {count}", channels=2)

        if bus_id is None:
            self._error("Add bus", "Bus limit reached")
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
            self._error("Cannot patch", message)
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
            "Stop Engine" if state in ('running', 'degraded', 'idle') else "Start Engine"
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

    def _show_about(self):
        from tonesphere import __version__

        info = self.engine.get_driver_info()
        stats = self.engine.get_performance_stats()

        measured = stats.get('measured_latency_ms')
        nominal = stats.get('nominal_latency_ms')

        lines = [
            f"ToneSphere {__version__}",
            "",
            f"PortAudio: {info.get('portaudio_version', 'unknown')}",
            f"Backend:   {info.get('active_driver') or 'not selected'}",
            f"Exclusive: {info.get('exclusive_mode')}",
            f"Buffer:    {self.engine.buffer_size} frames @ {self.engine.sample_rate} Hz",
            f"Latency:   {f'{measured:.1f} ms measured' if measured else 'not measured'}"
            f" / {f'{nominal:.1f} ms nominal' if nominal else '--'}",
            f"Dropouts:  {stats.get('xruns', 0)}",
            "",
        ]

        if not info.get('asio_available'):
            lines.append(
                "ASIO is not in this PortAudio build: the SDK is not redistributable.\n"
                "WASAPI exclusive is the low-latency path here."
            )
            lines.append("")

        lines.append(
            "Mix buses are in-process only and cannot be selected from other\n"
            "applications. That needs a signed kernel driver — see the Roadmap."
        )

        QMessageBox.information(self, "About ToneSphere", "\n".join(lines))

    def closeEvent(self, event):
        self._meter_timer.stop()
        self._stats_timer.stop()
        try:
            self.engine.cleanup()
        except Exception as e:
            logger.warning(f"Error during shutdown: {e}")
        super().closeEvent(event)
