"""
VST3 plugins: the browser that scans for them, and the insert chain of one device side.

Scanning runs each module in a subprocess (`tonesphere.plugins.scan`), which takes seconds
on a cold cache, so it runs on a worker thread and the window stays live. A module that
crashed, hung or is the wrong architecture is listed with its reason, never hidden: the
user installed it, and deserves to know why it is not offered.

Parameter values shown here are the plugin's own: its display text, read back after every
change, not a number this window computed.
"""

from pathlib import Path

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from tonesphere.i18n import tr
from tonesphere.ui.theme import Colors, Spacing
from tonesphere.utils.logger import get_logger

logger = get_logger(__name__)

SLIDER_STEPS = 1000
CUSTOM_PATHS_SECTION = 'plugins'
CUSTOM_PATHS_KEY = 'custom_paths'


def status_text(status: str) -> str:
    from tonesphere.plugins import scan

    return {
        scan.OK: tr('plugins.status.ok'),
        scan.CRASHED: tr('plugins.status.crashed'),
        scan.TIMED_OUT: tr('plugins.status.timed_out'),
        scan.FAILED: tr('plugins.status.failed'),
        scan.WRONG_ARCHITECTURE: tr('plugins.status.wrong_architecture'),
        scan.NOT_A_PLUGIN: tr('plugins.status.not_a_plugin'),
    }.get(status, status)


class ScanWorker(QThread):
    scanned = Signal(list)
    failed = Signal(str)

    def __init__(self, engine, roots: list[Path], parent=None):
        super().__init__(parent)
        self._engine = engine
        self._roots = roots

    def run(self):
        try:
            self.scanned.emit(self._engine.scan_plugins(self._roots))
        except Exception as e:  # a scan failure is shown to the user, not raised into Qt
            logger.warning(f"plugin scan failed: {e}")
            self.failed.emit(str(e))


class PluginBrowser(QDialog):
    """
    Every VST3 module in the standard folders and the user's own, with what the scan found.
    Opened from an insert chain, it also picks one: `chosen` carries its `PluginInfo`.
    """

    chosen = Signal(object)

    COLUMNS = 7

    def __init__(self, engine, config_manager=None, picking: bool = False, auto_scan: bool = True,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self._engine = engine
        self._config = config_manager
        self._picking = picking
        self._worker: ScanWorker | None = None
        self._rows: list[object | None] = []

        self.setWindowTitle(tr('plugins.browser.title'))
        self.resize(980, 560)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(Spacing.LG, Spacing.LG, Spacing.LG, Spacing.LG)
        layout.setSpacing(Spacing.MD)

        paths_row = QHBoxLayout()
        self.paths_label = QLabel()
        self.paths_label.setObjectName("Dim")
        self.paths_label.setWordWrap(True)
        paths_row.addWidget(self.paths_label, stretch=1)
        self.add_folder_button = QPushButton(tr('plugins.browser.add_folder'))
        self.add_folder_button.clicked.connect(self._add_folder)
        paths_row.addWidget(self.add_folder_button)
        self.scan_button = QPushButton(tr('plugins.browser.scan'))
        self.scan_button.clicked.connect(self.scan)
        paths_row.addWidget(self.scan_button)
        layout.addLayout(paths_row)

        self.table = QTableWidget(0, self.COLUMNS)
        self.table.setHorizontalHeaderLabels([
            tr('plugins.column.name'), tr('plugins.column.vendor'), tr('plugins.column.category'),
            tr('plugins.column.version'), tr('plugins.column.architecture'), tr('plugins.column.status'),
            tr('plugins.column.path'),
        ])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.table.itemDoubleClicked.connect(lambda _item: self._insert())
        layout.addWidget(self.table, stretch=1)

        bottom = QHBoxLayout()
        self.status_label = QLabel()
        self.status_label.setObjectName("Dim")
        self.status_label.setWordWrap(True)
        bottom.addWidget(self.status_label, stretch=1)
        self.insert_button = QPushButton(tr('plugins.browser.insert'))
        self.insert_button.setObjectName("Primary")
        self.insert_button.setEnabled(False)
        self.insert_button.setVisible(picking)
        self.insert_button.clicked.connect(self._insert)
        bottom.addWidget(self.insert_button)
        self.close_button = QPushButton(tr('dialog.close'))
        self.close_button.clicked.connect(self.reject)
        bottom.addWidget(self.close_button)
        layout.addLayout(bottom)

        self._show_paths()
        if auto_scan:
            QTimer.singleShot(0, self.scan)

    # --- Paths ---

    def custom_paths(self) -> list[str]:
        if self._config is None:
            return []
        stored = self._config.get(CUSTOM_PATHS_SECTION, CUSTOM_PATHS_KEY, []) or []
        return [p for p in stored if isinstance(p, str)]

    def roots(self) -> list[Path]:
        from tonesphere.plugins import scan

        return scan.standard_paths() + [Path(p) for p in self.custom_paths()]

    def _show_paths(self):
        self.paths_label.setText(tr('plugins.browser.paths', paths='  ·  '.join(str(p) for p in self.roots())))

    def _add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, tr('plugins.browser.add_folder'))
        if not folder or self._config is None:
            return
        paths = self.custom_paths()
        if folder not in paths:
            paths.append(folder)
            self._config.set(CUSTOM_PATHS_SECTION, CUSTOM_PATHS_KEY, paths)
        self._show_paths()
        self.scan()

    # --- Scanning ---

    def scan(self):
        if self._worker is not None and self._worker.isRunning():
            return
        self.scan_button.setEnabled(False)
        self.status_label.setText(tr('plugins.browser.scanning'))
        self._worker = ScanWorker(self._engine, self.roots(), self)
        self._worker.scanned.connect(self.show_results)
        self._worker.failed.connect(self._scan_failed)
        self._worker.finished.connect(lambda: self.scan_button.setEnabled(True))
        self._worker.start()

    def _scan_failed(self, message: str):
        self.status_label.setText(tr('plugins.browser.scan_failed', reason=message))

    def show_results(self, results: list):
        self.table.setRowCount(0)
        self._rows = []
        usable = 0
        for result in results:
            if result.effects:
                for info in result.effects:
                    if info.is_instrument:
                        self._add_row(info.name, info.vendor, info.subcategories, info.version, result.architecture,
                                      tr('plugins.status.instrument'), result.path, tr('plugins.instrument_detail'),
                                      None)
                        continue
                    self._add_row(info.name, info.vendor, info.subcategories or info.category, info.version,
                                  result.architecture, status_text(result.status), result.path, result.detail, info)
                    usable += 1
            else:
                self._add_row(Path(result.path).stem, '', '', '', result.architecture,
                              status_text(result.status), result.path, result.detail, None)
        self.status_label.setText(tr('plugins.browser.found', usable=usable, modules=len(results)))

    def _add_row(self, name, vendor, category, version, architecture, status, path, detail, info):
        row = self.table.rowCount()
        self.table.insertRow(row)
        for column, text in enumerate((name, vendor, category, version, architecture, status, path)):
            item = QTableWidgetItem(text)
            item.setToolTip(detail or path)
            if info is None:
                item.setForeground(Colors.TEXT_DIM if not detail else Colors.WARN)
            self.table.setItem(row, column, item)
        self._rows.append(info)

    def _selected_info(self):
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows:
            return None
        return self._rows[rows[0].row()]

    def _selection_changed(self):
        self.insert_button.setEnabled(self._picking and self._selected_info() is not None)

    def _insert(self):
        info = self._selected_info()
        if not self._picking or info is None:
            return
        self.chosen.emit(info)
        self.accept()

    def closeEvent(self, event):
        if self._worker is not None:
            self._worker.wait()
        super().closeEvent(event)

    def done(self, result):
        if self._worker is not None:
            self._worker.wait()
        super().done(result)


class ParameterPanel(QWidget):
    """One plugin's parameters, filtered by name, and a slider for the selected one."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._instance = None
        self._setter = None
        self._parameters: list = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(Spacing.SM)

        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText(tr('plugins.params.filter'))
        self.filter_edit.textChanged.connect(self._apply_filter)
        layout.addWidget(self.filter_edit)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels([tr('plugins.params.name'), tr('plugins.params.value'),
                                              tr('plugins.params.units')])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self._selected)
        layout.addWidget(self.table, stretch=1)

        row = QHBoxLayout()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, SLIDER_STEPS)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._slider_moved)
        row.addWidget(self.slider, stretch=1)
        self.value_label = QLabel()
        self.value_label.setMinimumWidth(120)
        row.addWidget(self.value_label)
        layout.addLayout(row)

    def show_plugin(self, instance, setter):
        """`setter(param_id, normalized)` sends a change through the engine's parameter queue."""
        self._instance = instance
        self._setter = setter
        self._parameters = instance.parameters() if instance is not None else []
        self.table.setRowCount(0)
        for p in self._parameters:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(p.title))
            self.table.setItem(row, 1, QTableWidgetItem(p.display))
            self.table.setItem(row, 2, QTableWidgetItem(p.units))
        self._apply_filter(self.filter_edit.text())
        self.slider.setEnabled(False)
        self.value_label.setText("")

    def _apply_filter(self, text: str):
        needle = text.strip().lower()
        for row, p in enumerate(self._parameters):
            self.table.setRowHidden(row, bool(needle) and needle not in p.title.lower())

    def _current(self):
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows or rows[0].row() >= len(self._parameters):
            return None, None
        return rows[0].row(), self._parameters[rows[0].row()]

    def _selected(self):
        row, p = self._current()
        if p is None:
            self.slider.setEnabled(False)
            return
        self.slider.blockSignals(True)
        self.slider.setValue(round(p.normalized * SLIDER_STEPS))
        self.slider.blockSignals(False)
        self.slider.setEnabled(not p.read_only and self._setter is not None)
        if p.step_count > 0:
            self.slider.setSingleStep(max(1, SLIDER_STEPS // p.step_count))
        self.value_label.setText(p.display)

    def _slider_moved(self, value: int):
        row, p = self._current()
        if p is None or self._setter is None:
            return
        normalized = value / SLIDER_STEPS
        if p.step_count > 0:
            normalized = round(normalized * p.step_count) / p.step_count
        self._setter(p.id, normalized)
        self.refresh(row)

    def refresh(self, row: int | None = None):
        """Read the plugin's own values back: what it says, not what was sent."""
        if self._instance is None:
            return
        self._parameters = self._instance.parameters()
        rows = [row] if row is not None else range(len(self._parameters))
        for r in rows:
            if r < len(self._parameters):
                self.table.item(r, 1).setText(self._parameters[r].display)
        if row is not None and row < len(self._parameters):
            self.value_label.setText(self._parameters[row].display)


class InsertsDialog(QDialog):
    """The VST3 plugins on one side of one device, in processing order."""

    changed = Signal()

    def __init__(self, engine, device_id: int, is_input: bool, device_name: str, config_manager=None,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self._engine = engine
        self._device_id = device_id
        self._is_input = is_input
        self._config = config_manager

        side = tr('device.direction.input') if is_input else tr('device.direction.output')
        self.setWindowTitle(tr('plugins.inserts.title', device=device_name, side=side))
        self.resize(860, 520)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(Spacing.LG, Spacing.LG, Spacing.LG, Spacing.LG)
        layout.setSpacing(Spacing.MD)

        self.hint_label = QLabel(tr('plugins.inserts.hint'))
        self.hint_label.setObjectName("Dim")
        self.hint_label.setWordWrap(True)
        layout.addWidget(self.hint_label)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.chain = QListWidget()
        self.chain.currentRowChanged.connect(self._row_changed)
        left_layout.addWidget(self.chain, stretch=1)

        buttons = QHBoxLayout()
        self.add_button = QPushButton(tr('plugins.inserts.add'))
        self.add_button.clicked.connect(self._browse)
        self.remove_button = QPushButton(tr('plugins.inserts.remove'))
        self.remove_button.clicked.connect(self._remove)
        self.bypass_button = QPushButton(tr('plugins.inserts.bypass'))
        self.bypass_button.setCheckable(True)
        self.bypass_button.setToolTip(tr('plugins.inserts.bypass_tooltip'))
        self.bypass_button.toggled.connect(self._bypass)
        self.editor_button = QPushButton(tr('plugins.inserts.open_editor'))
        self.editor_button.clicked.connect(self._open_editor)
        for b in (self.add_button, self.remove_button, self.bypass_button, self.editor_button):
            buttons.addWidget(b)
        left_layout.addLayout(buttons)

        self.fault_label = QLabel()
        self.fault_label.setWordWrap(True)
        self.fault_label.setStyleSheet(f"color: {Colors.ERROR.name()};")
        left_layout.addWidget(self.fault_label)
        splitter.addWidget(left)

        self.parameters = ParameterPanel()
        splitter.addWidget(self.parameters)
        splitter.setSizes([340, 520])
        layout.addWidget(splitter, stretch=1)

        close = QHBoxLayout()
        close.addStretch()
        self.close_button = QPushButton(tr('dialog.close'))
        self.close_button.clicked.connect(self.accept)
        close.addWidget(self.close_button)
        layout.addLayout(close)

        self.refresh()

    def entries(self) -> list[dict]:
        return self._engine.list_plugins(self._device_id, self._is_input)

    def refresh(self, select: int | None = None):
        entries = self.entries()
        current = self.chain.currentRow() if select is None else select
        self.chain.blockSignals(True)
        self.chain.clear()
        for e in entries:
            latency = tr('plugins.inserts.latency', samples=e['latency_samples']) if e['latency_samples'] else ''
            text = tr('plugins.inserts.entry', name=e['name'], vendor=e['vendor'] or '--', latency=latency)
            if e['crashed']:
                text = tr('plugins.inserts.crashed_entry', entry=text)
            elif e['bypassed']:
                text = tr('plugins.inserts.bypassed_entry', entry=text)
            item = QListWidgetItem(text)
            if e['crashed']:
                item.setForeground(Colors.ERROR)
            elif e['bypassed']:
                item.setForeground(Colors.TEXT_DIM)
            self.chain.addItem(item)
        self.chain.blockSignals(False)
        if entries:
            self.chain.setCurrentRow(min(max(current, 0), len(entries) - 1))
        else:
            self._row_changed(-1)

    def _row_changed(self, row: int):
        entries = self.entries()
        valid = 0 <= row < len(entries)
        self.remove_button.setEnabled(valid)
        self.bypass_button.setEnabled(valid and not (valid and entries[row]['crashed']))
        self.editor_button.setEnabled(valid and entries[row]['has_editor'] and not entries[row]['crashed'])
        self.bypass_button.blockSignals(True)
        self.bypass_button.setChecked(valid and entries[row]['bypassed'])
        self.bypass_button.blockSignals(False)
        crashed = valid and entries[row]['crashed']
        self.fault_label.setText(tr('plugins.inserts.fault', fault=entries[row]['fault'] or '--') if crashed else '')
        if not valid:
            self.parameters.show_plugin(None, None)
            return
        instance = self._engine.plugin_instances(self._device_id, self._is_input)[row]
        self.parameters.show_plugin(
            instance, lambda pid, value, index=row: self._engine.set_plugin_parameter(
                self._device_id, index, pid, value, self._is_input))

    def _browse(self):
        browser = PluginBrowser(self._engine, self._config, picking=True, parent=self)
        browser.chosen.connect(self.add)
        browser.exec()

    def add(self, info) -> bool:
        ok, message = self._engine.add_plugin(self._device_id, info, self._is_input)
        if not ok:
            self.fault_label.setText(tr('plugins.inserts.add_failed', reason=message))
            return False
        self.refresh(select=len(self.entries()) - 1)
        self.changed.emit()
        return True

    def _remove(self):
        row = self.chain.currentRow()
        if row >= 0 and self._engine.remove_plugin(self._device_id, row, self._is_input):
            self.refresh(select=row)
            self.changed.emit()

    def _bypass(self, bypassed: bool):
        row = self.chain.currentRow()
        if row >= 0:
            self._engine.set_plugin_bypassed(self._device_id, row, bypassed, self._is_input)
            self.refresh(select=row)
            self.changed.emit()

    def _open_editor(self):
        row = self.chain.currentRow()
        instances = self._engine.plugin_instances(self._device_id, self._is_input)
        if not 0 <= row < len(instances):
            return
        from tonesphere.plugins import PluginError

        try:
            instances[row].open_editor()
        except PluginError as e:
            self.fault_label.setText(tr('plugins.inserts.editor_failed', reason=str(e)))
