"""
Effects: the VST3 browser that scans for plugins, and the insert chain of one device side
or bus — ToneSphere's built-in EQ, compressor, limiter and delay, and VST3 plugins, in one
ordered chain.

Scanning runs each module in a subprocess (`tonesphere.plugins.scan`), which takes seconds
on a cold cache, so it runs on a worker thread and the window stays live. A module that
crashed, hung or is the wrong architecture is listed with its reason, never hidden: the
user installed it, and deserves to know why it is not offered.

Parameter values shown here are the plugin's own: its display text, read back after every
change, not a number this window computed.

Nothing here calls the engine or a plugin on the main thread: a plugin answers on its own
thread and can be slow to, so every call goes through the engine worker (`ui/tasks.py`)
and the result is shown when it arrives.
"""

from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
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
    QMenu,
    QPushButton,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from tonesphere.i18n import tr
from tonesphere.ui.tasks import EngineTasks
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


class PluginBrowser(QDialog):
    """
    Every VST3 module in the standard folders and the user's own, with what the scan found.
    Opened from an insert chain, it also picks one: `chosen` carries its `PluginInfo`.
    """

    chosen = Signal(object)

    COLUMNS = 7

    def __init__(self, engine, config_manager=None, picking: bool = False, auto_scan: bool = True,
                 parent: QWidget | None = None, instruments: bool = False):
        super().__init__(parent)
        self._engine = engine
        self._config = config_manager
        self._picking = picking
        # Picking for a chain offers effects; picking an instrument offers instruments.
        self._instruments = instruments
        self._tasks = EngineTasks(self, name='plugin-scan')
        self._scanning = False
        self._closed = False
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
        # Pressing Scan is asking again: modules whose last scan failed are loaded afresh.
        self.scan_button.clicked.connect(lambda: self.scan(retry_failed=True))
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

    def scan(self, retry_failed: bool = False):
        if self._scanning:
            return
        self._scanning = True
        self.scan_button.setEnabled(False)
        self.status_label.setText(tr('plugins.browser.scanning'))
        roots = self.roots()
        self._tasks.submit(lambda: self._engine.scan_plugins(roots, retry_failed), self._scanned, self._scan_failed)

    def _scanned(self, results: list):
        self._scanning = False
        if not self._closed:
            self.scan_button.setEnabled(True)
            self.show_results(results)

    def _scan_failed(self, error: BaseException):
        self._scanning = False
        logger.warning(f"plugin scan failed: {error}")
        if not self._closed:
            self.scan_button.setEnabled(True)
            self.status_label.setText(tr('plugins.browser.scan_failed', reason=str(error)))

    def show_results(self, results: list):
        self.table.setRowCount(0)
        self._rows = []
        usable = 0
        for result in results:
            if result.effects:
                for info in result.effects:
                    offered = info.is_instrument == self._instruments
                    if info.is_instrument:
                        status, detail = tr('plugins.status.instrument'), tr('plugins.instrument_detail')
                    else:
                        status, detail = status_text(result.status), result.detail or tr('plugins.effect_detail')
                    self._add_row(info.name, info.vendor, info.subcategories or info.category, info.version,
                                  result.architecture, status, result.path, detail, info if offered else None)
                    usable += offered
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

    def done(self, result):
        # A scan still running finishes on its own thread; its result is then dropped.
        self._closed = True
        super().done(result)


class ParameterPanel(QWidget):
    """
    One plugin's parameters, filtered by name, and a slider for the selected one. Reading
    and setting both go through `tasks`: a plugin answers on its own thread.
    """

    def __init__(self, tasks: EngineTasks, parent: QWidget | None = None):
        super().__init__(parent)
        self._tasks = tasks
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
        if instance is None:
            self._show_parameters(None, [])
            return
        self._tasks.submit(instance.parameters, lambda params: self._show_parameters(instance, params))

    def _show_parameters(self, instance, parameters: list):
        if instance is not self._instance:
            return   # another plugin was selected while these were being read
        self._parameters = parameters
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
        instance, setter = self._instance, self._setter

        def change():
            setter(p.id, normalized)
            return instance.parameters()

        self._tasks.submit(change, lambda params: self._read_back(instance, row, params))

    def refresh(self, row: int | None = None):
        """Read the plugin's own values back: what it says, not what was sent."""
        instance = self._instance
        if instance is not None:
            self._tasks.submit(instance.parameters, lambda params: self._read_back(instance, row, params))

    def _read_back(self, instance, row: int | None, parameters: list):
        if instance is not self._instance:
            return
        self._parameters = parameters
        rows = [row] if row is not None else range(len(self._parameters))
        for r in rows:
            if r < len(self._parameters) and self.table.item(r, 1) is not None:
                self.table.item(r, 1).setText(self._parameters[r].display)
        if row is not None and row < len(self._parameters):
            self.value_label.setText(self._parameters[row].display)


class InsertsDialog(QDialog):
    """The effects on one side of one device (or on a bus), in processing order."""

    changed = Signal()

    def __init__(self, engine, device_id: int, is_input: bool, device_name: str, config_manager=None,
                 parent: QWidget | None = None, tasks: EngineTasks | None = None):
        super().__init__(parent)
        self._engine = engine
        self._device_id = device_id
        self._is_input = is_input
        self._config = config_manager
        self._tasks = tasks or EngineTasks(self, name='inserts')
        self._entries: list[dict] = []
        self._closed = False

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
        self.builtin_button = QPushButton(tr('plugins.inserts.add_builtin'))
        builtin_menu = QMenu(self.builtin_button)
        from tonesphere.engine.builtins import KINDS
        for kind in KINDS:
            builtin_menu.addAction(tr(f'builtin.{kind}'), lambda k=kind: self.add_builtin(k))
        self.builtin_button.setMenu(builtin_menu)
        self.add_button = QPushButton(tr('plugins.inserts.add'))
        self.add_button.clicked.connect(self._browse)
        self.up_button = QPushButton(tr('plugins.inserts.move_up'))
        self.up_button.clicked.connect(lambda: self._move(-1))
        self.down_button = QPushButton(tr('plugins.inserts.move_down'))
        self.down_button.clicked.connect(lambda: self._move(1))
        self.remove_button = QPushButton(tr('plugins.inserts.remove'))
        self.remove_button.clicked.connect(self._remove)
        self.bypass_button = QPushButton(tr('plugins.inserts.bypass'))
        self.bypass_button.setCheckable(True)
        self.bypass_button.setToolTip(tr('plugins.inserts.bypass_tooltip'))
        self.bypass_button.toggled.connect(self._bypass)
        self.editor_button = QPushButton(tr('plugins.inserts.open_editor'))
        self.editor_button.clicked.connect(self._open_editor)
        self.keyboard_button = QPushButton(tr('plugins.inserts.keyboard'))
        self.keyboard_button.clicked.connect(self._open_keyboard)
        for b in (self.builtin_button, self.add_button, self.remove_button):
            buttons.addWidget(b)
        left_layout.addLayout(buttons)
        order = QHBoxLayout()
        for b in (self.up_button, self.down_button, self.bypass_button, self.editor_button, self.keyboard_button):
            order.addWidget(b)
        left_layout.addLayout(order)

        self.fault_label = QLabel()
        self.fault_label.setWordWrap(True)
        self.fault_label.setStyleSheet(f"color: {Colors.ERROR.name()};")
        left_layout.addWidget(self.fault_label)
        splitter.addWidget(left)

        self.parameters = ParameterPanel(self._tasks)
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

    @property
    def tasks(self) -> EngineTasks:
        return self._tasks

    def entries(self) -> list[dict]:
        """The chain as last read from the engine."""
        return list(self._entries)

    def _submit(self, fn, done=None, failed=None):
        """Results for a closed dialog are dropped."""
        self._tasks.submit(fn, lambda r: None if self._closed or done is None else done(r),
                           lambda e: None if self._closed or failed is None else failed(e))

    def refresh(self, select: int | None = None):
        current = self.chain.currentRow() if select is None else select
        self._submit(lambda: self._engine.list_inserts(self._device_id, self._is_input),
                     lambda entries: self._show_entries(entries, current))

    def _show_entries(self, entries: list[dict], current: int):
        self._entries = entries
        self.chain.blockSignals(True)
        self.chain.clear()
        for e in entries:
            if e['kind'] == 'builtin':
                text = tr('plugins.inserts.builtin_entry', name=e['name'])
                if e['gain_reduction_db']:
                    text += tr('plugins.inserts.reduction', db=f"{e['gain_reduction_db']:.1f}")
            else:
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
            row = min(max(current, 0), len(entries) - 1)
            self.chain.setCurrentRow(row)
            self._row_changed(row)
        else:
            self._row_changed(-1)

    def _row_changed(self, row: int):
        entries = self._entries
        valid = 0 <= row < len(entries)
        self.remove_button.setEnabled(valid)
        self.up_button.setEnabled(valid and row > 0)
        self.down_button.setEnabled(valid and row < len(entries) - 1)
        self.bypass_button.setEnabled(valid and not (valid and entries[row]['crashed']))
        self.editor_button.setEnabled(valid and entries[row]['has_editor'] and not entries[row]['crashed'])
        self.keyboard_button.setEnabled(valid and entries[row].get('instrument', False) and not entries[row]['crashed'])
        self.bypass_button.blockSignals(True)
        self.bypass_button.setChecked(valid and entries[row]['bypassed'])
        self.bypass_button.blockSignals(False)
        crashed = valid and entries[row]['crashed']
        self.fault_label.setText(tr('plugins.inserts.fault', fault=entries[row]['fault'] or '--') if crashed else '')
        if not valid:
            self.parameters.show_plugin(None, None)
            return
        device, is_input = self._device_id, self._is_input

        def show(instance):
            if self.chain.currentRow() == row and instance is not None:
                self.parameters.show_plugin(instance, lambda pid, value: self._engine.set_insert_parameter(
                    device, row, pid, value, is_input))

        self._submit(lambda: self._engine.insert_instance(device, row, is_input), show)

    def _browse(self):
        browser = PluginBrowser(self._engine, self._config, picking=True, parent=self)
        browser.chosen.connect(self.add)
        browser.exec()

    def add(self, info):
        """Insert `info` at the end of the chain; the result shows when the plugin has opened."""
        def added(result):
            ok, message = result
            if not ok:
                self.fault_label.setText(tr('plugins.inserts.add_failed', reason=message))
                return
            self.refresh(select=len(self._entries))
            self.changed.emit()

        self._submit(lambda: self._engine.add_plugin(self._device_id, info, self._is_input), added,
                     lambda e: self.fault_label.setText(tr('plugins.inserts.add_failed', reason=str(e))))

    def add_builtin(self, kind: str):
        """Append a built-in effect; the chain shows it once the engine has it."""
        def added(result):
            ok, message = result
            if not ok:
                self.fault_label.setText(tr('plugins.inserts.add_failed', reason=message))
                return
            self.refresh(select=len(self._entries))
            self.changed.emit()

        self._submit(lambda: self._engine.add_builtin(self._device_id, kind, self._is_input), added)

    def _move(self, step: int):
        row = self.chain.currentRow()
        if row < 0:
            return

        def moved(ok):
            if ok:
                self.refresh(select=row + step)
                self.changed.emit()

        self._submit(lambda: self._engine.move_insert(self._device_id, row, row + step, self._is_input), moved)

    def _remove(self):
        row = self.chain.currentRow()
        if row < 0:
            return

        def removed(ok):
            if ok:
                self.refresh(select=row)
                self.changed.emit()

        self._submit(lambda: self._engine.remove_insert(self._device_id, row, self._is_input), removed)

    def _bypass(self, bypassed: bool):
        row = self.chain.currentRow()
        if row < 0:
            return

        def done(_ok):
            self.refresh(select=row)
            self.changed.emit()

        self._submit(lambda: self._engine.set_insert_bypassed(self._device_id, row, bypassed, self._is_input), done)

    def _open_editor(self):
        row = self.chain.currentRow()
        if row < 0:
            return

        def open_editor():
            instance = self._engine.insert_instance(self._device_id, row, self._is_input)
            if instance is not None and not getattr(instance, 'is_builtin', False):
                instance.open_editor()

        self._submit(open_editor, None,
                     lambda e: self.fault_label.setText(tr('plugins.inserts.editor_failed', reason=str(e))))

    def _open_keyboard(self):
        from tonesphere.ui.keyboard import KeyboardDialog

        KeyboardDialog(self._engine, self._device_id, self.windowTitle(), self, tasks=self._tasks).show()

    def done(self, result):
        self._closed = True
        super().done(result)
