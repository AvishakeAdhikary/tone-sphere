"""
The ToneSphere virtual cables: each one a device Windows lists, with an output other programs
play into and an input they record from. Add, rename, disable, enable or uninstall one at a
time, or remove the driver.

Every change is made by Windows' device installer, which asks for an administrator's consent
with its own prompt (`engine/virtual_cables.py`); nothing here runs elevated. Calls go
through the engine worker (`ui/tasks.py`), so the prompt and the device changes never block
the window.
"""

from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from tonesphere.i18n import tr
from tonesphere.ui.tasks import EngineTasks
from tonesphere.ui.theme import Colors, Spacing
from tonesphere.utils.formatting import UNKNOWN


def state_text(state: str) -> str:
    """A cable's state as `engine/virtual_cables.Cable.state` gives it, in words."""
    kind = state.split()[0]
    if kind == 'working':
        return tr('cables.state.working')
    if kind == 'disabled':
        return tr('cables.state.disabled')
    return tr('cables.state.problem')


class CablesDialog(QDialog):
    def __init__(self, engine, parent: QWidget | None = None, tasks: EngineTasks | None = None):
        super().__init__(parent)
        self._engine = engine
        self._tasks = tasks or EngineTasks(self, name='cables')
        self._cables: list[dict] = []
        self._closed = False
        self.setWindowTitle(tr('cables.title'))
        self.resize(820, 420)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(Spacing.LG, Spacing.LG, Spacing.LG, Spacing.LG)
        layout.setSpacing(Spacing.MD)
        self.hint = QLabel(tr('cables.hint'))
        self.hint.setObjectName("Dim")
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels([tr('cables.column.name'), tr('cables.column.state'),
                                              tr('cables.column.output'), tr('cables.column.input')])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        layout.addWidget(self.table, stretch=1)

        buttons = QHBoxLayout()
        self.add_button = QPushButton(tr('cables.add'))
        self.add_button.clicked.connect(self._ask_add)
        self.rename_button = QPushButton(tr('cables.rename'))
        self.rename_button.clicked.connect(self._ask_rename)
        self.toggle_button = QPushButton(tr('cables.disable'))
        self.toggle_button.clicked.connect(self.toggle_selected)
        self.remove_button = QPushButton(tr('cables.uninstall'))
        self.remove_button.clicked.connect(self._ask_remove)
        self.remove_driver_button = QPushButton(tr('cables.remove_driver'))
        self.remove_driver_button.setObjectName("Danger")
        self.remove_driver_button.clicked.connect(self._ask_remove_driver)
        for b in (self.add_button, self.rename_button, self.toggle_button, self.remove_button):
            buttons.addWidget(b)
        buttons.addStretch()
        buttons.addWidget(self.remove_driver_button)
        layout.addLayout(buttons)

        bottom = QHBoxLayout()
        self.status = QLabel()
        self.status.setWordWrap(True)
        bottom.addWidget(self.status, stretch=1)
        self.close_button = QPushButton(tr('dialog.close'))
        self.close_button.clicked.connect(self.accept)
        bottom.addWidget(self.close_button)
        layout.addLayout(bottom)

        self._busy(True)
        self.refresh()

    @property
    def tasks(self) -> EngineTasks:
        return self._tasks

    # --- Reading ---

    def refresh(self):
        self._tasks.submit(self._engine.virtual_device_status, self._show)

    def _show(self, status: dict):
        if self._closed:
            return
        self._cables = status['cables']
        selected = self.selected()
        self.table.setRowCount(0)
        for cable in self._cables:
            row = self.table.rowCount()
            self.table.insertRow(row)
            state = state_text(cable['state'])
            for column, text in enumerate((cable['name'], state,
                                           (cable['render'] or {}).get('name', UNKNOWN),
                                           (cable['capture'] or {}).get('name', UNKNOWN))):
                item = QTableWidgetItem(text)
                if column == 1 and cable['state'] != 'working':
                    item.setForeground(Colors.WARN)
                self.table.setItem(row, column, item)
        if selected is not None:
            for row, cable in enumerate(self._cables):
                if cable['instance_id'] == selected['instance_id']:
                    self.table.selectRow(row)
        self._driver_installed = status.get('driver_installed', True)
        if not status['platform_supported']:
            self.status.setText(tr('cables.windows_only'))
        elif status['error']:
            self.status.setText(status['error'])
        elif not self._driver_installed:
            self.status.setText(tr('cables.no_driver'))
        elif not self._cables:
            self.status.setText(tr('cables.none'))
        self._busy(False)
        self._selection_changed()

    def selected(self) -> dict | None:
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        return self._cables[rows[0].row()] if rows and rows[0].row() < len(self._cables) else None

    def _selection_changed(self):
        cable = self.selected()
        for b in (self.rename_button, self.toggle_button, self.remove_button):
            b.setEnabled(cable is not None)
        self.toggle_button.setText(tr('cables.enable') if cable and not cable['enabled'] else tr('cables.disable'))

    def _busy(self, busy: bool):
        for b in (self.add_button, self.remove_driver_button):
            b.setEnabled(not busy and getattr(self, '_driver_installed', True))
        if busy:
            self.status.setText(tr('status.working'))

    # --- Changing (each one a Windows administrator prompt) ---

    def _manage(self, operation: str, *args: str):
        self._busy(True)

        def done(result):
            ok, message = result
            if self._closed:
                return
            self.status.setText('' if ok else message)
            self.status.setStyleSheet('' if ok else f"color: {Colors.ERROR.name()};")
            self.refresh()

        self._tasks.submit(lambda: self._engine.manage_virtual_cable(operation, *args), done)

    def add_cable(self, name: str):
        self._manage('add', name)

    def rename_selected(self, name: str):
        cable = self.selected()
        if cable is not None and name:
            self._manage('rename', cable['instance_id'], name)

    def toggle_selected(self):
        cable = self.selected()
        if cable is not None:
            self._manage('enable' if not cable['enabled'] else 'disable', cable['instance_id'])

    def remove_selected(self):
        cable = self.selected()
        if cable is not None:
            self._manage('remove', cable['instance_id'])

    def remove_driver(self):
        self._manage('remove-driver')

    def _ask_add(self):
        name, ok = QInputDialog.getText(self, tr('cables.add'), tr('cables.name_prompt'),
                                        text=tr('cables.default_name', number=len(self._cables) + 1))
        if ok and name.strip():
            self.add_cable(name.strip())

    def _ask_rename(self):
        cable = self.selected()
        if cable is None:
            return
        name, ok = QInputDialog.getText(self, tr('cables.rename'), tr('cables.name_prompt'), text=cable['name'])
        if ok and name.strip():
            self.rename_selected(name.strip())

    def _ask_remove(self):
        cable = self.selected()
        if cable is not None and QMessageBox.question(
                self, tr('cables.uninstall'), tr('cables.uninstall_confirm', name=cable['name'])) == \
                QMessageBox.StandardButton.Yes:
            self.remove_selected()

    def _ask_remove_driver(self):
        if QMessageBox.question(self, tr('cables.remove_driver'), tr('cables.remove_driver_confirm')) == \
                QMessageBox.StandardButton.Yes:
            self.remove_driver()

    def done(self, result):
        self._closed = True
        super().done(result)
