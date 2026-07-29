"""
Application entry point for the Qt interface.
"""

import sys

from PySide6.QtGui import QIcon, QPalette
from PySide6.QtWidgets import QApplication

from tonesphere.ui.theme import Colors, Type, stylesheet
from tonesphere.utils.config import ConfigManager
from tonesphere.utils.logger import get_logger

logger = get_logger(__name__)


def _apply_palette(app: QApplication):
    """
    Set the palette as well as the stylesheet.

    The stylesheet covers widgets we draw; the palette covers what Qt draws for us —
    tooltips, selection highlights, native dialogs. Setting only one leaves light-coloured
    system chrome punched through a dark interface.
    """
    palette = QPalette()

    palette.setColor(QPalette.ColorRole.Window, Colors.BG_BASE)
    palette.setColor(QPalette.ColorRole.WindowText, Colors.TEXT)
    palette.setColor(QPalette.ColorRole.Base, Colors.BG_SUNKEN)
    palette.setColor(QPalette.ColorRole.AlternateBase, Colors.BG_PANEL)
    palette.setColor(QPalette.ColorRole.Text, Colors.TEXT)
    palette.setColor(QPalette.ColorRole.Button, Colors.BG_RAISED)
    palette.setColor(QPalette.ColorRole.ButtonText, Colors.TEXT)
    palette.setColor(QPalette.ColorRole.Highlight, Colors.ACCENT)
    palette.setColor(QPalette.ColorRole.HighlightedText, Colors.ACCENT_TEXT)
    palette.setColor(QPalette.ColorRole.ToolTipBase, Colors.BG_ACTIVE)
    palette.setColor(QPalette.ColorRole.ToolTipText, Colors.TEXT)
    palette.setColor(QPalette.ColorRole.PlaceholderText, Colors.TEXT_DIM)

    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, Colors.TEXT_DIM)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText,
                     Colors.TEXT_DIM)

    app.setPalette(palette)


def create_app(argv: list | None = None) -> QApplication:
    """Build the QApplication with theme and DPI handling in place."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(argv if argv is not None else sys.argv)

    app.setApplicationName("ToneSphere")
    app.setOrganizationName("ToneSphere")

    # Fusion draws consistently across platforms and takes a palette properly; the native
    # Windows style ignores much of it and would leave the UI half light.
    app.setStyle("Fusion")

    _apply_palette(app)
    app.setStyleSheet(stylesheet())
    app.setFont(Type.font())

    icon_path = "assets/images/ToneSphere.png"
    try:
        icon = QIcon(icon_path)
        if not icon.isNull():
            app.setWindowIcon(icon)
    except Exception:
        pass

    return app


def run(config_manager: ConfigManager | None = None) -> int:
    """Launch the interface. Returns the process exit code."""
    from tonesphere.ui.main_window import MainWindow

    app = create_app()

    window = MainWindow(config_manager)
    window.show()

    return app.exec()
