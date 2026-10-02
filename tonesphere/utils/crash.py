"""
What happens when ToneSphere fails where nobody is watching.

A windowed build has no console: an exception during start-up, on a worker thread, or in a
Qt slot used to vanish, and the user saw an application that simply never appeared. Every
uncaught exception now lands in a crash report in the logs folder, the log says where, and
when the interface is up a dialog names the file.
"""

import platform
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path

from tonesphere.utils.logger import default_log_dir, get_logger

logger = get_logger(__name__)

_installed = False
_showing = threading.Lock()


def write_report(exc_type, exc, tb, where: str) -> Path | None:
    """The crash report on disk, or None if even that could not be written."""
    from tonesphere import __version__

    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    path = default_log_dir() / f'crash-{stamp}.log'
    text = (f"ToneSphere {__version__} crashed ({where})\n"
            f"{datetime.now().isoformat()}  {platform.platform()}  Python {platform.python_version()}"
            f"{'  frozen' if getattr(sys, 'frozen', False) else ''}\n\n"
            + ''.join(traceback.format_exception(exc_type, exc, tb)))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')
    except OSError:
        return None
    return path


def _show(path: Path | None, exc):
    """A dialog naming the report, when there is an interface to show it in."""
    try:
        from PySide6.QtCore import QThread
        from PySide6.QtWidgets import QApplication, QMessageBox
    except ImportError:
        return
    app = QApplication.instance()
    if app is None or QThread.currentThread() is not app.thread() or not _showing.acquire(blocking=False):
        return
    try:
        from tonesphere.i18n import tr
        QMessageBox.critical(None, tr('crash.title'), tr('crash.body', error=str(exc) or type(exc).__name__,
                                                          path=str(path) if path else '--'))
    finally:
        _showing.release()


def _handle(exc_type, exc, tb, where: str):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc, tb)
        return
    path = write_report(exc_type, exc, tb, where)
    logger.critical(f"Unhandled exception ({where}); report: {path}", exc_info=(exc_type, exc, tb))
    _show(path, exc)


def install():
    """Route every uncaught exception — main thread, other threads, Qt — to a crash report."""
    global _installed
    if _installed:
        return
    _installed = True
    sys.excepthook = lambda t, e, tb: _handle(t, e, tb, 'main thread')
    threading.excepthook = lambda a: _handle(a.exc_type, a.exc_value, a.exc_traceback,
                                             f'thread {a.thread.name if a.thread else "?"}')


def install_qt_messages():
    """Qt's own warnings and errors go to the log instead of a console nobody has."""
    from PySide6.QtCore import QtMsgType, qInstallMessageHandler

    levels = {QtMsgType.QtDebugMsg: logger.debug, QtMsgType.QtInfoMsg: logger.info,
              QtMsgType.QtWarningMsg: logger.warning, QtMsgType.QtCriticalMsg: logger.error,
              QtMsgType.QtFatalMsg: logger.critical}
    qInstallMessageHandler(lambda kind, context, message: levels.get(kind, logger.warning)(f"Qt: {message}"))
