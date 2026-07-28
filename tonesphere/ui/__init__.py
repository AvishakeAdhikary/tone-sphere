"""
ToneSphere's Qt interface.

Replaces the Tkinter UI. Tkinter was the ceiling on how this could look, not the styling:
no GPU compositing, no proper DPI scaling, and no way to custom-paint a dBFS meter, a
tapered fader or a bezier patch cable. Those are the parts that make audio software look
like audio software.

Layering is strict. Widgets emit intent and never touch the engine; `MainWindow` owns that
connection. The audio callback never touches Qt — it writes measurements into plain slots
that the UI reads on a timer.
"""

from tonesphere.ui.app import create_app, run

__all__ = ["create_app", "run"]
