"""
Design tokens.

One place for every colour, size and font in the application. The previous UI set colours
inline at each call site, which is how a codebase ends up with four slightly different
greys and no way to change any of them.

The palette is dark because audio software is used in dark rooms and next to other dark
tools, and because meters and level colours need a dark ground to read against. It is
built around a neutral grey ramp with a single amber accent, so that green/amber/red on a
meter always means level and never means "this is a button".
"""

from dataclasses import dataclass
from typing import Tuple

from PySide6.QtGui import QColor, QFont, QFontDatabase


class Colors:
    """
    The palette. Neutrals carry the interface; hues are reserved for meaning.

    Greys step in roughly even perceptual increments so adjacent surfaces separate without
    needing borders everywhere.
    """

    # Surfaces, darkest to lightest.
    BG_BASE = QColor("#0b0d10")        # window
    BG_SUNKEN = QColor("#07080a")      # wells: meter troughs, canvas
    BG_PANEL = QColor("#14171c")       # cards, strips
    BG_RAISED = QColor("#1b1f26")      # controls at rest
    BG_HOVER = QColor("#242932")       # controls under the pointer
    BG_ACTIVE = QColor("#2d333e")      # controls being pressed

    # Lines. Subtle by default; the strong one is for real separation only.
    BORDER = QColor("#232830")
    BORDER_STRONG = QColor("#333a45")

    # Text, in descending emphasis. Three levels is enough; more becomes mud.
    TEXT = QColor("#e8ecf1")
    TEXT_MUTED = QColor("#9aa4b2")
    TEXT_DIM = QColor("#5f6875")

    # One accent. Amber reads as "audio equipment" and does not collide with the
    # green/amber/red of level metering because it is never used at meter scale.
    ACCENT = QColor("#f0883e")
    ACCENT_DIM = QColor("#a85c26")
    ACCENT_TEXT = QColor("#0b0d10")

    # Status. Semantic only.
    OK = QColor("#3fb950")
    WARN = QColor("#d29922")
    ERROR = QColor("#f85149")
    INFO = QColor("#58a6ff")

    # Metering. These are the only place green/amber/red appear, so their meaning is
    # unambiguous: how close to clipping.
    METER_LOW = QColor("#2ea043")      # below -18 dBFS
    METER_MID = QColor("#7ec54b")      # -18 to -6
    METER_HIGH = QColor("#d4a72c")     # -6 to -1
    METER_PEAK = QColor("#f85149")     # above -1
    METER_HOLD = QColor("#e8ecf1")
    METER_BG = QColor("#0e1013")

    # Signal-flow states in the routing view.
    CABLE = QColor("#5f6875")
    CABLE_ACTIVE = QColor("#f0883e")
    CABLE_MUTED = QColor("#3a4049")
    CABLE_DEAD = QColor("#f85149")     # route whose device failed to open

    @staticmethod
    def alpha(color: QColor, a: float) -> QColor:
        faded = QColor(color)
        faded.setAlphaF(a)
        return faded


class Spacing:
    """
    A 4 px base grid.

    Everything is a multiple of it, which is what makes unrelated panels line up without
    anyone measuring anything.
    """
    XS = 2
    SM = 4
    MD = 8
    LG = 12
    XL = 16
    XXL = 24
    SECTION = 32

    RADIUS_SM = 3
    RADIUS = 5
    RADIUS_LG = 8


class Type:
    """
    Type scale.

    Four sizes, not eight. The old UI ranged from 9 pt to 32 pt with no system, so nothing
    established a hierarchy. Numeric readouts are tabular so digits do not jitter as values
    change — critical for a meter readout updating 30 times a second.
    """

    FAMILY = "Segoe UI"
    MONO = "Cascadia Mono, Consolas, monospace"

    DISPLAY = 20
    HEADING = 13
    BODY = 11
    SMALL = 10
    TINY = 9

    # Numeric CSS-style weights mapped to Qt's enum. Callers can pass either, because 600
    # reads more clearly at a call site than QFont.Weight.DemiBold.
    _WEIGHTS = {
        100: QFont.Weight.Thin,
        200: QFont.Weight.ExtraLight,
        300: QFont.Weight.Light,
        400: QFont.Weight.Normal,
        500: QFont.Weight.Medium,
        600: QFont.Weight.DemiBold,
        700: QFont.Weight.Bold,
        800: QFont.Weight.ExtraBold,
        900: QFont.Weight.Black,
    }

    @classmethod
    def font(cls, size: int = BODY, weight=QFont.Weight.Normal, mono: bool = False) -> QFont:
        font = QFont(cls.MONO if mono else cls.FAMILY, size)

        if isinstance(weight, int) and not isinstance(weight, QFont.Weight):
            # PySide6 rejects a bare int here, so snap to the nearest defined weight.
            weight = cls._WEIGHTS.get(weight) or min(
                cls._WEIGHTS.items(), key=lambda item: abs(item[0] - weight)
            )[1]

        font.setWeight(weight)

        if mono:
            # Fixed advance, so a changing number does not shift the characters beside it.
            font.setStyleHint(QFont.StyleHint.Monospace)

        return font

    @classmethod
    def numeric(cls, size: int = SMALL) -> QFont:
        """For any readout that changes: tabular figures, so digits hold their column."""
        font = cls.font(size, mono=True)
        return font


# Metering scale. -60 dBFS is the bottom of the display: below that a fader is at its
# floor anyway, and stretching further wastes the useful part of the scale.
METER_MIN_DB = -60.0
METER_MAX_DB = 6.0

# Where the meter changes colour. These are conventions, not arbitrary: -18 dBFS is a
# common alignment level, and -1 is close enough to full scale to warn.
METER_MID_DB = -18.0
METER_HIGH_DB = -6.0
METER_PEAK_DB = -1.0


def meter_color(db: float) -> QColor:
    """Colour for a level, by how close it is to clipping."""
    if db >= METER_PEAK_DB:
        return Colors.METER_PEAK
    if db >= METER_HIGH_DB:
        return Colors.METER_HIGH
    if db >= METER_MID_DB:
        return Colors.METER_MID
    return Colors.METER_LOW


def db_to_fraction(db: float, minimum: float = METER_MIN_DB,
                   maximum: float = METER_MAX_DB) -> float:
    """
    Position of a dB value on a 0..1 scale, for drawing.

    Linear in dB rather than in amplitude. An amplitude-linear meter spends most of its
    length on the top 6 dB and shows nothing useful in the range people actually mix in.
    """
    if db <= minimum:
        return 0.0
    if db >= maximum:
        return 1.0
    return (db - minimum) / (maximum - minimum)


# Fader taper. A fader linear in dB gives unusable resolution around unity, where the
# fine adjustments happen. Real consoles use a taper that expands the top of the travel;
# this is a simple two-segment approximation of one.
FADER_MIN_DB = -60.0
FADER_MAX_DB = 12.0
_FADER_UNITY_POSITION = 0.75   # where 0 dB sits on the travel


def fader_position_to_db(position: float) -> float:
    """Fader travel (0 at the bottom, 1 at the top) to gain in dB."""
    position = min(max(position, 0.0), 1.0)

    if position >= _FADER_UNITY_POSITION:
        # Top quarter of the travel covers 0 .. +12 dB.
        upper = (position - _FADER_UNITY_POSITION) / (1.0 - _FADER_UNITY_POSITION)
        return upper * FADER_MAX_DB

    # Lower three quarters cover -60 .. 0 dB, curved so the useful region near unity gets
    # more of the travel than the near-silent region at the bottom.
    lower = position / _FADER_UNITY_POSITION
    return FADER_MIN_DB * ((1.0 - lower) ** 1.8)


def db_to_fader_position(db: float) -> float:
    """Inverse of `fader_position_to_db`."""
    if db >= 0.0:
        upper = min(db / FADER_MAX_DB, 1.0)
        return _FADER_UNITY_POSITION + upper * (1.0 - _FADER_UNITY_POSITION)

    if db <= FADER_MIN_DB:
        return 0.0

    ratio = (db / FADER_MIN_DB) ** (1.0 / 1.8)
    return (1.0 - ratio) * _FADER_UNITY_POSITION


def format_db(db: float, places: int = 1) -> str:
    """
    Render a dB value the way a console does.

    Explicit + on positive values, because on a fader the difference between +3 and -3 is
    the whole point and a bare "3" is ambiguous.
    """
    if db <= FADER_MIN_DB:
        return "-∞"
    if db > 0:
        return f"+{db:.{places}f}"
    return f"{db:.{places}f}"


@dataclass(frozen=True)
class Metrics:
    """Fixed sizes for the mixer, so strips align across panels."""
    # Wide enough for a fader and a stereo meter side by side without either being
    # squeezed to uselessness, and for a device name to survive elision.
    STRIP_WIDTH: int = 112
    STRIP_WIDTH_WIDE: int = 136
    FADER_HEIGHT: int = 180
    METER_WIDTH: int = 11
    KNOB_SIZE: int = 46
    BUTTON_HEIGHT: int = 22
    HEADER_HEIGHT: int = 34
    STATUSBAR_HEIGHT: int = 28


METRICS = Metrics()


def stylesheet() -> str:
    """
    Application-wide Qt stylesheet.

    Only the standard widgets are styled here. Meters, faders and the routing view are
    custom-painted, because a stylesheet cannot express a dBFS scale or a bezier cable.
    """
    c = Colors
    s = Spacing

    return f"""
    QWidget {{
        background: {c.BG_BASE.name()};
        color: {c.TEXT.name()};
        font-family: "{Type.FAMILY}";
        font-size: {Type.BODY}pt;
    }}

    QFrame#Panel {{
        background: {c.BG_PANEL.name()};
        border: 1px solid {c.BORDER.name()};
        border-radius: {s.RADIUS}px;
    }}

    QFrame#Sunken {{
        background: {c.BG_SUNKEN.name()};
        border: 1px solid {c.BORDER.name()};
        border-radius: {s.RADIUS_SM}px;
    }}

    QLabel#Heading {{
        color: {c.TEXT.name()};
        font-size: {Type.HEADING}pt;
        font-weight: 600;
    }}

    QLabel#Muted   {{ color: {c.TEXT_MUTED.name()}; }}
    QLabel#Dim     {{ color: {c.TEXT_DIM.name()}; font-size: {Type.SMALL}pt; }}

    /* Strip captions declare their size here rather than relying on setFont(). A
       stylesheet rule wins over setFont, so measuring elision against a font set in code
       while the stylesheet renders a larger one makes long device names overflow the
       strip. Declaring both in the same place keeps measurement and rendering agreed. */
    QLabel#StripName {{
        color: {c.TEXT.name()};
        font-size: {Type.SMALL}pt;
        font-weight: 600;
    }}
    QLabel#StripSub {{
        color: {c.TEXT_DIM.name()};
        font-size: {Type.TINY}pt;
    }}
    QLabel#Numeric {{
        color: {c.TEXT_MUTED.name()};
        font-family: {Type.MONO};
        font-size: {Type.SMALL}pt;
    }}

    QPushButton {{
        background: {c.BG_RAISED.name()};
        border: 1px solid {c.BORDER.name()};
        border-radius: {s.RADIUS_SM}px;
        padding: {s.SM}px {s.LG}px;
        color: {c.TEXT.name()};
    }}
    QPushButton:hover    {{ background: {c.BG_HOVER.name()};
                            border-color: {c.BORDER_STRONG.name()}; }}
    QPushButton:pressed  {{ background: {c.BG_ACTIVE.name()}; }}
    QPushButton:disabled {{ color: {c.TEXT_DIM.name()};
                            background: {c.BG_PANEL.name()}; }}

    QPushButton#Primary {{
        background: {c.ACCENT.name()};
        border-color: {c.ACCENT.name()};
        color: {c.ACCENT_TEXT.name()};
        font-weight: 600;
    }}
    QPushButton#Primary:hover {{ background: {c.ACCENT.lighter(112).name()}; }}

    QPushButton#Danger {{
        background: {c.ERROR.name()};
        border-color: {c.ERROR.name()};
        color: {c.TEXT.name()};
        font-weight: 600;
    }}
    QPushButton#Danger:hover {{ background: {c.ERROR.lighter(110).name()}; }}

    /* Mute and solo latch, so their on-state has to be unmistakable at a glance. */
    QPushButton#Mute:checked {{
        background: {c.ERROR.name()};
        border-color: {c.ERROR.name()};
        color: {c.TEXT.name()};
        font-weight: 700;
    }}
    QPushButton#Solo:checked {{
        background: {c.WARN.name()};
        border-color: {c.WARN.name()};
        color: {c.ACCENT_TEXT.name()};
        font-weight: 700;
    }}

    QComboBox {{
        background: {c.BG_RAISED.name()};
        border: 1px solid {c.BORDER.name()};
        border-radius: {s.RADIUS_SM}px;
        padding: {s.SM}px {s.MD}px;
        min-height: {METRICS.BUTTON_HEIGHT}px;
    }}
    QComboBox:hover {{ border-color: {c.BORDER_STRONG.name()}; }}
    QComboBox::drop-down {{ border: none; width: 18px; }}
    QComboBox QAbstractItemView {{
        background: {c.BG_RAISED.name()};
        border: 1px solid {c.BORDER_STRONG.name()};
        selection-background-color: {c.ACCENT.name()};
        selection-color: {c.ACCENT_TEXT.name()};
        outline: none;
    }}

    QLineEdit, QSpinBox {{
        background: {c.BG_SUNKEN.name()};
        border: 1px solid {c.BORDER.name()};
        border-radius: {s.RADIUS_SM}px;
        padding: {s.SM}px;
        selection-background-color: {c.ACCENT.name()};
    }}
    QLineEdit:focus, QSpinBox:focus {{ border-color: {c.ACCENT.name()}; }}

    QScrollArea {{ border: none; background: transparent; }}

    QScrollBar:vertical {{
        background: transparent; width: 10px; margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: {c.BG_ACTIVE.name()};
        border-radius: 5px;
        min-height: 24px;
    }}
    QScrollBar::handle:vertical:hover {{ background: {c.BORDER_STRONG.name()}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

    QScrollBar:horizontal {{
        background: transparent; height: 10px; margin: 0;
    }}
    QScrollBar::handle:horizontal {{
        background: {c.BG_ACTIVE.name()};
        border-radius: 5px;
        min-width: 24px;
    }}

    QToolTip {{
        background: {c.BG_ACTIVE.name()};
        color: {c.TEXT.name()};
        border: 1px solid {c.BORDER_STRONG.name()};
        padding: {s.SM}px {s.MD}px;
    }}

    QMenuBar {{ background: {c.BG_PANEL.name()}; }}
    QMenuBar::item:selected {{ background: {c.BG_HOVER.name()}; }}
    QMenu {{
        background: {c.BG_RAISED.name()};
        border: 1px solid {c.BORDER_STRONG.name()};
        padding: {s.SM}px;
    }}
    QMenu::item {{ padding: {s.SM}px {s.XL}px; border-radius: {s.RADIUS_SM}px; }}
    QMenu::item:selected {{ background: {c.ACCENT.name()};
                            color: {c.ACCENT_TEXT.name()}; }}

    QSplitter::handle {{ background: {c.BORDER.name()}; }}
    QSplitter::handle:horizontal {{ width: 1px; }}
    QSplitter::handle:vertical {{ height: 1px; }}
    """
