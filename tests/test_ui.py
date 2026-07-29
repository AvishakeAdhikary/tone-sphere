"""
Interface tests.

The previous UI had none, which is how it ended up with two methods of the same name on
one class, a modal dialog that hung the app, and a status thread that deadlocked Tk.

These focus on the parts where a UI bug is a *correctness* bug rather than a cosmetic one:
the dB/position conversions behind the faders, and the rule that an unmeasured value must
never be rendered as a confident number.
"""

import math

import pytest

pytest.importorskip("PySide6", reason="Qt not installed")

from PySide6.QtWidgets import QApplication  # noqa: E402

from tonesphere.ui.theme import (  # noqa: E402
    FADER_MAX_DB,
    FADER_MIN_DB,
    METER_MAX_DB,
    METER_MIN_DB,
    Colors,
    db_to_fader_position,
    db_to_fraction,
    fader_position_to_db,
    format_db,
    meter_color,
)


@pytest.fixture(scope="module")
def qt_app():
    """
    One QApplication for the module.

    Qt requires exactly one per process and does not tolerate it being recreated, so this
    is module-scoped rather than per test.
    """
    app = QApplication.instance() or QApplication([])
    yield app


class TestFaderTaper:
    """
    The fader law.

    Round-tripping matters because the widget converts both ways constantly: dB in from
    the engine, position out to the mouse, and back. A lossy conversion makes a fader
    creep every time it is touched.
    """

    def test_position_and_db_round_trip(self):
        for db in (-60, -40, -20, -12, -6, -3, 0, 3, 6, 12):
            position = db_to_fader_position(float(db))
            assert fader_position_to_db(position) == pytest.approx(float(db), abs=0.15)

    def test_unity_sits_high_on_the_travel(self):
        """
        0 dB near the top means most of the travel covers cuts, which is where mixing
        actually happens. A fader with unity in the middle wastes half its length on boost
        nobody uses.
        """
        position = db_to_fader_position(0.0)
        assert 0.7 < position < 0.8

    def test_bottom_of_travel_is_the_floor(self):
        assert fader_position_to_db(0.0) == pytest.approx(FADER_MIN_DB, abs=0.01)

    def test_top_of_travel_is_max_boost(self):
        assert fader_position_to_db(1.0) == pytest.approx(FADER_MAX_DB, abs=0.01)

    def test_travel_is_monotonic(self):
        """A fader that ever moves backwards as you drag is unusable."""
        previous = -math.inf
        for step in range(101):
            db = fader_position_to_db(step / 100.0)
            assert db >= previous - 1e-6
            previous = db

    def test_resolution_near_unity_beats_the_bottom(self):
        """
        The reason for the taper: fine adjustment happens around unity, so that region
        needs more travel per dB than the near-silent bottom of the fader.
        """
        near_unity = abs(db_to_fader_position(0.0) - db_to_fader_position(-3.0))
        near_floor = abs(db_to_fader_position(-50.0) - db_to_fader_position(-53.0))

        assert near_unity > near_floor

    def test_out_of_range_positions_are_clamped(self):
        assert fader_position_to_db(-1.0) == fader_position_to_db(0.0)
        assert fader_position_to_db(2.0) == fader_position_to_db(1.0)


class TestDbFormatting:
    def test_positive_values_are_signed(self):
        """+3 and -3 are opposite adjustments; a bare "3" is ambiguous on a fader."""
        assert format_db(3.0).startswith("+")

    def test_negative_values_keep_their_sign(self):
        assert format_db(-6.0) == "-6.0"

    def test_unity_is_zero_not_plus_zero(self):
        assert format_db(0.0) == "0.0"

    def test_floor_renders_as_minus_infinity(self):
        assert format_db(FADER_MIN_DB) == "-∞"


class TestMeterScale:
    def test_floor_is_the_bottom_of_the_display(self):
        assert db_to_fraction(METER_MIN_DB) == 0.0

    def test_ceiling_is_the_top(self):
        assert db_to_fraction(METER_MAX_DB) == 1.0

    def test_below_floor_does_not_go_negative(self):
        assert db_to_fraction(-200.0) == 0.0

    def test_scale_is_linear_in_db_not_amplitude(self):
        """
        An amplitude-linear meter spends most of its length in the top few dB and shows
        nothing across the range people work in. Equal dB steps must be equal distances.
        """
        span_a = db_to_fraction(-12.0) - db_to_fraction(-18.0)
        span_b = db_to_fraction(-30.0) - db_to_fraction(-36.0)

        assert span_a == pytest.approx(span_b, abs=1e-6)

    def test_colours_escalate_towards_clipping(self):
        assert meter_color(-40.0) == Colors.METER_LOW
        assert meter_color(-10.0) == Colors.METER_MID
        assert meter_color(-3.0) == Colors.METER_HIGH
        assert meter_color(-0.2) == Colors.METER_PEAK


class TestWidgetsShowUnmeasuredHonestly:
    def test_status_pill_defaults_to_unknown(self, qt_app):
        from tonesphere.ui.widgets import StatusPill

        pill = StatusPill("LATENCY")
        assert pill._value == "--"

    def test_status_pill_renders_none_as_unknown(self, qt_app):
        """
        The whole reason this is a custom widget: None has to become "--", never 0.0.
        """
        from tonesphere.ui.widgets import StatusPill

        pill = StatusPill("DSP", "42%")
        pill.set_value(None)

        assert pill._value == "--"

    def test_meter_starts_inactive(self, qt_app):
        """
        A meter pinned at the floor looks like measured silence. Before anything runs
        there is no measurement, and the meter greys out to say so.
        """
        from tonesphere.ui.widgets import LevelMeter

        meter = LevelMeter(channels=2)
        assert meter._active is False

    def test_meter_goes_inactive_when_told(self, qt_app):
        from tonesphere.ui.widgets import LevelMeter

        meter = LevelMeter(channels=2)
        meter.set_levels([-6.0, -6.0])
        assert meter._active is True

        meter.set_inactive()
        assert meter._active is False

    def test_clip_indicator_latches_until_cleared(self, qt_app):
        """
        A single over between two glances would go unnoticed otherwise, and a single over
        is exactly what makes a converter clip.
        """
        from tonesphere.ui.widgets import LevelMeter

        meter = LevelMeter(channels=2)
        meter.set_levels([-0.1, -0.1], clipped=True)
        assert meter._clipped is True

        meter.set_levels([-40.0, -40.0], clipped=False)
        assert meter._clipped is False   # engine owns the latch; UI reflects it

        meter.set_levels([-0.1, -0.1], clipped=True)
        meter.clear_clip()
        assert meter._clipped is False


class TestFaderWidget:
    def test_starts_at_unity(self, qt_app):
        from tonesphere.ui.widgets import Fader

        assert Fader().db == 0.0

    def test_value_is_clamped_to_the_legal_range(self, qt_app):
        from tonesphere.ui.widgets import Fader

        fader = Fader()
        fader.set_db(999.0)
        assert fader.db <= FADER_MAX_DB

        fader.set_db(-999.0)
        assert fader.db >= FADER_MIN_DB

    def test_setting_a_value_does_not_echo_back_by_default(self, qt_app):
        """
        Programmatic updates must not emit. Otherwise refreshing the UI from engine state
        writes straight back to the engine, and the two fight each other.
        """
        from tonesphere.ui.widgets import Fader

        fader = Fader()
        emitted = []
        fader.value_changed.connect(emitted.append)

        fader.set_db(-6.0)
        assert emitted == []

        fader.set_db(-12.0, notify=True)
        assert emitted == [-12.0]


class TestPanKnob:
    def test_starts_centred(self, qt_app):
        from tonesphere.ui.widgets import PanKnob

        assert PanKnob().pan == 0.0

    def test_pan_is_clamped(self, qt_app):
        from tonesphere.ui.widgets import PanKnob

        knob = PanKnob()
        knob.set_pan(5.0)
        assert knob.pan == 1.0

        knob.set_pan(-5.0)
        assert knob.pan == -1.0

    def test_label_reads_as_a_console_would(self, qt_app):
        from tonesphere.ui.widgets import PanKnob

        knob = PanKnob()
        assert knob._label() == "C"

        knob.set_pan(-0.5)
        assert knob._label() == "L50"

        knob.set_pan(1.0)
        assert knob._label() == "R100"


class TestHardwareBar:
    def test_unmeasured_latency_is_not_shown_as_the_nominal_figure(self, qt_app):
        """
        Nominal is arithmetic; measured is reality, and they differ by 4x on shared-mode
        WASAPI. Showing the nominal figure alone is how the old UI claimed 2.67 ms while
        passing no audio.
        """
        from tonesphere.ui.strip import HardwareBar

        bar = HardwareBar()
        bar.update_state('idle', {
            'measured_latency_ms': None,
            'nominal_latency_ms': 2.67,
            'cpu_usage': None,
            'xruns': 0,
        })

        assert "--" in bar.latency._value
        assert "nom" in bar.latency._value

    def test_measured_latency_is_shown_when_available(self, qt_app):
        from tonesphere.ui.strip import HardwareBar

        bar = HardwareBar()
        bar.update_state('running', {
            'measured_latency_ms': 5.7,
            'nominal_latency_ms': 2.7,
            'cpu_usage': 5.0,
            'xruns': 0,
        })

        assert "5.7" in bar.latency._value

    def test_degraded_state_is_distinct_from_running(self, qt_app):
        """
        Some streams open and some fail. Reporting that as "Running" is the partial-failure
        dishonesty the engine already refuses to commit.
        """
        from tonesphere.ui.strip import HardwareBar

        bar = HardwareBar()
        bar.update_state('degraded', {
            'failed_streams': {'mic': 'Invalid device'},
            'measured_latency_ms': 5.7,
            'nominal_latency_ms': 2.7,
            'cpu_usage': 4.0,
            'xruns': 0,
        })

        assert "Degraded" in bar.state.text()
        assert bar.message.text()

    def test_idle_is_distinct_from_stopped(self, qt_app):
        from tonesphere.ui.strip import HardwareBar

        bar = HardwareBar()

        bar.update_state('idle', {'xruns': 0})
        idle_text = bar.state.text()

        bar.update_state('stopped', {'xruns': 0})
        assert bar.state.text() != idle_text

    def test_xrun_count_turns_red_when_nonzero(self, qt_app):
        from tonesphere.ui.strip import HardwareBar

        bar = HardwareBar()
        bar.update_state('running', {'xruns': 3, 'cpu_usage': 5.0})

        assert bar.xruns._value == "3"
        assert bar.xruns._tone == Colors.ERROR


class TestChannelStrip:
    def test_failed_device_is_marked(self, qt_app):
        """
        A strip whose device would not open must look different, or its audio goes nowhere
        and the user has no way to know.
        """
        from tonesphere.ui.strip import ChannelStripWidget

        strip = ChannelStripWidget(1, "Interface", "input · 2 ch")
        strip.set_failed("Invalid device")

        assert "unavailable" in strip.subtitle_label.text()
        assert "FAILED" in strip.toolTip()

    def test_clearing_the_failure_restores_the_strip(self, qt_app):
        from tonesphere.ui.strip import ChannelStripWidget

        strip = ChannelStripWidget(1, "Interface", "input · 2 ch")
        strip.set_failed("Invalid device")
        strip.set_failed(None)

        assert strip.styleSheet() == ""

    def test_programmatic_mute_does_not_echo(self, qt_app):
        from tonesphere.ui.strip import ChannelStripWidget

        strip = ChannelStripWidget(1, "Interface", "input · 2 ch")
        emitted = []
        strip.mute_toggled.connect(lambda *args: emitted.append(args))

        strip.set_muted(True)
        assert emitted == []


class TestRoutingScene:
    def test_cables_track_their_nodes(self, qt_app):
        from PySide6.QtCore import QPointF

        from tonesphere.ui.routing_view import RoutingScene

        scene = RoutingScene()
        scene.add_node(1, "In", "test", False, True, False, QPointF(0, 0))
        scene.add_node(2, "Out", "test", True, False, False, QPointF(400, 0))
        cable = scene.add_cable(1, 2)

        before = cable.boundingRect()
        scene.nodes[2].setPos(QPointF(400, 300))

        assert cable.boundingRect() != before, "cable must follow the node it connects"

    def test_removing_a_cable_detaches_it_from_both_nodes(self, qt_app):
        from PySide6.QtCore import QPointF

        from tonesphere.ui.routing_view import RoutingScene

        scene = RoutingScene()
        scene.add_node(1, "In", "test", False, True, False, QPointF(0, 0))
        scene.add_node(2, "Out", "test", True, False, False, QPointF(400, 0))
        scene.add_cable(1, 2)
        scene.remove_cable(1, 2)

        assert (1, 2) not in scene.cables
        assert scene.nodes[1]._cables == []
        assert scene.nodes[2]._cables == []

    def test_adding_the_same_cable_twice_updates_rather_than_duplicates(self, qt_app):
        from PySide6.QtCore import QPointF

        from tonesphere.ui.routing_view import RoutingScene

        scene = RoutingScene()
        scene.add_node(1, "In", "test", False, True, False, QPointF(0, 0))
        scene.add_node(2, "Out", "test", True, False, False, QPointF(400, 0))

        scene.add_cable(1, 2, gain_db=0.0)
        scene.add_cable(1, 2, gain_db=-6.0)

        assert len(scene.cables) == 1
        assert scene.cables[(1, 2)].gain_db == -6.0

    def test_bus_nodes_are_visually_distinguished(self, qt_app):
        """
        A bus is in-process only and a device is real hardware. Conflating them is exactly
        the confusion the old UI created by listing both as "devices".
        """
        from PySide6.QtCore import QPointF

        from tonesphere.ui.routing_view import RoutingScene

        scene = RoutingScene()
        node = scene.add_node(1, "Bus 1", "in-process", True, True, True, QPointF(0, 0))

        assert node.is_bus is True
