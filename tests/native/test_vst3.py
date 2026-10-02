"""
The native VST3 host, proven with ToneSphere's own deterministic test plugin
(native/test_plugin/test_plugin.cpp), which multiplies by a known gain and delays by a
known 64 samples. Every audio assertion compares samples against that arithmetic.

This is a plugin ToneSphere built for the purpose, not a third-party plugin. Commercial
and third-party compatibility is in tests/hardware/test_vst3_third_party.py and claims
only what it ran.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from tests.signals import RATE, dominant_frequency, sine, white_noise
from tonesphere.native import VST3, Insert, NativeEngine, NativeError, Node, Route
from tonesphere.plugins import PluginError, PluginInstance, PluginState, classes_in
from tonesphere.plugins import scan as scanner

BLOCK = 256
SRC, OUT = 1, 2
LATENCY = 64
ROOT = Path(__file__).resolve().parents[2]


def built(pattern: str) -> Path:
    found = sorted(ROOT.glob(f"native/build/*/{pattern}"))
    if not found:
        pytest.fail(f"{pattern} is not built: run `uv run python scripts/build_native.py` (needs the VST3 SDK)")
    return found[0]


@pytest.fixture(scope="module")
def module_path():
    return built("VST3/Release/tonesphere_test_gain.vst3")


@pytest.fixture(scope="module")
def classes(module_path):
    return {c.name: c for c in classes_in(module_path)}


@pytest.fixture
def gain_plugin(classes):
    with PluginInstance(classes["ToneSphere Test Gain"], RATE, BLOCK, 2) as plugin:
        yield plugin


def run_through(plugin, signal, blocks=None, changes=None):
    """SRC -> [plugin insert] -> OUT, block by block; `changes` maps block index -> (param, value)."""
    with NativeEngine(RATE, BLOCK) as engine:
        engine.apply_plan([Node.source(SRC, signal.shape[1]), Node.sink(OUT, signal.shape[1])], [Route(SRC, OUT)],
                          [Insert(SRC, 0, VST3, plugin=plugin.handle)])
        engine.process({SRC: np.zeros((BLOCK, signal.shape[1]), np.float32)}, {OUT: signal.shape[1]})
        out = []
        for b in range(blocks or len(signal) // BLOCK):
            if changes and b in changes:
                plugin.set_parameter(*changes[b])
            out.append(engine.process({SRC: signal[b * BLOCK:(b + 1) * BLOCK]}, {OUT: signal.shape[1]})[OUT])
        stats = engine.stats()
    return np.concatenate(out), stats


class TestDiscovery:
    def test_the_module_lists_its_classes_by_id_and_kind(self, classes):
        assert {"ToneSphere Test Gain", "ToneSphere Test Crash", "ToneSphere Test Crash In Process"} <= set(classes)
        gain = classes["ToneSphere Test Gain"]
        assert gain.is_audio_effect and gain.vendor == "Neural Nexus Studios"
        assert gain.uid == "5E1B7C019A2F4D3E8B6C1D4F2A3B4C01"
        assert gain.subcategories == "Fx|Tools" and gain.sdk_version.startswith("VST 3.8")
        assert not classes["ToneSphere Test Gain Controller"].is_audio_effect

    def test_scanning_runs_out_of_process_and_caches(self, module_path, tmp_path):
        cache = scanner.ScanCache(tmp_path / "cache.json")
        first = scanner.scan([module_path.parent], cache)
        assert [r.status for r in first] == [scanner.OK]
        assert first[0].architecture == 'x64'
        again = scanner.ScanCache(tmp_path / "cache.json").get(module_path)
        assert again is not None and [c.name for c in again.effects] == [c.name for c in first[0].effects]

    def test_a_module_that_crashes_on_load_is_contained(self, tmp_path):
        """The crash module faults inside GetPluginFactory; the scanner must survive it."""
        crash_module = built("test_plugins/ToneSphereTestCrashModule.vst3")
        result = scanner.scan_module(crash_module)
        assert result.status == scanner.CRASHED
        assert "access violation" in result.detail

    def test_a_scanner_that_dies_outright_is_reported_not_trusted(self, module_path, monkeypatch):
        """A crash no handler catches (os.abort in the scanner stands in for one)."""
        monkeypatch.setenv('TONESPHERE_SCAN_TEST_ABORT', '1')
        result = scanner.scan_module(module_path)
        assert result.status == scanner.CRASHED and "died" in result.detail

    def test_wrong_architecture_is_rejected_without_loading(self, tmp_path):
        bundle = tmp_path / "Old32.vst3" / "Contents" / "x86_64-win"
        bundle.mkdir(parents=True)
        header = bytearray(512)
        header[:2] = b'MZ'
        header[0x3C:0x40] = (0x80).to_bytes(4, 'little')
        header[0x80:0x84] = b'PE\0\0'
        header[0x84:0x86] = (0x14C).to_bytes(2, 'little')  # IMAGE_FILE_MACHINE_I386
        (bundle / "Old32.vst3").write_bytes(bytes(header))
        result = scanner.scan_module(tmp_path / "Old32.vst3")
        assert result.status == scanner.WRONG_ARCHITECTURE and result.architecture == 'x86'

    def test_a_file_that_is_not_a_plugin_fails_with_a_reason(self, tmp_path):
        fake = tmp_path / "Fake.vst3"
        fake.write_bytes(b"not a dll")
        result = scanner.scan_module(fake)
        assert result.status == scanner.WRONG_ARCHITECTURE and "not a DLL" in result.detail

    def test_bundles_are_found_recursively_and_not_searched_inside(self, module_path, tmp_path):
        nested = tmp_path / "Vendor" / "Series"
        nested.mkdir(parents=True)
        (nested / "Loose.vst3").write_bytes(b"x")
        found = scanner.find_modules([tmp_path, module_path.parent])
        assert module_path in found and nested / "Loose.vst3" in found
        assert not any(str(p).startswith(str(module_path) + os.sep) for p in found)


class TestProcessing:
    def test_audio_is_delayed_by_the_reported_latency_and_scaled_by_the_gain(self, gain_plugin):
        """Unity gain (0.5 normalised) and 64 samples of delay: the output is the input, shifted, exactly."""
        assert gain_plugin.latency_samples == LATENCY
        signal = white_noise(BLOCK * 16, amplitude=0.3)
        out, stats = run_through(gain_plugin, signal)
        assert np.array_equal(out[LATENCY:], signal[:-LATENCY])
        assert np.all(out[:LATENCY] == 0)
        assert stats['rt_allocations'] == 0

    def test_measured_latency_equals_reported_latency(self, gain_plugin):
        signal = white_noise(BLOCK * 16, amplitude=0.3, seed=3)
        out, _ = run_through(gain_plugin, signal)
        corr = np.correlate(out[:, 0], signal[:, 0], mode='full')[len(signal) - 1:]
        assert int(np.argmax(corr[:BLOCK])) == gain_plugin.latency_samples

    def test_a_parameter_change_takes_effect_on_the_next_block(self, gain_plugin):
        tone = sine(BLOCK * 12, 1000.0, amplitude=0.4)
        out, _ = run_through(gain_plugin, tone, changes={6: (0, 0.25)})  # x1.0 -> x0.5
        before = out[4 * BLOCK:6 * BLOCK]
        after = out[7 * BLOCK:12 * BLOCK]
        assert np.max(np.abs(before)) == pytest.approx(0.4, rel=0.01)
        assert np.max(np.abs(after)) == pytest.approx(0.2, rel=0.01)
        assert dominant_frequency(after) == pytest.approx(1000.0, abs=100.0)

    def test_the_controller_reports_the_value_it_was_given(self, gain_plugin):
        gain_plugin.set_parameter(0, 0.8)
        p = gain_plugin.parameters()[0]
        assert p.title == "Gain" and p.normalized == pytest.approx(0.8)
        assert p.plain == pytest.approx(1.6) and p.automatable
        with pytest.raises(PluginError):
            gain_plugin.set_parameter(0, 1.5)

    def test_mono_and_multichannel_instances(self, classes):
        for channels in (1, 4):
            with PluginInstance(classes["ToneSphere Test Gain"], RATE, BLOCK, channels) as plugin:
                signal = white_noise(BLOCK * 6, amplitude=0.2, channels=channels)
                out, _ = run_through(plugin, signal)
                assert np.array_equal(out[LATENCY:], signal[:-LATENCY])

    def test_width_must_match_the_node(self, gain_plugin):
        with NativeEngine(RATE, BLOCK) as engine, pytest.raises(NativeError) as refused:
            engine.apply_plan([Node.source(SRC, 1), Node.sink(OUT, 1)], [Route(SRC, OUT)],
                              [Insert(SRC, 0, VST3, plugin=gain_plugin.handle)])
        assert "opened for 2 channels" in str(refused.value)

    def test_one_instance_cannot_sit_on_two_nodes(self, gain_plugin):
        with NativeEngine(RATE, BLOCK) as engine, pytest.raises(NativeError) as refused:
            engine.apply_plan([Node.source(SRC, 2), Node.bus(3, 2), Node.sink(OUT, 2)],
                              [Route(SRC, 3), Route(3, OUT)],
                              [Insert(SRC, 0, VST3, plugin=gain_plugin.handle),
                               Insert(3, 0, VST3, plugin=gain_plugin.handle)])
        assert "already inserted" in str(refused.value)


class TestState:
    def test_state_round_trips_into_a_new_instance(self, classes, gain_plugin):
        gain_plugin.set_parameter(0, 0.8)
        run_through(gain_plugin, sine(BLOCK * 2), blocks=2)  # the processor applies the change
        saved = gain_plugin.state()
        restored_dict = PluginState.from_dict(saved.to_dict())
        with PluginInstance(classes["ToneSphere Test Gain"], RATE, BLOCK, 2) as fresh:
            fresh.restore(restored_dict)
            assert fresh.parameters()[0].normalized == pytest.approx(0.8)
            signal = white_noise(BLOCK * 8, amplitude=0.2)
            out, _ = run_through(fresh, signal)
            assert np.allclose(out[LATENCY:], signal[:-LATENCY] * 1.6, atol=1e-6)

    def test_a_change_made_while_nothing_processes_is_in_the_saved_state(self, classes, gain_plugin):
        """A plugin hears a change only through process(): with the engine stopped, saving
        the session once kept the old value (Guitar Rig 7's master volume, on this host)."""
        gain_plugin.set_parameter(0, 0.25)
        saved = gain_plugin.state()
        with PluginInstance(classes["ToneSphere Test Gain"], RATE, BLOCK, 2) as fresh:
            fresh.restore(saved)
            signal = white_noise(BLOCK * 8, amplitude=0.2)
            out, _ = run_through(fresh, signal)
            assert np.allclose(out[LATENCY:], signal[:-LATENCY] * 0.5, atol=1e-6)
        out, _ = run_through(gain_plugin, signal)
        assert np.allclose(out[LATENCY:], signal[:-LATENCY] * 0.5, atol=1e-6), "and it still plays at that gain"

    def test_foreign_state_is_rejected_and_the_plugin_keeps_working(self, gain_plugin):
        with pytest.raises(PluginError) as rejected:
            gain_plugin.restore(PluginState(b"somebody else's preset", b""))
        assert "rejected" in str(rejected.value)
        signal = white_noise(BLOCK * 6, amplitude=0.2)
        out, _ = run_through(gain_plugin, signal)
        assert np.array_equal(out[LATENCY:], signal[:-LATENCY])


class TestFailureIsolation:
    def test_a_plugin_that_crashes_on_initialise_is_reported_and_the_host_survives(self, classes, gain_plugin):
        with pytest.raises(PluginError) as crashed:
            PluginInstance(classes["ToneSphere Test Crash"], RATE, BLOCK, 2)
        assert "crashed during initialize" in str(crashed.value)
        assert "access violation" in str(crashed.value)
        signal = white_noise(BLOCK * 4, amplitude=0.2)
        out, _ = run_through(gain_plugin, signal)  # another plugin in the same host still works
        assert np.array_equal(out[LATENCY:], signal[:-LATENCY])

    def test_a_crash_on_the_audio_thread_bypasses_the_plugin_and_the_audio_continues(self, classes):
        """
        The class faults on its tenth process() call. From then on it is bypassed — the
        dry signal passes — and never called again; the engine keeps producing blocks.
        """
        with PluginInstance(classes["ToneSphere Test Crash In Process"], RATE, BLOCK, 2) as plugin:
            signal = white_noise(BLOCK * 20, amplitude=0.2)
            out, stats = run_through(plugin, signal)
            status = plugin.status()
        assert status.crashed and "process" in status.fault and "access violation" in status.fault
        # run_through's silent warm-up block is call 1, so call 10 is signal block 8: nine
        # calls completed, the tenth faulted, and none followed.
        assert status.blocks == 9, "processed until the faulting call, then never again"
        assert stats['blocks'] == 21
        tail = out[12 * BLOCK:]
        assert np.array_equal(tail, signal[12 * BLOCK:]), "after the fault, the input passes through unchanged"

    def test_a_crashed_plugin_cannot_be_inserted_again(self, classes):
        with PluginInstance(classes["ToneSphere Test Crash In Process"], RATE, BLOCK, 2) as plugin:
            run_through(plugin, white_noise(BLOCK * 12, amplitude=0.1))
            with NativeEngine(RATE, BLOCK) as engine, pytest.raises(NativeError) as refused:
                engine.apply_plan([Node.source(SRC, 2), Node.sink(OUT, 2)], [Route(SRC, OUT)],
                                  [Insert(SRC, 1, VST3, plugin=plugin.handle)])
            assert "crashed" in str(refused.value)


def test_the_test_plugin_offers_no_editor(gain_plugin):
    assert gain_plugin.has_editor() is False
    with pytest.raises(PluginError):
        gain_plugin.open_editor()
