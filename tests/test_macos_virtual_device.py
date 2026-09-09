"""
Track 3 — the macOS CoreAudio HAL plug-in, Python side only.

Read this before adding to it: **nothing in this file proves the plug-in works.** It
cannot. The C in `native/coreaudio-plugin/` was written on a Windows machine and has never
been compiled; the only thing that can prove it moves audio is the `build-macos-plugin`
job in `.github/workflows/ci.yml`, which builds it, installs it on a real macOS runner,
restarts coreaudiod and round-trips a 1 kHz sine through the device
(`.github/scripts/macos_plugin_roundtrip.py`). There is deliberately no test here — not
even a `hardware`-marked one — that claims otherwise, because a test that skips on every
machine anyone will run it on is a claim disguised as a check.

What *is* provable everywhere, and is what this file holds itself to:

- the platform guard is honest: off macOS the backend refuses with a real reason and the
  engine returns None rather than a placeholder device id;
- the registry bookkeeping and the `origin` label are right, which is pure dict logic;
- the device name and UID in the C source and in `macos_virtual.py` are the same strings,
  which is the one way the C and the Python are actually coupled;
- the REST surface is reachable and refuses honestly.

Same two-tier shape as `tests/test_linux_virtual_device.py` and
`tests/test_process_capture.py`, minus the tier that needs the OS — because on this track
that tier lives in CI.
"""

import platform
import re
from pathlib import Path

import pytest

PLUGIN_SOURCE = Path(__file__).resolve().parents[1] / "native" / "coreaudio-plugin"


def _run(coroutine):
    """Call a REST endpoint function directly, the same shortcut the sibling tests use."""
    import asyncio

    return asyncio.run(coroutine)


class TestConstantsMatchTheCPlugin:
    """
    `MACOS_DEVICE_NAME`/`MACOS_DEVICE_UID` are duplicated in C, and a duplicate that can
    drift is a bug waiting to happen: if the plug-in publishes one name and Python looks
    for another, `create_macos_system_device` returns None forever and the reason is
    invisible. This reads the actual C file, so the two cannot disagree silently.
    """

    def _defines(self) -> dict[str, str]:
        source = (PLUGIN_SOURCE / "ToneSphereAudio.c").read_text(encoding="utf-8")
        return dict(re.findall(r'^#define\s+(k\w+)\s+"([^"]*)"', source, re.MULTILINE))

    def test_device_name_matches(self):
        from tonesphere.engine.macos_virtual import MACOS_DEVICE_NAME

        assert self._defines()["kDevice_Name"] == MACOS_DEVICE_NAME

    def test_device_uid_matches(self):
        from tonesphere.engine.macos_virtual import MACOS_DEVICE_UID

        assert self._defines()["kDevice_UID"] == MACOS_DEVICE_UID

    def test_bundle_identifier_matches_the_plist(self):
        from tonesphere.engine.macos_virtual import MACOS_BUNDLE_ID

        plist = (PLUGIN_SOURCE / "Info.plist").read_text(encoding="utf-8")
        assert f"<string>{MACOS_BUNDLE_ID}</string>" in plist

    def test_plist_declares_the_real_audioserverplugin_type_uuid(self):
        """
        443ABAB8-E7B3-491A-B985-BEB9187030DB is `kAudioServerPlugInTypeUUID` from Apple's
        CoreAudio/AudioServerPlugIn.h written out as a string. Get it wrong and coreaudiod
        never asks the bundle for anything, with no error logged anywhere — so it is worth
        pinning here even though only a Mac can prove the consequence.
        """
        plist = (PLUGIN_SOURCE / "Info.plist").read_text(encoding="utf-8")
        assert "443ABAB8-E7B3-491A-B985-BEB9187030DB" in plist

    def test_plist_factory_function_is_the_exported_symbol(self):
        plist = (PLUGIN_SOURCE / "Info.plist").read_text(encoding="utf-8")
        source = (PLUGIN_SOURCE / "ToneSphereAudio.c").read_text(encoding="utf-8")

        assert "<string>ToneSphereAudio_Create</string>" in plist
        assert "void* ToneSphereAudio_Create(" in source

    def test_the_c_source_does_not_claim_to_be_verified(self):
        """
        The one honesty guarantee this track can enforce in code: the plug-in source has
        to keep saying, in its own header comment, that it was never compiled or heard by
        its author. If someone deletes that, they have to delete this test too and think
        about why.
        """
        source = (PLUGIN_SOURCE / "ToneSphereAudio.c").read_text(encoding="utf-8")
        header = source[: source.index("#include")]

        assert "never been compiled" in header
        assert "NOT verified" in header


class TestPlatformGuardIsHonest:
    """
    Same "platform-only assertion, testable everywhere" pattern
    `tests/test_honesty.py::TestCaptureStatusMatchesWhatIsImplemented` holds Track 1's
    `process_loopback_supported()` to, and Track 2's `virtual_sink_supported()`.
    """

    def test_supported_flag_matches_the_real_platform(self):
        from tonesphere.engine.macos_virtual import hal_plugin_supported

        assert hal_plugin_supported() == (platform.system() == "Darwin")

    def test_unavailable_reason_names_this_platform_off_macos(self):
        from tonesphere.engine.macos_virtual import hal_plugin_supported, unavailable_reason

        if hal_plugin_supported():
            pytest.skip("this machine is macOS; the reason depends on whether it is installed")

        reason = unavailable_reason()
        assert reason is not None
        assert platform.system() in reason

    def test_installed_is_false_rather_than_an_error_off_macos(self):
        from tonesphere.engine.macos_virtual import hal_plugin_installed, hal_plugin_supported

        if hal_plugin_supported():
            pytest.skip("this machine is macOS; installation state is a real answer here")

        assert hal_plugin_installed() is False

    def test_status_reports_unknown_visibility_rather_than_false_off_macos(self):
        """
        `device_visible` must be None off macOS, not False: nothing looked, and this
        project reports a measurement it did not take as unknown rather than as a
        confident negative.
        """
        from tonesphere.engine.macos_virtual import hal_plugin_supported, plugin_status

        status = plugin_status()

        assert status["platform"] == platform.system()
        assert status["supported"] == hal_plugin_supported()

        if not hal_plugin_supported():
            assert status["device_visible"] is None
            assert status["installed"] is False

    def test_module_imports_with_no_macos_only_top_level_import(self):
        import importlib

        module = importlib.import_module("tonesphere.engine.macos_virtual")

        assert hasattr(module, "hal_plugin_supported")
        assert hasattr(module, "plugin_status")
        assert hasattr(module, "unavailable_reason")


class TestEngineRefusesHonestlyOffMacOS:
    """
    `AudioEngine.create_macos_system_device`/`remove_macos_system_device`, run for real on
    whatever platform this is — not mocked, so on Windows the actual refusal path executes
    end to end.
    """

    def test_create_returns_none_rather_than_a_placeholder_id(self):
        from tonesphere.core.engine import AudioEngine
        from tonesphere.engine.macos_virtual import hal_plugin_installed

        if hal_plugin_installed():
            pytest.skip("the HAL plug-in is installed on this machine; see the CI job instead")

        engine = AudioEngine()
        engine.initialize()

        try:
            assert engine.create_macos_system_device() is None
            assert engine._system_virtual_devices == {}
        finally:
            engine.cleanup()

    def test_remove_reports_an_unknown_id_rather_than_success(self):
        from tonesphere.core.engine import AudioEngine

        engine = AudioEngine()
        engine.initialize()

        try:
            assert engine.remove_macos_system_device(999_999) is False
        finally:
            engine.cleanup()

    def test_remove_refuses_an_id_belonging_to_the_other_backend(self):
        """
        One registry, two backends, two remove endpoints that both take a device id — so
        an id from the Linux backend really can arrive at the macOS remover and vice
        versa. Each must decline rather than tear down something it does not own.
        """
        from tonesphere.core.engine import AudioEngine
        from tonesphere.engine.macos_virtual import MACOS_BACKEND

        engine = AudioEngine()
        engine.initialize()

        try:
            engine._system_virtual_devices[500] = {
                'name': 'a linux sink', 'handle': None, 'backend': 'linux-null-sink',
            }
            engine._system_virtual_devices[501] = {
                'name': 'the hal device', 'handle': None, 'backend': MACOS_BACKEND,
            }

            assert engine.remove_macos_system_device(500) is False
            assert engine.remove_linux_system_sink(501) is False

            assert 500 in engine._system_virtual_devices
            assert 501 in engine._system_virtual_devices
        finally:
            engine._system_virtual_devices.clear()
            engine.cleanup()

    def test_cleanup_releases_every_tracked_macos_registration(self):
        """
        The macOS mirror of Track 2's `test_cleanup_tears_down_every_tracked_sink`: the
        engine leaves nothing of its own behind. Here that is a registration rather than
        an OS resource, since the plug-in itself is installed by a human and stays.
        """
        from tonesphere.core.engine import AudioEngine
        from tonesphere.engine.macos_virtual import MACOS_BACKEND

        engine = AudioEngine()
        engine.initialize()

        engine._system_virtual_devices[321] = {
            'name': 'the hal device', 'handle': None, 'backend': MACOS_BACKEND,
        }

        engine.cleanup()

        assert engine._system_virtual_devices == {}


class TestOriginLabelling:
    """
    The mirror image of `tests/test_honesty.py::TestBusesAreNotSystemDevices`, for this
    backend: a device tracked in `_system_virtual_devices` with the macOS backend is
    labelled `os_virtual_endpoint`, never `in_process_bus` and never bare `hardware`.
    Pure dict bookkeeping in `get_devices()`, so it is fully provable on this machine.
    """

    def test_a_macos_hal_device_is_labelled_os_virtual_endpoint(self):
        from tonesphere.core.engine import AudioEngine
        from tonesphere.engine.macos_virtual import MACOS_BACKEND

        engine = AudioEngine()
        engine.initialize()

        # `get_devices()` filters to the active backend, so relabel an id that survives it.
        visible_ids = [d["id"] for d in engine.get_devices()]
        if not visible_ids:
            pytest.skip("no enumerated devices on this machine to relabel")

        target_id = visible_ids[0]
        engine._system_virtual_devices[target_id] = {
            'name': 'the hal device', 'handle': None, 'backend': MACOS_BACKEND,
        }

        entries = [d for d in engine.get_devices() if d["id"] == target_id]

        assert entries
        assert all(d["origin"] == "os_virtual_endpoint" for d in entries)
        assert all(d["origin"] != "in_process_bus" for d in entries)

        engine._system_virtual_devices.clear()

    def test_an_unclaimed_hal_device_is_not_labelled_as_ours(self):
        """
        A HAL device nobody attached to — someone else's BlackHole, or ours before
        `create_macos_system_device` was called — is plain hardware. `origin` says who put
        the endpoint there, and until ToneSphere has actually claimed one, the answer is
        not ToneSphere.
        """
        from tonesphere.core.engine import AudioEngine

        engine = AudioEngine()
        engine.initialize()

        devices = engine.get_devices()
        if not devices:
            pytest.skip("no enumerated devices on this machine")

        assert all(d["origin"] == "hardware" for d in devices
                   if d["id"] not in engine._system_virtual_devices)


class TestExposureIsActuallyWired:
    """The REST surface, proved reachable rather than assumed."""

    def test_status_endpoint_returns_the_honest_status(self):
        import tonesphere.api.server as server
        from tonesphere.engine.macos_virtual import MACOS_DEVICE_NAME

        status = _run(server.get_macos_hal_status())

        assert status["platform"] == platform.system()
        assert status["device_name"] == MACOS_DEVICE_NAME
        assert status["backend"] == "macos-hal-plugin"

    def test_attach_refuses_off_macos_with_a_reason(self):
        from fastapi import HTTPException

        import tonesphere.api.server as server
        from tonesphere.core.engine_factory import UnifiedAudioEngine
        from tonesphere.engine.macos_virtual import hal_plugin_supported

        if hal_plugin_supported():
            pytest.skip(f"this machine ({platform.system()}) is macOS")

        engine = UnifiedAudioEngine()
        engine.initialize()
        server.audio_engine = engine

        try:
            with pytest.raises(HTTPException) as raised:
                _run(server.attach_macos_hal_device())
        finally:
            server.audio_engine = None
            engine.cleanup()

        assert raised.value.status_code == 400
        assert platform.system() in raised.value.detail

    def test_release_reports_an_unknown_device_rather_than_success(self):
        from fastapi import HTTPException

        import tonesphere.api.server as server
        from tonesphere.core.engine_factory import UnifiedAudioEngine

        engine = UnifiedAudioEngine()
        engine.initialize()
        server.audio_engine = engine

        try:
            with pytest.raises(HTTPException) as raised:
                _run(server.release_macos_hal_device(999_999))
        finally:
            server.audio_engine = None
            engine.cleanup()

        assert raised.value.status_code == 404


class TestTheProofLivesInCI:
    """
    The round-trip proof is a CI job, not a test — so the things that job depends on are
    checked here, where a rename or a deletion shows up immediately instead of as a red
    build nobody connected to this change.
    """

    def test_the_ci_job_exists_and_runs_the_round_trip(self):
        import yaml

        workflow = yaml.safe_load(
            (Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml")
            .read_text(encoding="utf-8")
        )

        job = workflow["jobs"]["build-macos-plugin"]
        assert job["runs-on"] == "macos-latest"

        steps = " ".join(str(step.get("run", "")) for step in job["steps"])
        assert "macos_plugin_roundtrip.py" in steps

    def test_the_round_trip_script_exists_and_uses_the_shared_constant(self):
        script = (
            Path(__file__).resolve().parents[1]
            / ".github" / "scripts" / "macos_plugin_roundtrip.py"
        ).read_text(encoding="utf-8")

        assert "MACOS_DEVICE_NAME" in script
        # It must measure the captured signal, not merely find a device name in a list.
        assert "dominant_frequency" in script
