#!/usr/bin/env python3
"""
ToneSphere - Professional Audio Routing Engine
Main entry point for the application
"""

import logging
import sys
import time

from tonesphere.utils.config import ConfigManager
from tonesphere.utils.logger import logger_manager


def setup_console_encoding():
    """
    Force UTF-8 on stdout/stderr.

    Windows consoles default to cp1252, which cannot encode the symbols used in
    our output and raises UnicodeEncodeError mid-print. errors='replace' keeps a
    legacy console readable instead of crashing the process.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, ValueError):
            pass


def setup_logging(config: dict):
    """Setup logging based on configuration"""
    log_config = config.get('logging', {})

    # Get log level
    level_str = log_config.get('level', 'INFO')
    level = getattr(logging, level_str.upper(), logging.INFO)

    # Setup logger with configuration
    logger_manager.setup_logger(
        name="tonesphere",
        level=level,
        enable_file_logging=log_config.get('enable_file_logging', False),
        log_file=log_config.get('log_file', 'tonesphere.log'),
        log_dir=log_config.get('log_dir', 'logs'),
        max_bytes=log_config.get('max_file_size_mb', 10) * 1024 * 1024,
        backup_count=log_config.get('backup_count', 5),
        structured=log_config.get('structured', False),
        console_colors=log_config.get('colored_console', True)
    )


def run_diagnostics() -> int:
    """
    Report what the engine can and cannot do on this machine.

    This replaces the old `test` command, which printed a checkmark next to every
    line it reached — including "Found 0 audio devices" — and then claimed "All
    tests passed" regardless of outcome. Findings are reported as PASS/WARN/FAIL and
    the exit code reflects them, so this is usable from a script.
    """
    from tonesphere.core.engine_factory import UnifiedAudioEngine
    from tonesphere.utils.formatting import format_measurement

    failures = 0
    warnings = 0

    def report(status: str, message: str):
        nonlocal failures, warnings
        if status == "FAIL":
            failures += 1
        elif status == "WARN":
            warnings += 1
        print(f"[{status:4}] {message}")

    print("ToneSphere diagnostics")
    print("-" * 60)

    try:
        engine = UnifiedAudioEngine()
        engine.initialize()
        engine.start_engine()
    except Exception as e:
        report("FAIL", f"Engine failed to start: {e}")
        import traceback
        traceback.print_exc()
        return 1

    try:
        driver_info = engine.get_driver_info()
        report("INFO", f"PortAudio:        {driver_info.get('portaudio_version', 'unknown')}")
        report("INFO", f"Active backend:   {driver_info.get('active_driver') or 'none'}")
        report("INFO", f"Available:        {', '.join(engine.get_available_drivers()) or 'none'}")

        if driver_info.get('error'):
            report("FAIL", f"Audio backend:    {driver_info['error']}")

        if not driver_info.get('asio_available'):
            report("INFO", "ASIO:             not in this PortAudio build (SDK is not "
                           "redistributable); WASAPI exclusive is the low-latency path")

        devices = engine.get_devices()
        inputs = [d for d in devices if d['direction'] == 'input' and 'bus' not in d['host_api'].lower()]
        outputs = [d for d in devices if d['direction'] == 'output' and 'bus' not in d['host_api'].lower()]

        if inputs:
            report("PASS", f"Hardware inputs:  {len(inputs)}")
        else:
            report("WARN", "Hardware inputs:  0")

        if outputs:
            report("PASS", f"Hardware outputs: {len(outputs)}")
        else:
            report("FAIL", "Hardware outputs: 0 — nothing can be played")

        # Prove audio reaches real hardware, by measuring rather than trusting a
        # return value. This is the check the project never had.
        if outputs:
            import numpy as np

            target_id = engine.engine.default_output_id()
            target = next((d for d in outputs if d['id'] == target_id), outputs[0])
            bus_id = engine.create_virtual_input("Diagnostic tone", channels=2)

            ok, message = engine.create_routing(bus_id, target['id'], 0.25)
            if not ok:
                report("FAIL", f"Route to {target['name']}: {message}")
            else:
                engine.start_engine()
                stats = engine.get_performance_stats()

                if not stats['audio_path_active']:
                    report("FAIL", f"Stream would not start: {'; '.join(stats['problems'])}")
                else:
                    report("PASS", f"Stream open on {target['name'][:34]}")

                    # 440 Hz for two seconds at -12 dBFS; audible but not startling.
                    block = engine.buffer_size
                    phase = 0
                    deadline = time.monotonic() + 2.0
                    while time.monotonic() < deadline:
                        t = (np.arange(block) + phase) / engine.sample_rate
                        wave = (0.25 * np.sin(2 * 3.14159265 * 440.0 * t)).astype(np.float32)
                        engine.write_to_bus(bus_id, np.repeat(wave.reshape(-1, 1), 2, axis=1))
                        phase += block
                        time.sleep(block / engine.sample_rate / 3)

                    stats = engine.get_performance_stats()

                    if stats['callback_count'] > 50:
                        report("PASS", f"Audio callback ran {stats['callback_count']} times")
                    else:
                        report("FAIL", f"Callback barely ran ({stats['callback_count']})")

                    if stats['callback_errors'] == 0:
                        report("PASS", "No callback errors")
                    else:
                        report("FAIL", f"{stats['callback_errors']} callback error(s): "
                                       f"{engine.engine.host.last_callback_error}")

                    if stats['xruns'] == 0:
                        report("PASS", "No dropouts (xruns)")
                    else:
                        report("WARN", f"{stats['xruns']} xrun(s) — try a larger buffer")

                    report("INFO", f"Measured latency: "
                                   f"{format_measurement(stats.get('measured_latency_ms'), ' ms')} round trip")
                    report("INFO", f"Nominal latency:  "
                                   f"{format_measurement(stats.get('nominal_latency_ms'), ' ms')} "
                                   f"(buffer arithmetic only)")
                    report("INFO", f"DSP load:         {format_measurement(stats.get('cpu_usage'), '%')}")
                    report("INFO", f"Buffer / rate:    {engine.buffer_size} frames @ "
                                   f"{engine.sample_rate} Hz, "
                                   f"exclusive={driver_info.get('exclusive_mode')}")

    finally:
        engine.stop_engine()
        engine.cleanup()

    print("-" * 60)
    print(f"{failures} failure(s), {warnings} warning(s)")
    return 1 if failures else 0


def main():
    """Main entry point"""
    setup_console_encoding()

    config_manager = ConfigManager()
    config = config_manager.load_config()

    # Setup logging first
    setup_logging(config)

    from tonesphere import __version__
    print(f"ToneSphere {__version__}")
    print("=" * 50)

    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "server":
            from tonesphere.api.server import run_api_server
            print("Starting ToneSphere API server...")
            run_api_server()

        elif command == "gui":
            from tonesphere.ui import run

            print("Starting ToneSphere...")
            return run(config_manager)

        elif command == "cli":
            from tonesphere.cli.interface import AudioEngineCLI
            print("Starting ToneSphere CLI...")
            cli = AudioEngineCLI()
            cli.run_interactive_mode()

        elif command == "test":
            return run_diagnostics()

        else:
            print(f"Unknown command: {command}")
            print("Available commands: server, gui, cli, test")
            return 2
    else:
        from tonesphere import __version__
        print(f"ToneSphere - Audio Routing Engine (v{__version__}, pre-alpha)")
        print()
        print("Usage:")
        print("  python main.py server  - Start API server")
        print("  python main.py gui     - Start GUI application")
        print("  python main.py cli     - Interactive CLI mode")
        print("  python main.py test    - Run basic tests")
        print()
        print("STATUS: ToneSphere does not process audio yet.")
        print("The routing matrix, mixer state and UI are in place, but the audio")
        print("path is under construction. Nothing you route will be audible.")
        print("See the Roadmap in README.md for what works and what is next.")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
