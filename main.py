#!/usr/bin/env python3
"""
ToneSphere - Professional Audio Routing Engine
Main entry point for the application
"""

import sys
import time
import logging
from tonesphere.utils.config import ConfigManager
from tonesphere.utils.logger import logger_manager, enable_file_logging


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
        active = driver_info.get('active_driver')
        available = engine.get_available_drivers()

        report("INFO", f"Active driver:    {active or 'none'}")
        report("INFO", f"Available:        {', '.join(available) or 'none'}")

        devices = engine.get_devices()
        physical = [d for d in devices if d['type'].startswith('physical')]
        virtual = [d for d in devices if d['type'].startswith('virtual')]

        if physical:
            report("PASS", f"Physical devices: {len(physical)}")
        else:
            report("FAIL", "Physical devices: 0 — the active driver enumerated no "
                           "hardware, so nothing can be routed to or from your interface")

        report("INFO", f"Virtual buses:    {len(virtual)} (in-process only, not visible to other apps)")

        # Prove whether audio actually moves, rather than trusting the return value.
        if len(virtual) >= 2:
            import numpy as np
            source, dest = virtual[0], virtual[1]
            ok, message = engine.create_routing(source['id'], dest['id'], 1.0)
            report("INFO", f"Route {source['id']} -> {dest['id']}: {message}")

            src_device = engine.engine.virtual_manager.get_device(source['id'])
            dst_device = engine.engine.virtual_manager.get_device(dest['id'])

            if src_device and dst_device:
                src_device.write_audio(np.full((src_device.buffer_size, src_device.channels),
                                               0.5, dtype=np.float32))
                deadline = time.time() + 2.0
                peak = 0.0
                while time.time() < deadline:
                    peak = float(np.max(np.abs(dst_device.current_frame)))
                    if peak > 0.0:
                        break
                    time.sleep(0.01)

                if peak > 0.0:
                    report("PASS", f"Virtual bus carried audio (peak {peak:.3f})")
                else:
                    report("FAIL", "Virtual bus carried no audio (peak 0.0)")

        stats = engine.get_performance_stats()
        report("INFO", f"Nominal latency:  {format_measurement(stats.get('nominal_latency_ms'), ' ms')}")
        report("INFO", f"Measured latency: {format_measurement(stats.get('measured_latency_ms'), ' ms')}")
        report("INFO", f"CPU load:         {format_measurement(stats.get('cpu_usage'), '%')}")

        if not stats.get('audio_path_active', False):
            report("WARN", "No audio path to hardware — see the Roadmap in README.md")

    finally:
        engine.stop_engine()

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
    
    print("ToneSphere - Professional Audio Routing Engine")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        from tonesphere.core.engine_factory import UnifiedAudioEngine
        command = sys.argv[1].lower()
        
        if command == "server":
            from tonesphere.api.server import run_api_server
            print("Starting ToneSphere API server...")
            run_api_server()

        elif command == "gui":
            from tonesphere.gui.studio import ToneSphereStudioGUI
            from tonesphere.gui.system_tray import SystemTrayIcon
            print("Starting ToneSphere Studio GUI...")
            
            gui = ToneSphereStudioGUI()
            
            # Setup system tray with engine access
            def show_gui():
                gui.root.deiconify()
                gui.root.lift()
            
            def exit_app():
                # Stop engine if running
                if gui.engine and gui.is_running:
                    gui.engine.stop_engine()
                gui.root.quit()
            
            # Create a simple wrapper to access the GUI's engine dynamically
            class EngineAccessor:
                @property
                def is_running(self):
                    return gui.engine.is_running if gui.engine else False
                
                def start_engine(self):
                    if not gui.engine:
                        gui.toggle_engine()
                    elif not gui.engine.is_running:
                        gui.engine.start_engine()
                
                def stop_engine(self):
                    if gui.engine and gui.engine.is_running:
                        gui.engine.stop_engine()
                
                def refresh_devices(self):
                    if gui.engine:
                        gui.refresh_devices()
            
            # Pass the engine accessor to the system tray
            tray = SystemTrayIcon(show_gui, exit_app, engine=EngineAccessor())
            tray.start()
            
            # Handle window close to minimize to tray
            def on_closing():
                gui.root.withdraw()
            
            gui.root.protocol("WM_DELETE_WINDOW", on_closing)
            
            gui.run()
            
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