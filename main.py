#!/usr/bin/env python3
"""
ToneSphere - Professional Audio Routing Engine
Main entry point for the application
"""

import sys
import logging
from tonesphere.utils.config import ConfigManager
from tonesphere.utils.logger import logger_manager, enable_file_logging


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


def main():
    """Main entry point"""
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
            print("Running basic engine tests...")
            try:
                engine = UnifiedAudioEngine()
                engine.initialize()
                engine.start_engine()
                
                # Display driver info
                driver_info = engine.get_driver_info()
                print(f"✓ Using driver: {driver_info.get('active_driver', 'unknown')}")
                print(f"✓ Available drivers: {', '.join(engine.get_available_drivers())}")
                
                devices = engine.get_devices()
                print(f"✓ Found {len(devices)} audio devices")
                
                # Test virtual device creation
                virtual_id = engine.create_virtual_input("Test Input")
                print(f"✓ Created virtual input with ID: {virtual_id}")
                
                # Test routing
                devices = engine.get_devices()
                inputs = [d for d in devices if 'input' in d['type']]
                outputs = [d for d in devices if 'output' in d['type']]
                
                if len(inputs) > 0 and len(outputs) > 0:
                    success, message = engine.create_routing(inputs[0]['id'], outputs[0]['id'])
                    if success:
                        print("✓ Created test routing")
                    else:
                        print("✗ Failed to create routing")
                
                # Display performance stats
                stats = engine.get_performance_stats()
                print(f"✓ Latency: {stats.get('latency_ms', 0):.2f}ms")
                print(f"✓ CPU Usage: {stats.get('cpu_usage', 0):.1f}%")
                
                engine.stop_engine()
                print("✓ All tests passed")
                
            except Exception as e:
                print(f"✗ Test failed: {e}")
                import traceback
                traceback.print_exc()
            
        else:
            print(f"Unknown command: {command}")
            print("Available commands: server, gui, cli, test")
    else:
        print("ToneSphere - Professional Audio Routing System")
        print("Version 1.0.0")
        print()
        print("Usage:")
        print("  python main.py server  - Start API server")
        print("  python main.py gui     - Start GUI application")
        print("  python main.py cli     - Interactive CLI mode")
        print("  python main.py test    - Run basic tests")
        print()
        print("Features:")
        print("  ✓ Native audio driver support (ASIO, WASAPI, ALSA, PulseAudio, JACK, etc.)")
        print("  ✓ Virtual audio device creation")
        print("  ✓ Advanced routing matrix")
        print("  ✓ Real-time audio effects")
        print("  ✓ RESTful API with WebSocket events")
        print("  ✓ VoiceMeeter Potato-like functionality")
        print("  ✓ Cross-platform support (Windows, Linux, macOS)")
        print("  ✓ Low-latency professional audio processing")


if __name__ == "__main__":
    main()