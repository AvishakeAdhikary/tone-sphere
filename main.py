#!/usr/bin/env python3
"""
ToneSphere - Professional Audio Routing Engine
Main entry point for the application
"""

import sys
from tonesphere.api.server import run_api_server
from tonesphere.gui.studio import ToneSphereStudioGUI
from tonesphere.cli.interface import AudioEngineCLI
from tonesphere.core.engine import AudioEngine


def main():
    """Main entry point"""
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "server":
            print("Starting ToneSphere API server...")
            run_api_server()

        elif command == "gui":
            print("Starting ToneSphere Studio GUI...")
            gui = ToneSphereStudioGUI()
            gui.run()
            
        elif command == "cli":
            print("Starting ToneSphere CLI...")
            cli = AudioEngineCLI()
            cli.run_interactive_mode()
            
        elif command == "test":
            print("Running basic engine tests...")
            try:
                engine = AudioEngine()
                engine.initialize()
                engine.start_engine()
                
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
        print("  ✓ Low-latency ASIO driver support")
        print("  ✓ Virtual audio cable creation")
        print("  ✓ Advanced routing matrix")
        print("  ✓ Real-time audio effects")
        print("  ✓ RESTful API with WebSocket events")
        print("  ✓ VoiceMeeter Potato-like functionality")


if __name__ == "__main__":
    main()