from tonesphere.utils.config import ConfigManager
from tonesphere.core.engine import AudioEngine

class AudioEngineCLI:
    """Command-line interface for the audio engine"""
    
    def __init__(self):
        self.engine = None
        self.config_manager = ConfigManager()
        
    def initialize_engine(self):
        """Initialize the audio engine"""
        try:
            config = self.config_manager.load_config()
            self.engine = AudioEngine(
                sample_rate=config['engine']['sample_rate'],
                buffer_size=config['engine']['buffer_size']
            )
            self.engine.initialize()
            self.engine.start_engine()
            print("✓ Audio engine initialized successfully")
        except Exception as e:
            print(f"✗ Failed to initialize audio engine: {e}")
            return False
        return True
    
    def list_devices(self):
        """List all available devices"""
        if not self.engine:
            print("Engine not initialized")
            return
            
        devices = self.engine.get_devices()
        print("\nAvailable Audio Devices:")
        print("-" * 80)
        print(f"{'ID':<4} {'Name':<40} {'Type':<15} {'Ch':<3} {'ASIO':<6} {'Latency':<8}")
        print("-" * 80)
        
        for device in devices:
            print(f"{device['id']:<4} {device['name']:<40} {device['type']:<15} "
                  f"{device['channels']:<3} {'✓' if device['is_asio'] else '✗':<6} "
                  f"{device['latency_ms']:.1f}ms")
    
    def create_test_routing(self):
        """Create a test routing setup"""
        if not self.engine:
            print("Engine not initialized")
            return
            
        devices = self.engine.get_devices()
        inputs = [d for d in devices if 'input' in d['type']]
        outputs = [d for d in devices if 'output' in d['type']]
        
        if not inputs or not outputs:
            print("No suitable input/output devices found")
            return
            
        # Create routing from first input to first output
        source_id = inputs[0]['id']
        dest_id = outputs[0]['id']
        
        success = self.engine.create_routing(source_id, dest_id, 1.0)
        if success:
            print(f"✓ Created routing: {inputs[0]['name']} -> {outputs[0]['name']}")
        else:
            print("✗ Failed to create routing")
    
    def show_routing_matrix(self):
        """Display current routing matrix"""
        if not self.engine:
            print("Engine not initialized")
            return
            
        matrix = self.engine.get_routing_matrix()
        if not matrix:
            print("No active routings")
            return
            
        print("\nActive Routings:")
        print("-" * 60)
        print(f"{'Source':<6} {'Destination':<6} {'Volume':<8} {'Status':<10}")
        print("-" * 60)
        
        for connection in matrix.values():
            status = "MUTED" if connection['muted'] else "ACTIVE"
            if connection['solo']:
                status += " (SOLO)"
                
            print(f"{connection['source_id']:<6} {connection['destination_id']:<6} "
                  f"{connection['volume']:.2f}x{'':<3} {status:<10}")
    
    def show_performance(self):
        """Show performance statistics"""
        if not self.engine:
            print("Engine not initialized")
            return
            
        stats = self.engine.get_performance_stats()
        print(f"\nPerformance Statistics:")
        print(f"CPU Usage: {stats['cpu_usage']:.1f}%")
        print(f"Buffer Underruns: {stats['buffer_underruns']}")
        print(f"Latency: {stats['latency_ms']:.1f}ms")
    
    def run_interactive_mode(self):
        """Run interactive CLI mode"""
        if not self.initialize_engine():
            return
            
        print("\nPyAudioEngine Interactive Mode")
        print("Commands: devices, routing, matrix, performance, quit")
        
        while True:
            try:
                command = input("\n> ").strip().lower()
                
                if command in ["quit", "exit"]:
                    break
                elif command == "devices":
                    self.list_devices()
                elif command == "routing":
                    self.create_test_routing()
                elif command == "matrix":
                    self.show_routing_matrix()
                elif command in ["performance", "stats"]:
                    self.show_performance()
                elif command == "help":
                    print("Available commands:")
                    print("  devices    - List all audio devices")
                    print("  routing    - Create test routing")
                    print("  matrix     - Show routing matrix")
                    print("  performance- Show performance stats")
                    print("  quit       - Exit")
                else:
                    print(f"Unknown command: {command}")
                    
            except KeyboardInterrupt:
                break
            except EOFError:
                break
        
        print("\nShutting down...")
        if self.engine:
            self.engine.stop_engine()