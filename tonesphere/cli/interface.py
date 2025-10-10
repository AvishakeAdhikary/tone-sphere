from tonesphere.utils.config import ConfigManager
from tonesphere.core.engine_factory import UnifiedAudioEngine

class AudioEngineCLI:
    """Command-line interface for the audio engine"""
    
    def __init__(self):
        self.engine = None
        self.config_manager = ConfigManager()
        
    def initialize_engine(self):
        """Initialize the audio engine"""
        try:
            self.engine = UnifiedAudioEngine(self.config_manager)
            self.engine.initialize()
            self.engine.start_engine()
            
            # Show driver info
            driver_info = self.engine.get_driver_info()
            print("✓ Audio engine initialized successfully")
            print(f"  Driver: {driver_info.get('active_driver', 'unknown')}")
            print(f"  Available: {', '.join(self.engine.get_available_drivers())}")
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
    
    def refresh_devices(self):
        """Refresh device list to detect newly launched applications"""
        if not self.engine:
            print("Engine not initialized")
            return
        
        print("\n🔄 Refreshing device list...")
        success = self.engine.refresh_devices()
        
        if success:
            devices = self.engine.get_devices()
            print(f"✓ Device list refreshed: {len(devices)} devices found")
            print("\nTip: Run 'devices' command to see the updated list")
        else:
            print("✗ Failed to refresh devices")
    
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
    
    def show_network_info(self):
        """Show network information"""
        if not self.engine:
            print("Engine not initialized")
            return
        
        clients = self.engine.get_network_clients()
        connections = self.engine.get_network_connections()
        stats = self.engine.get_network_statistics()
        
        print("\nNetwork Information:")
        print(f"Incoming clients: {len(clients)}")
        for client in clients:
            print(f"  - {client}")
        
        print(f"Outgoing connections: {len(connections)}")
        for conn in connections:
            print(f"  - {conn}")
        
        if stats:
            print(f"\nNetwork Statistics:")
            print(f"  Packets sent: {stats.get('packets_sent', 0)}")
            print(f"  Packets received: {stats.get('packets_received', 0)}")
            print(f"  Bytes sent: {stats.get('bytes_sent', 0)}")
            print(f"  Quality: {stats.get('quality', 'unknown')}")
    
    def show_channel_controls(self):
        """Show channel controls for a device"""
        if not self.engine:
            print("Engine not initialized")
            return
        
        device_id = input("Enter device ID: ").strip()
        if not device_id.isdigit():
            print("Invalid device ID")
            return
        
        device_id = int(device_id)
        info = self.engine.get_device_channels(device_id)
        
        if not info:
            print(f"No channel control available for device {device_id}")
            return
        
        print(f"\nChannel Controls for Device {device_id}:")
        print(f"Master Volume: {info['master_volume']:.2f}")
        print(f"Master Muted: {info['master_muted']}")
        print(f"Channels Swapped: {info['channels_swapped']}")
        print(f"\nChannels ({info['num_channels']}):")
        
        for ch_idx, ch_info in info['channels'].items():
            print(f"  Channel {ch_idx}:")
            print(f"    Volume: {ch_info['volume']:.2f}")
            print(f"    Muted: {ch_info['muted']}")
            print(f"    Solo: {ch_info['solo']}")
            print(f"    Pan: {ch_info['pan']:.2f}")
            print(f"    Inverted: {ch_info['inverted']}")
    
    def set_channel_volume(self):
        """Set channel volume"""
        if not self.engine:
            print("Engine not initialized")
            return
        
        try:
            device_id = int(input("Enter device ID: ").strip())
            channel = int(input("Enter channel (0=L, 1=R): ").strip())
            volume = float(input("Enter volume (0.0-2.0): ").strip())
            
            self.engine.set_channel_volume(device_id, channel, volume)
            print(f"✓ Set channel {channel} volume to {volume}")
        except ValueError:
            print("Invalid input")
    
    def swap_channels(self):
        """Swap L/R channels"""
        if not self.engine:
            print("Engine not initialized")
            return
        
        try:
            device_id = int(input("Enter device ID: ").strip())
            self.engine.swap_channels(device_id)
            print(f"✓ Swapped channels for device {device_id}")
        except ValueError:
            print("Invalid device ID")
    
    def change_sample_rate(self):
        """Change sample rate"""
        if not self.engine:
            print("Engine not initialized")
            return
        
        try:
            current = self.engine.sample_rate
            print(f"Current sample rate: {current}Hz")
            new_rate = int(input("Enter new sample rate (8000-192000): ").strip())
            
            if 8000 <= new_rate <= 192000:
                self.engine.set_sample_rate(new_rate)
                print(f"✓ Sample rate changed to {new_rate}Hz")
                print("Note: Some devices may require engine restart")
            else:
                print("Sample rate must be between 8000 and 192000")
        except ValueError:
            print("Invalid sample rate")
    
    def manage_virtual_devices(self):
        """Manage virtual devices"""
        if not self.engine:
            print("Engine not initialized")
            return
        
        while True:
            print("\nVirtual Device Management:")
            print("1. List virtual devices")
            print("2. Create virtual input")
            print("3. Create virtual output")
            print("4. Delete virtual device")
            print("5. Back to main menu")
            
            choice = input("\nChoice: ").strip()
            
            if choice == "1":
                self._list_virtual_devices()
            elif choice == "2":
                self._create_virtual_input()
            elif choice == "3":
                self._create_virtual_output()
            elif choice == "4":
                self._delete_virtual_device()
            elif choice == "5":
                break
            else:
                print("Invalid choice")
    
    def _list_virtual_devices(self):
        """List virtual devices"""
        devices = self.engine.get_devices()
        virtual_devices = [d for d in devices if 'virtual' in d['type'].lower()]
        
        if not virtual_devices:
            print("No virtual devices")
            return
        
        print("\nVirtual Devices:")
        print("-" * 80)
        print(f"{'ID':<6} {'Name':<40} {'Type':<15} {'Channels':<10}")
        print("-" * 80)
        
        for device in virtual_devices:
            print(f"{device['id']:<6} {device['name']:<40} {device['type']:<15} {device['channels']:<10}")
    
    def _create_virtual_input(self):
        """Create virtual input"""
        try:
            channels = int(input("Enter number of channels (default 2): ").strip() or "2")
            device_id = self.engine.create_virtual_input(f"Virtual Input", channels)
            if device_id:
                print(f"✓ Created virtual input with ID: {device_id}")
            else:
                print("✗ Failed to create virtual input (limit reached?)")
        except ValueError:
            print("Invalid input")
    
    def _create_virtual_output(self):
        """Create virtual output"""
        try:
            channels = int(input("Enter number of channels (default 2): ").strip() or "2")
            device_id = self.engine.create_virtual_output(f"Virtual Output", channels)
            if device_id:
                print(f"✓ Created virtual output with ID: {device_id}")
            else:
                print("✗ Failed to create virtual output (limit reached?)")
        except ValueError:
            print("Invalid input")
    
    def _delete_virtual_device(self):
        """Delete virtual device"""
        try:
            device_id = int(input("Enter device ID to delete: ").strip())
            # Note: Need to add delete method to engine
            print(f"Device deletion not yet implemented in engine")
        except ValueError:
            print("Invalid device ID")
    
    def connect_to_network(self):
        """Connect to another ToneSphere instance"""
        if not self.engine:
            print("Engine not initialized")
            return
        
        host = input("Enter host address: ").strip()
        port = input("Enter port (default 9001): ").strip() or "9001"
        
        try:
            port = int(port)
            success = self.engine.connect_to_network(host, port)
            if success:
                print(f"✓ Connected to {host}:{port}")
            else:
                print(f"✗ Failed to connect to {host}:{port}")
        except ValueError:
            print("Invalid port number")
    
    def enable_logging(self):
        """Enable file logging"""
        from tonesphere.utils.logger import enable_file_logging
        enable_file_logging()
        print("✓ File logging enabled (logs/tonesphere.log)")
    
    def show_log_stats(self):
        """Show logging statistics"""
        from tonesphere.utils.logger import get_log_stats
        stats = get_log_stats()
        
        print("\nLogging Statistics:")
        print(f"File logging: {'Enabled' if stats['file_logging_enabled'] else 'Disabled'}")
        if stats['log_file_path']:
            print(f"Log file: {stats['log_file_path']}")
            if 'log_file_size' in stats:
                size_mb = stats['log_file_size'] / (1024 * 1024)
                print(f"Log file size: {size_mb:.2f} MB")
        print(f"Active loggers: {stats['active_loggers']}")
        print(f"Structured logging: {stats['structured_logging']}")
    
    def run_interactive_mode(self):
        """Run interactive CLI mode"""
        if not self.initialize_engine():
            return
            
        print("\nToneSphere Interactive Mode")
        print("Type 'help' for available commands")
        
        while True:
            try:
                command = input("\n> ").strip().lower()
                
                if command in ["quit", "exit"]:
                    break
                elif command == "devices":
                    self.list_devices()
                elif command == "refresh":
                    self.refresh_devices()
                elif command == "routing":
                    self.create_test_routing()
                elif command == "matrix":
                    self.show_routing_matrix()
                elif command in ["performance", "stats"]:
                    self.show_performance()
                elif command == "network":
                    self.show_network_info()
                elif command == "drivers":
                    driver_info = self.engine.get_driver_info()
                    print(f"\nActive driver: {driver_info.get('active_driver', 'unknown')}")
                    print(f"Available drivers: {', '.join(self.engine.get_available_drivers())}")
                elif command == "channels":
                    self.show_channel_controls()
                elif command == "setchannel":
                    self.set_channel_volume()
                elif command == "swap":
                    self.swap_channels()
                elif command == "samplerate":
                    self.change_sample_rate()
                elif command == "virtual":
                    self.manage_virtual_devices()
                elif command == "connect":
                    self.connect_to_network()
                elif command == "logging":
                    self.enable_logging()
                elif command == "logstats":
                    self.show_log_stats()
                elif command == "help":
                    print("\n" + "="*60)
                    print("ToneSphere CLI Commands")
                    print("="*60)
                    print("\nDevice Management:")
                    print("  devices      - List all audio devices")
                    print("  refresh      - Refresh device list (detect new apps)")
                    print("  virtual      - Manage virtual devices (CRUD)")
                    print("\nAudio Routing:")
                    print("  routing      - Create test routing")
                    print("  matrix       - Show routing matrix")
                    print("\nChannel Controls:")
                    print("  channels     - Show channel controls for device")
                    print("  setchannel   - Set channel volume")
                    print("  swap         - Swap L/R channels")
                    print("\nAudio Settings:")
                    print("  samplerate   - Change sample rate")
                    print("  drivers      - Show driver information")
                    print("\nNetwork:")
                    print("  network      - Show network information")
                    print("  connect      - Connect to remote instance")
                    print("\nMonitoring:")
                    print("  performance  - Show performance stats")
                    print("  logging      - Enable file logging")
                    print("  logstats     - Show logging statistics")
                    print("\nGeneral:")
                    print("  help         - Show this help")
                    print("  quit         - Exit")
                    print("="*60)
                else:
                    print(f"Unknown command: {command}. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                break
            except EOFError:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("\nShutting down...")
        if self.engine:
            self.engine.stop_engine()