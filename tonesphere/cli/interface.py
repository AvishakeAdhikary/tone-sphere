from tonesphere.core.engine_factory import UnifiedAudioEngine
from tonesphere.utils.config import ConfigManager
from tonesphere.utils.formatting import format_measurement


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
        print("-" * 96)
        # 'Backend' replaces the old 'ASIO' column: with PortAudio the backend actually in
        # use is the useful fact, and the old column reported ASIO on devices without it.
        print(f"{'ID':<4} {'Name':<44} {'Dir':<7} {'Ch':<3} {'Backend':<22} {'Latency':<8}")
        print("-" * 96)

        for device in devices:
            name = device['name']
            if len(name) > 43:
                name = name[:40] + '...'
            print(f"{device['id']:<4} {name:<44} {device['direction']:<7} "
                  f"{device['channels']:<3} {device['host_api'][:21]:<22} "
                  f"{device['latency_ms']:.1f} ms")

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
        print("\nPerformance Statistics:")
        print(f"CPU Usage: {format_measurement(stats.get('cpu_usage'), '%')}")
        print(f"Buffer Underruns: {stats['buffer_underruns']}")
        print(f"Latency (measured): {format_measurement(stats.get('measured_latency_ms'), 'ms')}")
        print(f"Latency (nominal):  {format_measurement(stats.get('nominal_latency_ms'), 'ms')}")
        if not stats.get('audio_path_active', False):
            print("Audio path: INACTIVE - no audio is being processed")

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
            print("\nTCP Statistics:")
            print(f"  Packets sent: {stats.get('packets_sent', 0)}")
            print(f"  Packets received: {stats.get('packets_received', 0)}")
            print(f"  Bytes sent: {stats.get('bytes_sent', 0)}")
            print(f"  Quality: {stats.get('quality', 'unknown')}")

        self._show_udp_info(stats.get('udp') if stats else None)

    def _show_udp_info(self, udp: dict | None):
        """
        The realtime transport's own section.

        Every number printed here is a count of something that happened. A jitter buffer
        that has not seen a packet prints as such rather than as a row of zeros, which
        would read as a healthy link delivering silence.
        """
        if not udp:
            return

        transport = udp.get('transport', {})
        print("\nUDP Transport (realtime):")
        if transport.get('running'):
            print(f"  Listening on: {transport['bound_host']}:{transport['bound_port']}")
        else:
            print("  Listening on: not running")
        print(f"  Codec: {transport.get('codec', 'unknown')} "
              f"(quality {transport.get('quality', 'unknown')})")
        print(f"  Packets sent / received: {transport.get('packets_sent', 0)} / "
              f"{transport.get('packets_received', 0)}")
        print(f"  Datagrams rejected: {transport.get('packets_rejected', 0)}")

        peers = transport.get('peers') or {}
        print(f"  Peers: {', '.join(f'{n}={a}' for n, a in peers.items()) or 'none'}")

        if not transport.get('opus_available', False):
            print("  Opus: not implemented (PCM only) — see network/udp_transport.py")

        routes = (udp.get('send') or {}).get('routes') or {}
        if routes:
            print("\n  Sending:")
            for route in routes.values():
                ring = route.get('ring') or {}
                print(f"    device {route['device_id']} -> {route['target'] or 'all peers'}: "
                      f"{route['packets_sent']} packet(s), "
                      f"{route['frames_read']} frame(s) read, "
                      f"{ring.get('underflow_count', 0)} starved read(s)")
                if route.get('last_error'):
                    print(f"      last error: {route['last_error']}")

        for entry in (udp.get('receive') or {}).values():
            print(f"\n  Receiving on device {entry['device_id']}:")
            print(f"    Packets received: {entry['packets_received']} "
                  f"({entry['packets_rejected']} rejected)")
            print(f"    Frames written: {entry['frames_written']} "
                  f"({entry['writes_refused']} refused — nothing routed out of the bus)")
            print(f"    Playout thread alive: {entry['playout_alive']}")

            buffer = entry.get('jitter_buffer')
            if buffer is None:
                print("    Jitter buffer: no packet has arrived yet")
            else:
                print(f"    Jitter buffer: {buffer['buffered_packets']} packet(s) / "
                      f"{buffer['buffered_latency_ms']:.1f} ms buffered, "
                      f"target {buffer['target_latency_ms']:.0f} ms"
                      f"{' (priming)' if buffer['priming'] else ''}")
                print(f"    Played on time: {buffer['packets_played_on_time']}, "
                      f"lost: {buffer['packets_lost']}, "
                      f"late: {buffer['packets_late_dropped']}, "
                      f"resyncs: {buffer['resync_events']}")

            if entry.get('error'):
                print(f"    Last error: {entry['error']}")

    def send_to_network(self):
        """Start streaming a device's or bus's audio to the network"""
        if not self.engine:
            print("Engine not initialized")
            return

        try:
            device_id = int(input("Enter device/bus ID to send: ").strip())
        except ValueError:
            print("Invalid device ID")
            return

        target = input("Peer name (blank = every registered peer): ").strip() or None
        transport = input("Transport [udp/tcp] (default udp): ").strip().lower() or "udp"

        success, message = self.engine.send_device_to_network(device_id, target, transport)
        print(f"{'✓' if success else '✗'} {message}")

    def manage_udp_transport(self):
        """Start/stop the realtime UDP transport and manage its peers"""
        if not self.engine:
            print("Engine not initialized")
            return

        print("\nUDP transport:")
        print("  1. Start listening")
        print("  2. Stop")
        print("  3. Add peer")
        print("  4. Remove peer")
        print("  5. List peers")
        print("  6. Receive network audio into a bus")
        print("  7. Stop receiving on a bus")
        print("  8. Stop sending from a device")

        choice = input("\nSelect (1-8): ").strip()

        if choice == "1":
            host = input("Bind address (default 127.0.0.1): ").strip() or "127.0.0.1"
            port = input("Bind port (default 9002): ").strip() or "9002"
            if not port.isdigit():
                print("Invalid port number")
                return
            # 0.0.0.0 is offered but never the default: it is what raises a firewall
            # prompt, and a local monitoring setup does not need it.
            success, message = self.engine.start_udp_transport(host, int(port))
            print(f"{'✓' if success else '✗'} {message}")

        elif choice == "2":
            self.engine.stop_udp_transport()
            print("✓ UDP transport stopped")

        elif choice == "3":
            name = input("Peer name: ").strip()
            host = input("Peer address: ").strip()
            port = input("Peer port (default 9002): ").strip() or "9002"
            if not name or not host or not port.isdigit():
                print("Name, address and a numeric port are all required")
                return
            self.engine.add_udp_peer(name, host, int(port))
            print(f"✓ Peer '{name}' is {host}:{port}")

        elif choice == "4":
            name = input("Peer name: ").strip()
            if self.engine.remove_udp_peer(name):
                print(f"✓ Removed peer '{name}'")
            else:
                print(f"✗ No peer named '{name}'")

        elif choice == "5":
            peers = self.engine.get_udp_peers()
            if not peers:
                print("No peers registered")
            for name, address in peers.items():
                print(f"  {name}: {address}")

        elif choice == "6":
            try:
                device_id = int(input("Bus ID to receive into: ").strip())
                latency = float(input("Jitter buffer target ms (default 40): ").strip() or "40")
            except ValueError:
                print("Invalid input")
                return
            success, message = self.engine.register_network_receive(
                device_id, 'udp', latency
            )
            print(f"{'✓' if success else '✗'} {message}")

        elif choice == "7":
            try:
                device_id = int(input("Bus ID to stop receiving on: ").strip())
            except ValueError:
                print("Invalid device ID")
                return
            if self.engine.unregister_network_receive(device_id):
                print(f"✓ Device {device_id} is no longer receiving UDP audio")
            else:
                print(f"✗ Device {device_id} was not receiving UDP audio")

        elif choice == "8":
            try:
                device_id = int(input("Device ID to stop sending: ").strip())
            except ValueError:
                print("Invalid device ID")
                return
            if self.engine.disable_network_send(device_id):
                print(f"✓ Device {device_id} is no longer sending")
            else:
                print(f"✗ Device {device_id} was not sending to the network")

        else:
            print("Invalid selection")

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
            device_id = self.engine.create_virtual_input("Virtual Input", channels)
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
            device_id = self.engine.create_virtual_output("Virtual Output", channels)
            if device_id:
                print(f"✓ Created virtual output with ID: {device_id}")
            else:
                print("✗ Failed to create virtual output (limit reached?)")
        except ValueError:
            print("Invalid input")

    def _delete_virtual_device(self):
        """Delete a bus by id"""
        if not self.engine:
            print("Engine not initialized")
            return

        try:
            device_id = int(input("Enter device ID to delete: ").strip())
        except ValueError:
            print("Invalid device ID")
            return

        if self.engine.delete_virtual_device(device_id):
            print(f"✓ Deleted device {device_id}")
        else:
            print(f"✗ No such bus: {device_id} (only buses can be deleted, not hardware)")

    def manage_linux_sink(self):
        """
        Create or remove a Linux OS-visible virtual sink (Track 2).

        Kept in this session's interactive loop rather than as a one-shot `main.py`
        subcommand: the sink itself is torn down in `engine.cleanup()` when this session
        ends (see `core/engine.py`'s `_teardown_all_system_virtual_devices`), and the device id it
        gets back is only meaningful for the `AudioEngine` instance that created it — the
        same reason bus CRUD lives here rather than as a one-shot command too.
        """
        if not self.engine:
            print("Engine not initialized")
            return

        import platform

        if platform.system() != "Linux":
            print(
                f"\nLinux virtual sinks need pactl and a running PulseAudio/PipeWire "
                f"server — unavailable on {platform.system()}"
            )
            return

        print("\nLinux Virtual Sink:")
        print("1. Create a sink")
        print("2. Remove a sink")
        print("3. Back to main menu")

        choice = input("\nChoice: ").strip()

        if choice == "1":
            name = input("Sink name: ").strip() or "ToneSphere"
            channels_raw = input("Channels (default 2): ").strip()
            channels = int(channels_raw) if channels_raw.isdigit() else 2

            device_id = self.engine.create_linux_system_sink(name, channels)
            if device_id is None:
                print(f"✗ Could not create '{name}' — see the log for the real reason")
            else:
                print(f"✓ Created '{name}' -> device {device_id}")
                print(f"  Route audio to device {device_id} to send it there")
        elif choice == "2":
            try:
                device_id = int(input("Device ID to remove: ").strip())
            except ValueError:
                print("Invalid device ID")
                return
            if self.engine.remove_linux_system_sink(device_id):
                print(f"✓ Removed the sink at device {device_id}")
            else:
                print(f"✗ No Linux virtual sink at device {device_id}")
        elif choice == "3":
            return
        else:
            print("Invalid choice")

    def manage_app_capture(self):
        """Capture one application's audio output by process id."""
        if not self.engine:
            print("Engine not initialized")
            return

        from tonesphere.engine.app_capture import capture_status

        status = capture_status()

        print(f"\nPer-Application Capture ({status['platform']})")
        print(f"  Platform supports it: {status['process_loopback_supported']}")
        print(f"  Implemented here:     {status['process_loopback_implemented']}")
        print(f"  Whole-system loopback available: {status['system_loopback_available']}")

        if not status['process_loopback_implemented']:
            print(f"\n  {status['note']}")
            return

        while True:
            print("\n1. List applications playing audio")
            print("2. Capture an application")
            print("3. Show running captures")
            print("4. Stop a capture")
            print("5. Back to main menu")

            choice = input("\nChoice: ").strip()

            if choice == "1":
                self._list_audio_sessions()
            elif choice == "2":
                self._start_app_capture()
            elif choice == "3":
                self._show_app_captures()
            elif choice == "4":
                self._stop_app_capture()
            elif choice == "5":
                break
            else:
                print("Invalid choice")

    def _list_audio_sessions(self):
        """Applications the OS reports as holding an audio session."""
        from tonesphere.engine.app_capture import list_audio_sessions

        sessions = list_audio_sessions()
        if not sessions:
            print("\nNo applications are holding an audio session")
            return

        print("\nApplications with audio sessions:")
        print("-" * 52)
        print(f"{'PID':<8} {'Application':<32} {'State':<10}")
        print("-" * 52)

        for session in sessions:
            state = "playing" if session.is_active else "idle"
            print(f"{session.pid:<8} {session.display_name[:31]:<32} {state:<10}")

    def _start_app_capture(self):
        from tonesphere.engine.process_capture import ProcessCaptureError

        try:
            pid = int(input("Enter process ID: ").strip())
        except ValueError:
            print("Invalid process ID")
            return

        name = input("Bus name (blank for a default): ").strip() or None
        tree = input("Include child processes? [Y/n]: ").strip().lower() != 'n'

        try:
            bus_id = self.engine.engine.start_process_capture(pid, name, tree)
        except ProcessCaptureError as e:
            print(f"✗ {e}")
            return

        capture = self.engine.engine.process_capture_status(bus_id)[0]
        print(f"✓ Capturing process {pid} into bus {bus_id} ({capture['format']})")
        print(f"  Route bus {bus_id} to an output to hear it")

    def _show_app_captures(self):
        """Only measured values; an unmeasured one prints as '--', never as 0."""
        captures = self.engine.engine.process_capture_status()

        if not captures:
            print("\nNo captures running")
            return

        print("\nRunning captures:")
        for capture in captures:
            print(f"\n  Bus {capture['bus_id']}: {capture['name']} (pid {capture['pid']})")
            print(f"    Running:  {capture['running']}")
            print(f"    Format:   {capture['format'] or format_measurement(None)}")
            print(f"    Measured: {format_measurement(capture['measured_sample_rate'], ' Hz')}")
            print(f"    Captured: {format_measurement(capture['seconds_captured'], ' s')} "
                  f"({capture['frames_captured']} frames)")
            print(f"    Glitches: {capture['glitch_count']}")
            if capture['unrouted_writes']:
                print(f"    Unrouted writes: {capture['unrouted_writes']} "
                      f"(nothing is routed out of this bus yet)")
            if capture['error']:
                print(f"    ERROR: {capture['error']}")

    def _stop_app_capture(self):
        try:
            bus_id = int(input("Enter the bus ID of the capture to stop: ").strip())
        except ValueError:
            print("Invalid bus ID")
            return

        if self.engine.engine.stop_process_capture(bus_id):
            print(f"✓ Stopped the capture feeding bus {bus_id}")
        else:
            print(f"✗ No capture is feeding bus {bus_id}")

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
                elif command == "linuxsink":
                    self.manage_linux_sink()
                elif command == "appcapture":
                    self.manage_app_capture()
                elif command == "connect":
                    self.connect_to_network()
                elif command == "netsend":
                    self.send_to_network()
                elif command == "netudp":
                    self.manage_udp_transport()
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
                    print("  linuxsink    - Create/remove a Linux OS-visible virtual sink")
                    print("  appcapture   - Capture one application's audio by PID")
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
                    print("  connect      - Connect to remote instance (TCP)")
                    print("  netsend      - Stream a device/bus to the network")
                    print("  netudp       - Realtime UDP transport, peers and receive")
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
