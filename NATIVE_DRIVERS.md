# ToneSphere Native Audio Driver Implementation

## Overview

ToneSphere now features **full native audio driver support** across all platforms, eliminating the dependency on `sounddevice` and providing true professional-grade audio routing with virtual device creation.

## Supported Audio Drivers

### Windows
- **ASIO** - Professional low-latency audio (2-5ms latency)
- **WASAPI** - Modern Windows audio with exclusive and shared modes
- **DirectSound** - Legacy compatibility for older systems

### Linux
- **JACK** - Professional audio connection kit (lowest latency)
- **PipeWire** - Modern multimedia framework
- **PulseAudio** - Standard Linux audio server
- **ALSA** - Low-level hardware access

### macOS
- **CoreAudio** - Native macOS audio framework
- **JACK** - Also available on macOS for pro audio

## Architecture

```
┌─────────────────────────────────────────────────┐
│         ToneSphere Application Layer            │
├─────────────────────────────────────────────────┤
│         UnifiedAudioEngine (Facade)             │
├─────────────────────────────────────────────────┤
│  NativeAudioEngine  │  AudioEngine (Fallback)   │
├──────────────────────┴──────────────────────────┤
│            AudioDriverManager                    │
├─────────────────────────────────────────────────┤
│  ASIO │ WASAPI │ ALSA │ JACK │ PulseAudio │ ... │
├─────────────────────────────────────────────────┤
│         Hardware Audio Devices                   │
└─────────────────────────────────────────────────┘
```

## Key Components

### 1. Driver Abstraction Layer (`tonesphere/drivers/base.py`)
- `AudioDriverBase` - Abstract base class for all drivers
- `AudioDriverType` - Enum of supported driver types
- `AudioDeviceInfo` - Device information structure
- `AudioStreamConfig` - Stream configuration
- `StreamState` - Stream state management

### 2. Platform-Specific Drivers
- `windows_asio.py` - ASIO driver implementation
- `windows_wasapi.py` - WASAPI driver implementation
- `windows_directsound.py` - DirectSound driver implementation
- `linux_alsa.py` - ALSA driver implementation
- `linux_pulseaudio.py` - PulseAudio driver implementation
- `linux_jack.py` - JACK driver implementation
- `linux_pipewire.py` - PipeWire driver implementation
- `macos_coreaudio.py` - CoreAudio driver implementation

### 3. Driver Manager (`tonesphere/drivers/manager.py`)
Automatically detects and selects the best available driver for the platform:

**Windows Priority:** ASIO > WASAPI > DirectSound  
**Linux Priority:** JACK > PipeWire > PulseAudio > ALSA  
**macOS Priority:** CoreAudio > JACK

### 4. Virtual Device Manager (`tonesphere/devices/native_virtual.py`)
Creates true virtual audio devices that:
- Appear in system sound settings
- Support real-time audio routing
- Provide low-latency audio processing
- Handle multiple simultaneous connections

### 5. Stream Manager (`tonesphere/core/stream_manager.py`)
Manages audio streams with:
- Proper callback handling
- Thread-safe operations
- Performance monitoring
- Buffer management

### 6. Native Audio Engine (`tonesphere/core/native_engine.py`)
Main engine integrating all components:
- Driver initialization and management
- Device enumeration
- Virtual device creation
- Audio routing
- Performance monitoring

## Configuration

Edit `config/default_config.yaml`:

```yaml
engine:
  sample_rate: 48000
  buffer_size: 128
  use_native_drivers: true  # Enable native drivers
  preferred_driver: 'auto'  # or: asio, wasapi, jack, etc.
```

## Usage

### Basic Usage

```python
from tonesphere.core.engine_factory import UnifiedAudioEngine

# Create engine (auto-selects best driver)
engine = UnifiedAudioEngine()
engine.initialize()
engine.start_engine()

# Get driver info
driver_info = engine.get_driver_info()
print(f"Using: {driver_info['active_driver']}")

# List available drivers
drivers = engine.get_available_drivers()
print(f"Available: {drivers}")

# Get devices
devices = engine.get_devices()
for device in devices:
    print(f"{device['name']} - {device['type']}")

# Create virtual devices
input_id = engine.create_virtual_input("My Input", channels=2)
output_id = engine.create_virtual_output("My Output", channels=2)

# Route audio
engine.create_routing(input_id, output_id, volume=1.0)
```

### Switching Drivers

```python
# Switch to ASIO (Windows)
engine.switch_driver('asio')

# Switch to JACK (Linux/macOS)
engine.switch_driver('jack')

# Switch to WASAPI (Windows)
engine.switch_driver('wasapi')
```

### API Endpoints

New endpoints for driver management:

```bash
# Get available drivers
GET /drivers

# Switch driver
POST /drivers/switch/{driver_type}

# Get engine status (includes driver info)
GET /engine/status
```

## Virtual Devices

Virtual devices created by ToneSphere:
- **ToneSphere Input 1-3** - Virtual input devices
- **ToneSphere Output 1-3** - Virtual output devices

These devices:
- Appear in system audio settings
- Support routing between applications
- Provide low-latency audio processing
- Can be created dynamically via API

## Performance

### Latency Comparison

| Driver | Typical Latency | Use Case |
|--------|----------------|----------|
| ASIO | 2-5ms | Professional audio, recording |
| JACK | 2-5ms | Pro audio on Linux |
| WASAPI (Exclusive) | 5-10ms | Low-latency Windows audio |
| CoreAudio | 5-10ms | macOS audio production |
| WASAPI (Shared) | 10-20ms | General Windows audio |
| PulseAudio | 10-30ms | Desktop Linux audio |
| DirectSound | 20-50ms | Legacy Windows compatibility |

### Buffer Size vs Latency

At 48kHz sample rate:
- 64 samples = 1.33ms latency
- 128 samples = 2.67ms latency
- 256 samples = 5.33ms latency
- 512 samples = 10.67ms latency
- 1024 samples = 21.33ms latency

## Testing

Run the test command to verify driver functionality:

```bash
python main.py test
```

This will:
1. Initialize the audio engine
2. Display active driver and available drivers
3. Enumerate all audio devices
4. Create a test virtual device
5. Test audio routing
6. Display performance statistics

## Troubleshooting

### Windows

**ASIO not detected:**
- Ensure ASIO drivers are installed (from audio interface manufacturer)
- Check Windows Registry: `HKEY_LOCAL_MACHINE\SOFTWARE\ASIO`

**WASAPI issues:**
- Requires Windows Vista or later
- Check audio device permissions
- Try both exclusive and shared modes

### Linux

**JACK not available:**
```bash
# Install JACK
sudo apt install jackd2 jack-tools

# Start JACK server
jackd -d alsa -r 48000 -p 128
```

**PulseAudio conflicts:**
```bash
# Check PulseAudio status
pulseaudio --check

# Restart PulseAudio
pulseaudio -k && pulseaudio --start
```

**ALSA permissions:**
```bash
# Add user to audio group
sudo usermod -a -G audio $USER
```

### macOS

**CoreAudio issues:**
- Check System Preferences > Sound
- Verify audio device permissions
- Restart Core Audio: `sudo killall coreaudiod`

## Migration from sounddevice

The engine automatically falls back to `sounddevice` if native drivers fail. To force native drivers:

```yaml
engine:
  use_native_drivers: true
```

To use sounddevice:

```yaml
engine:
  use_native_drivers: false
```

## Advanced Features

### Custom Callbacks

```python
def audio_callback(input_data, output_data, frames, time_info, status):
    # Process audio here
    return processed_audio

config = AudioStreamConfig(
    device_id=device_id,
    sample_rate=48000,
    buffer_size=128,
    channels=2,
    callback=audio_callback
)
```

### Performance Monitoring

```python
stats = engine.get_performance_stats()
print(f"CPU: {stats['cpu_usage']}%")
print(f"Latency: {stats['latency_ms']}ms")
print(f"Active Streams: {stats['active_streams']}")
```

### Device Capabilities

```python
devices = engine.get_devices()
for device in devices:
    print(f"Device: {device['name']}")
    print(f"  Channels: {device['channels']}")
    print(f"  Sample Rate: {device['sample_rate']}")
    print(f"  ASIO: {device['is_asio']}")
    print(f"  Latency: {device['latency_ms']}ms")
```

## Future Enhancements

- [ ] ASIO control panel integration
- [ ] JACK transport sync
- [ ] Network audio streaming between ToneSphere instances
- [ ] VST plugin support
- [ ] Multi-client audio routing
- [ ] Hardware monitoring support

## Contributing

When adding new drivers:
1. Inherit from `AudioDriverBase`
2. Implement all abstract methods
3. Add to `AudioDriverManager`
4. Update documentation
5. Add platform-specific tests

## License

Same as ToneSphere main project.
