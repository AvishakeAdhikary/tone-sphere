import asyncio
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from tonesphere.api.models import (
    CreateLinuxSinkRequest,
    CreateRoutingRequest,
    CreateVirtualDeviceRequest,
    DeviceInfo,
    PerformanceStats,
    SetVolumeRequest,
    StartProcessCaptureRequest,
)
from tonesphere.core.engine_factory import UnifiedAudioEngine
from tonesphere.utils.config import ConfigManager
from tonesphere.utils.logger import logger

# Global audio engine instance
audio_engine: UnifiedAudioEngine | None = None
config_manager = ConfigManager()
connected_websockets: set = set()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan handler"""
    global audio_engine

    # Startup
    audio_engine = UnifiedAudioEngine(config_manager)

    try:
        audio_engine.initialize()
        audio_engine.start_engine()
        logger.info("Audio engine started successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to start audio engine: {e}")
        raise
    finally:
        # Shutdown
        if audio_engine:
            audio_engine.stop_engine()
        logger.info("Audio engine stopped")

# Create FastAPI app
app = FastAPI(
    title="ToneSphere API",
    description="Professional Audio Routing Engine API",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# API Endpoints
# ==============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ToneSphere API",
        "version": "1.0.0",
        "status": "running" if audio_engine and audio_engine.is_running else "stopped"
    }

@app.get("/devices", response_model=list[DeviceInfo])
async def get_devices():
    """Get all available audio devices"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    devices = audio_engine.get_devices()
    return [DeviceInfo(**device) for device in devices]

@app.post("/devices/refresh")
async def refresh_devices():
    """Refresh device list to detect newly launched applications"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    success = audio_engine.refresh_devices()

    if success:
        devices = audio_engine.get_devices()
        return {
            "success": True,
            "message": "Device list refreshed successfully",
            "device_count": len(devices)
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to refresh devices")

@app.post("/devices/virtual")
async def create_virtual_device(request: CreateVirtualDeviceRequest):
    """Create a new virtual audio device"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    if request.device_type.lower() == "input":
        device_id = audio_engine.create_virtual_input(request.name, request.channels)
    elif request.device_type.lower() == "output":
        device_id = audio_engine.create_virtual_output(request.name, request.channels)
    else:
        raise HTTPException(status_code=400, detail="Invalid device type")

    return {"device_id": device_id, "message": "Virtual device created successfully"}

@app.post("/routing")
async def create_routing(request: CreateRoutingRequest):
    """Create a routing connection"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    success, message = audio_engine.create_routing(
        request.source_id,
        request.destination_id,
        request.volume
    )

    return {"success": success, "message": message}

@app.delete("/routing/{source_id}/{destination_id}")
async def remove_routing(source_id: int, destination_id: int):
    """Remove a routing connection"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    success = audio_engine.remove_routing(source_id, destination_id)

    if success:
        return {"message": "Routing removed successfully"}
    else:
        raise HTTPException(status_code=404, detail="Routing not found")

@app.put("/routing/volume")
async def set_routing_volume(request: SetVolumeRequest):
    """Set volume for a routing connection"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    audio_engine.set_routing_volume(
        request.source_id,
        request.destination_id,
        request.volume
    )

    return {"message": "Volume updated successfully"}

@app.get("/routing")
async def get_routing_matrix():
    """Get current routing matrix"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    return audio_engine.get_routing_matrix()

@app.get("/performance", response_model=PerformanceStats)
async def get_performance_stats():
    """Get engine performance statistics"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    stats = audio_engine.get_performance_stats()
    return PerformanceStats(**stats)

@app.post("/engine/start")
async def start_engine():
    """Start the audio engine"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    try:
        audio_engine.start_engine()
        return {"message": "Audio engine started successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start engine: {str(e)}") from e

@app.post("/engine/stop")
async def stop_engine():
    """Stop the audio engine"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    audio_engine.stop_engine()
    return {"message": "Audio engine stopped successfully"}

@app.get("/engine/status")
async def get_engine_status():
    """Get engine status"""
    if not audio_engine:
        return {"status": "not_initialized"}

    status = {
        "status": "running" if audio_engine.is_running else "stopped",
        "sample_rate": audio_engine.sample_rate,
        "buffer_size": audio_engine.buffer_size,
        "master_volume": audio_engine.master_volume,
        "driver_info": audio_engine.get_driver_info(),
        "available_drivers": audio_engine.get_available_drivers()
    }
    return status

@app.websocket("/ws/events")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time events with proper client management"""
    await websocket.accept()
    connected_websockets.add(websocket)

    try:
        while True:
            # Send performance stats every second
            if audio_engine:
                stats = audio_engine.get_performance_stats()
                network_clients = audio_engine.get_network_clients()

                await websocket.send_json({
                    "type": "performance_stats",
                    "data": stats
                })

                await websocket.send_json({
                    "type": "network_clients",
                    "data": {"clients": network_clients, "count": len(network_clients)}
                })

            await asyncio.sleep(1.0)

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        connected_websockets.discard(websocket)
        try:
            await websocket.close()
        except Exception:
            # Already gone (client disconnected first); nothing to clean up. A bare
            # `except:` would also swallow asyncio.CancelledError, which must propagate
            # so the task actually cancels instead of silently continuing.
            pass

@app.post("/network/start")
async def start_network_streaming():
    """Start network audio streaming"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    audio_engine.start_network_streaming()
    return {"message": "Network streaming started"}

@app.post("/network/stop")
async def stop_network_streaming():
    """Stop network audio streaming"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    audio_engine.stop_network_streaming()
    return {"message": "Network streaming stopped"}

@app.get("/network/clients")
async def get_network_clients():
    """Get connected network clients"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    clients = audio_engine.get_network_clients()
    return {"clients": clients, "count": len(clients)}

@app.get("/drivers")
async def get_available_drivers():
    """Get available audio drivers"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    return {
        "available_drivers": audio_engine.get_available_drivers(),
        "current_driver": audio_engine.get_driver_info()
    }

@app.post("/drivers/switch/{driver_type}")
async def switch_driver(driver_type: str):
    """Switch to a different audio driver"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    success = audio_engine.switch_driver(driver_type)
    if success:
        return {"message": f"Switched to {driver_type} driver", "driver_info": audio_engine.get_driver_info()}
    else:
        raise HTTPException(status_code=400, detail=f"Failed to switch to {driver_type} driver")

# Channel Control Endpoints
@app.get("/devices/{device_id}/channels")
async def get_device_channels(device_id: int):
    """Get channel information for a device"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    if hasattr(audio_engine.engine, 'channel_control_manager'):
        info = audio_engine.engine.channel_control_manager.get_device_info(device_id)
        if info:
            return info
    raise HTTPException(status_code=404, detail="Device not found or no channel control")

@app.put("/devices/{device_id}/channels/{channel}/volume")
async def set_channel_volume(device_id: int, channel: int, volume: float):
    """Set volume for specific channel"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    if hasattr(audio_engine.engine, 'channel_control_manager'):
        audio_engine.engine.channel_control_manager.set_device_channel_volume(device_id, channel, volume)
        return {"message": f"Channel {channel} volume set to {volume}"}
    raise HTTPException(status_code=400, detail="Channel control not available")

@app.put("/devices/{device_id}/channels/{channel}/mute")
async def set_channel_mute(device_id: int, channel: int, muted: bool):
    """Mute/unmute specific channel"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    if hasattr(audio_engine.engine, 'channel_control_manager'):
        audio_engine.engine.channel_control_manager.set_device_channel_mute(device_id, channel, muted)
        return {"message": f"Channel {channel} muted: {muted}"}
    raise HTTPException(status_code=400, detail="Channel control not available")

@app.put("/devices/{device_id}/channels/{channel}/solo")
async def set_channel_solo(device_id: int, channel: int, solo: bool):
    """Solo specific channel"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    if hasattr(audio_engine.engine, 'channel_control_manager'):
        audio_engine.engine.channel_control_manager.set_device_channel_solo(device_id, channel, solo)
        return {"message": f"Channel {channel} solo: {solo}"}
    raise HTTPException(status_code=400, detail="Channel control not available")

@app.put("/devices/{device_id}/channels/{channel}/pan")
async def set_channel_pan(device_id: int, channel: int, pan: float):
    """Set pan for specific channel"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    if hasattr(audio_engine.engine, 'channel_control_manager'):
        audio_engine.engine.channel_control_manager.set_device_channel_pan(device_id, channel, pan)
        return {"message": f"Channel {channel} pan set to {pan}"}
    raise HTTPException(status_code=400, detail="Channel control not available")

@app.post("/devices/{device_id}/channels/swap")
async def swap_channels(device_id: int):
    """Swap L/R channels"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    if hasattr(audio_engine.engine, 'channel_control_manager'):
        audio_engine.engine.channel_control_manager.swap_device_channels(device_id)
        return {"message": "Channels swapped"}
    raise HTTPException(status_code=400, detail="Channel control not available")

@app.put("/devices/{device_id}/master/volume")
async def set_device_master_volume(device_id: int, volume: float):
    """Set master volume for device"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    if hasattr(audio_engine.engine, 'channel_control_manager'):
        audio_engine.engine.channel_control_manager.set_device_master_volume(device_id, volume)
        return {"message": f"Master volume set to {volume}"}
    raise HTTPException(status_code=400, detail="Channel control not available")

@app.put("/devices/{device_id}/master/mute")
async def set_device_master_mute(device_id: int, muted: bool):
    """Mute/unmute entire device"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    if hasattr(audio_engine.engine, 'channel_control_manager'):
        audio_engine.engine.channel_control_manager.set_device_master_mute(device_id, muted)
        return {"message": f"Device muted: {muted}"}
    raise HTTPException(status_code=400, detail="Channel control not available")

# Sample Rate Control Endpoints
@app.get("/engine/sample-rate")
async def get_sample_rate():
    """Get current sample rate"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    return {"sample_rate": audio_engine.sample_rate}

@app.put("/engine/sample-rate")
async def set_sample_rate(sample_rate: int):
    """Set sample rate (requires engine restart)"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    if hasattr(audio_engine.engine, 'sample_rate_manager'):
        audio_engine.engine.sample_rate_manager.set_master_sample_rate(sample_rate)
        return {"message": f"Sample rate set to {sample_rate}Hz", "note": "Some devices may require restart"}
    raise HTTPException(status_code=400, detail="Sample rate control not available")

# Network Audio Routing Endpoints
@app.post("/network/connect")
async def connect_to_network(host: str, port: int = 9001):
    """Connect to another ToneSphere instance"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    if hasattr(audio_engine.engine, 'connect_to_network'):
        success = audio_engine.engine.connect_to_network(host, port)
        if success:
            return {"message": f"Connected to {host}:{port}"}
        else:
            raise HTTPException(status_code=400, detail=f"Failed to connect to {host}:{port}")
    raise HTTPException(status_code=400, detail="Network routing not available")

@app.post("/network/disconnect/{conn_id}")
async def disconnect_from_network(conn_id: str):
    """Disconnect from network instance"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    if hasattr(audio_engine.engine, 'disconnect_from_network'):
        audio_engine.engine.disconnect_from_network(conn_id)
        return {"message": f"Disconnected from {conn_id}"}
    raise HTTPException(status_code=400, detail="Network routing not available")

@app.get("/network/connections")
async def get_network_connections():
    """Get all network connections"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    if hasattr(audio_engine.engine, 'get_network_connections'):
        return {
            "incoming": audio_engine.get_network_clients(),
            "outgoing": audio_engine.engine.get_network_connections()
        }
    return {"incoming": audio_engine.get_network_clients(), "outgoing": []}

@app.post("/network/send/{device_id}")
async def send_device_to_network(
    device_id: int, target: str | None = None, transport: str = "tcp"
):
    """
    Send a device's or bus's audio over the network.

    `transport` defaults to "tcp" so a caller written against the old endpoint gets the
    behaviour it already had. TCP send is still unwired and says so; "udp" is the path
    that carries audio.
    """
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    success, message = audio_engine.engine.send_device_audio_to_network(
        device_id, target, transport
    )
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {"message": message, "sends": audio_engine.engine.list_network_sends()}

@app.delete("/network/send/{device_id}")
async def disable_network_send(device_id: int):
    """Stop sending a device's audio, and remove the route it was using"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    if not audio_engine.engine.disable_network_send(device_id):
        raise HTTPException(
            status_code=404, detail=f"Device {device_id} is not sending to the network"
        )
    return {"message": f"Device {device_id} is no longer sending to the network"}

@app.get("/network/sends")
async def list_network_sends():
    """Active network send routes"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    return {"sends": audio_engine.engine.list_network_sends()}

@app.post("/network/receive/{device_id}")
async def register_network_receive(
    device_id: int,
    transport: str = "tcp",
    target_latency_ms: float = 40.0,
    conceal: str = "silence",
):
    """
    Register a bus to receive network audio.

    `target_latency_ms` and `conceal` apply to the UDP jitter buffer only; TCP arrives
    pre-buffered by TCP itself and is written straight through as it always was.
    """
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    success, message = audio_engine.engine.register_network_receive(
        device_id, transport, target_latency_ms, conceal
    )
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {"message": message}

@app.delete("/network/receive/{device_id}")
async def unregister_network_receive(device_id: int):
    """Stop a device receiving network audio, and stop its playout thread"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    removed = audio_engine.engine.unregister_network_receive(device_id)
    return {
        "message": f"Device {device_id} is no longer receiving network audio",
        # The TCP callback is unregistered either way; this says whether a UDP jitter
        # buffer and playout thread were actually torn down, rather than implying both.
        "udp_playout_stopped": removed,
    }

@app.post("/network/udp/start")
async def start_udp_transport(bind_host: str = "127.0.0.1", bind_port: int = 9002):
    """
    Bind the realtime UDP socket.

    Defaults to loopback. Binding every interface is a deliberate act — it triggers a
    Windows firewall prompt — so a caller that wants to be reachable from another machine
    passes that address explicitly.
    """
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    success, message = audio_engine.engine.start_udp_transport(bind_host, bind_port)
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {
        "message": message,
        "transport": audio_engine.engine.udp_transport.statistics(),
    }

@app.post("/network/udp/stop")
async def stop_udp_transport():
    """Close the UDP socket and stop the send worker and every playout thread"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    audio_engine.engine.stop_udp_transport()
    return {"message": "UDP transport stopped"}

@app.post("/network/udp/peer")
async def add_udp_peer(name: str, host: str, port: int = 9002):
    """
    Register where UDP audio should be sent.

    UDP is connectionless, so there is nothing to connect to and nothing to fail here —
    a peer is an address we will send to, and whether anything is listening only becomes
    visible in the peer's own receive statistics.
    """
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    audio_engine.engine.add_udp_peer(name, host, port)
    return {
        "message": f"UDP peer '{name}' is {host}:{port}",
        "peers": audio_engine.engine.get_udp_peers(),
    }

@app.delete("/network/udp/peer/{name}")
async def remove_udp_peer(name: str):
    """Remove a UDP peer"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    if not audio_engine.engine.remove_udp_peer(name):
        raise HTTPException(status_code=404, detail=f"No UDP peer named '{name}'")
    return {"message": f"Removed UDP peer '{name}'"}

@app.get("/network/udp/peers")
async def get_udp_peers():
    """Registered UDP peers"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    return {"peers": audio_engine.engine.get_udp_peers()}

@app.post("/network/quality")
async def set_network_quality(quality: str):
    """Set the quality preset for both transports"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    success, message = audio_engine.engine.set_network_quality(quality)
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {"message": message}

@app.get("/network/statistics")
async def get_network_statistics():
    """
    Network statistics for both transports.

    TCP's counters stay at the top level, where callers have always read them. UDP is
    added under `udp`, with per-route send statistics and per-device jitter-buffer
    statistics — `jitter_buffer` is null until a first packet has actually arrived,
    because before that there is nothing measured about one.
    """
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    return audio_engine.engine.get_network_statistics()

# Virtual Device Management Endpoints
@app.get("/virtual-devices")
async def list_virtual_devices():
    """List all virtual devices"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    return audio_engine.list_virtual_devices()

@app.get("/virtual-devices/counts")
async def get_virtual_device_counts():
    """Get virtual device counts and limits"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    return audio_engine.get_virtual_device_counts()

@app.post("/virtual-devices/input")
async def create_virtual_input(channels: int = 2):
    """Create a new virtual input device"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    device_id = audio_engine.create_virtual_input("Virtual Input", channels)
    if device_id:
        return {"device_id": device_id, "message": "Virtual input created"}
    else:
        raise HTTPException(status_code=400, detail="Failed to create virtual input (limit reached?)")

@app.post("/virtual-devices/output")
async def create_virtual_output(channels: int = 2):
    """Create a new virtual output device"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    device_id = audio_engine.create_virtual_output("Virtual Output", channels)
    if device_id:
        return {"device_id": device_id, "message": "Virtual output created"}
    else:
        raise HTTPException(status_code=400, detail="Failed to create virtual output (limit reached?)")

@app.delete("/virtual-devices/{device_id}")
async def delete_virtual_device(device_id: int):
    """Delete a virtual device"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    success = audio_engine.delete_virtual_device(device_id)
    if success:
        return {"message": f"Virtual device {device_id} deleted"}
    else:
        raise HTTPException(status_code=404, detail="Virtual device not found or cannot be deleted")

@app.put("/virtual-devices/{device_id}/sample-rate")
async def update_virtual_device_sample_rate(device_id: int, sample_rate: int):
    """Update virtual device sample rate"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    if sample_rate < 8000 or sample_rate > 192000:
        raise HTTPException(status_code=400, detail="Sample rate must be between 8000 and 192000")

    success = audio_engine.update_virtual_device_sample_rate(device_id, sample_rate)
    if success:
        return {"message": f"Sample rate updated to {sample_rate}Hz"}
    else:
        raise HTTPException(status_code=404, detail="Virtual device not found")

@app.put("/virtual-devices/{device_id}/channels")
async def update_virtual_device_channels(device_id: int, channels: int):
    """Update virtual device channels"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    if channels < 1 or channels > 32:
        raise HTTPException(status_code=400, detail="Channels must be between 1 and 32")

    success = audio_engine.update_virtual_device_channels(device_id, channels)
    if success:
        return {"message": f"Channels updated to {channels}"}
    else:
        raise HTTPException(status_code=404, detail="Virtual device not found")

# Linux Virtual Sink Endpoints (Track 2)
@app.post("/virtual-devices/system/linux")
async def create_linux_virtual_sink(request: CreateLinuxSinkRequest):
    """
    Create an OS-visible virtual sink on Linux: a PulseAudio/PipeWire null-sink bridged
    into ALSA, so other applications — not just ToneSphere — can select it.

    Refused with a real reason on every other platform, and if pactl exists but no
    Pulse/PipeWire server is running, rather than returning a device id backed by nothing.
    """
    import platform

    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    if platform.system() != "Linux":
        raise HTTPException(
            status_code=400,
            detail=(
                f"Linux virtual sinks need pactl and a running PulseAudio/PipeWire "
                f"server; unavailable on {platform.system()}"
            ),
        )

    device_id = audio_engine.create_linux_system_sink(request.name, request.channels)
    if device_id is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Could not create '{request.name}' as an OS-visible sink — see the "
                f"server log for the real reason (pactl failure, or it loaded but never "
                f"appeared as a PortAudio device)"
            ),
        )

    return {
        "device_id": device_id,
        "message": f"Created Linux virtual sink '{request.name}'",
    }

@app.delete("/virtual-devices/system/{device_id}")
async def remove_linux_virtual_sink(device_id: int):
    """Tear down a Linux virtual sink's routing node, then the OS-level sink itself."""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    if not audio_engine.remove_linux_system_sink(device_id):
        raise HTTPException(
            status_code=404, detail=f"No Linux virtual sink at device {device_id}"
        )
    return {"message": f"Removed Linux virtual sink (device {device_id})"}

# macOS CoreAudio HAL Device Endpoints (Track 3)
@app.get("/virtual-devices/system/macos")
async def get_macos_hal_status():
    """
    What is actually true about the ToneSphere CoreAudio HAL plug-in on this machine.

    Three separate facts, never collapsed into one: whether this OS could have it at all,
    whether the bundle is installed on disk, and whether coreaudiod has actually published
    the device (`device_visible`, which is null rather than false when nothing looked).
    """
    from tonesphere.engine.macos_virtual import plugin_status

    return plugin_status()

@app.post("/virtual-devices/system/macos")
async def attach_macos_hal_device():
    """
    Claim the installed ToneSphere HAL device as an OS-visible ToneSphere endpoint.

    Takes no name or channel count, unlike the Linux sink endpoint: both are compiled into
    the plug-in bundle, and this cannot create a device — the bundle is installed by a
    human with `sudo` (see native/coreaudio-plugin/README.md). Refused with a real reason
    on every other platform, and when the plug-in is not installed or coreaudiod has not
    been restarted since it was, rather than returning a device id backed by nothing.
    """
    from tonesphere.engine.macos_virtual import unavailable_reason

    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    reason = unavailable_reason()
    if reason is not None:
        raise HTTPException(status_code=400, detail=reason)

    device_id = audio_engine.create_macos_system_device()
    if device_id is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "The plug-in is installed but its device is not in PortAudio's device "
                "list — coreaudiod has most likely not been restarted since the install "
                "(sudo launchctl kickstart -k system/com.apple.audio.coreaudiod)"
            ),
        )

    return {
        "device_id": device_id,
        "message": "Attached the macOS HAL device",
    }

@app.delete("/virtual-devices/system/macos/{device_id}")
async def release_macos_hal_device(device_id: int):
    """
    Stop claiming the HAL device. The device itself stays: uninstalling the plug-in needs
    sudo and a coreaudiod restart, so this endpoint honestly does not pretend to do it.
    """
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    if not audio_engine.remove_macos_system_device(device_id):
        raise HTTPException(
            status_code=404, detail=f"No macOS HAL device registered at device {device_id}"
        )
    return {
        "message": (
            f"Released the macOS HAL device (device {device_id}); the plug-in is still "
            f"installed — see native/coreaudio-plugin/README.md to uninstall it"
        )
    }

# Per-Application Capture Endpoints
@app.get("/app-capture")
async def get_app_capture_status():
    """
    What per-application capture can do here, and which applications are playing.

    `process_loopback_supported` is the platform's answer and `process_loopback_implemented`
    is ours; both are reported so a client can tell "this machine cannot" from "ToneSphere
    cannot" instead of being told a flat no.
    """
    from tonesphere.engine.app_capture import capture_status

    status = capture_status()

    if audio_engine:
        status['active_captures'] = audio_engine.engine.process_capture_status()

    return status

@app.post("/app-capture")
async def start_app_capture(request: StartProcessCaptureRequest):
    """
    Capture a process's audio into a new bus, returning the bus id.

    A failure here is a real refusal — a process that has exited, a build of Windows
    without process loopback, an application Windows will not let anyone capture — and is
    returned as an error rather than as a bus id that would only ever carry silence.
    """
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    from tonesphere.engine.process_capture import ProcessCaptureError

    try:
        bus_id = audio_engine.engine.start_process_capture(
            request.pid, request.name, request.include_process_tree)
    except ProcessCaptureError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return {
        "bus_id": bus_id,
        "capture": audio_engine.engine.process_capture_status(bus_id)[0],
    }

@app.get("/app-capture/{bus_id}")
async def get_one_app_capture(bus_id: int):
    """Measured statistics for one running capture."""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    captures = audio_engine.engine.process_capture_status(bus_id)
    if not captures:
        raise HTTPException(status_code=404, detail=f"No capture feeding bus {bus_id}")

    return captures[0]

@app.delete("/app-capture/{bus_id}")
async def stop_app_capture(bus_id: int):
    """Stop a capture and remove the bus it fed."""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")

    if not audio_engine.engine.stop_process_capture(bus_id):
        raise HTTPException(status_code=404, detail=f"No capture feeding bus {bus_id}")

    return {"message": f"Stopped the capture feeding bus {bus_id}"}

# Logging Control Endpoints
@app.post("/logging/enable")
async def enable_file_logging():
    """Enable file logging"""
    from tonesphere.utils.logger import enable_file_logging
    enable_file_logging()
    return {"message": "File logging enabled"}

@app.get("/logging/stats")
async def get_logging_stats():
    """Get logging statistics"""
    from tonesphere.utils.logger import get_log_stats
    return get_log_stats()

def run_api_server():
    """Run the FastAPI server"""
    config = config_manager.load_config()
    api_config = config['api']

    print(f"Starting ToneSphere API server on {api_config['host']}:{api_config['port']}")
    uvicorn.run(
        app,
        host=api_config['host'],
        port=api_config['port'],
        log_level="info"
    )
