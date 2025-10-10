from tonesphere.core.engine_factory import UnifiedAudioEngine
from tonesphere.utils.logger import logger
from tonesphere.utils.config import ConfigManager
from tonesphere.api.models import DeviceInfo, CreateVirtualDeviceRequest, CreateRoutingRequest, SetVolumeRequest, PerformanceStats
from typing import List, Optional
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
import uvicorn

# Global audio engine instance
audio_engine: Optional[UnifiedAudioEngine] = None
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

@app.get("/devices", response_model=List[DeviceInfo])
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
        raise HTTPException(status_code=500, detail=f"Failed to start engine: {str(e)}")

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
        except:
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
async def send_device_to_network(device_id: int, target: Optional[str] = None):
    """Send device audio over network"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")
    
    if hasattr(audio_engine.engine, 'send_device_audio_to_network'):
        audio_engine.engine.send_device_audio_to_network(device_id, target)
        return {"message": f"Sending device {device_id} audio to network"}
    raise HTTPException(status_code=400, detail="Network routing not available")

@app.post("/network/receive/{device_id}")
async def register_network_receive(device_id: int):
    """Register device to receive network audio"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")
    
    if hasattr(audio_engine.engine, 'register_network_receive'):
        audio_engine.engine.register_network_receive(device_id)
        return {"message": f"Device {device_id} registered for network receive"}
    raise HTTPException(status_code=400, detail="Network routing not available")

@app.get("/network/statistics")
async def get_network_statistics():
    """Get network statistics"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")
    
    if hasattr(audio_engine.engine, 'get_network_statistics'):
        return audio_engine.engine.get_network_statistics()
    return {"error": "Network statistics not available"}

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
    
    device_id = audio_engine.create_virtual_input(f"Virtual Input", channels)
    if device_id:
        return {"device_id": device_id, "message": "Virtual input created"}
    else:
        raise HTTPException(status_code=400, detail="Failed to create virtual input (limit reached?)")

@app.post("/virtual-devices/output")
async def create_virtual_output(channels: int = 2):
    """Create a new virtual output device"""
    if not audio_engine:
        raise HTTPException(status_code=500, detail="Audio engine not initialized")
    
    device_id = audio_engine.create_virtual_output(f"Virtual Output", channels)
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