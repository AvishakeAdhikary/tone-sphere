from tonesphere.core.engine import AudioEngine
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
audio_engine: Optional[AudioEngine] = None
config_manager = ConfigManager()
connected_websockets: set = set()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan handler"""
    global audio_engine
    
    # Startup
    config = config_manager.load_config()
    audio_engine = AudioEngine(
        sample_rate=config['engine']['sample_rate'],
        buffer_size=config['engine']['buffer_size']
    )
    
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
    
    return {
        "status": "running" if audio_engine.is_running else "stopped",
        "sample_rate": audio_engine.sample_rate,
        "buffer_size": audio_engine.buffer_size,
        "master_volume": audio_engine.master_volume
    }

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