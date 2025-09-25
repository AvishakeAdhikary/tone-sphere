# ToneSphere API Client
# Example client for communicating with the audio engine backend

import requests
import json
import websockets
import asyncio
import threading
from typing import Dict, List, Optional, Callable

class PyAudioEngineClient:
    """Client for communicating with ToneSphere API"""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.session = requests.Session()
        self.websocket = None
        self.event_callbacks = {}
    
    # Device Management API
    def get_devices(self) -> List[Dict]:
        """Get all available audio devices"""
        response = self.session.get(f"{self.base_url}/devices")
        response.raise_for_status()
        return response.json()
    
    def create_virtual_device(self, name: str, device_type: str, channels: int = 2) -> int:
        """Create a virtual audio device"""
        data = {"name": name, "device_type": device_type, "channels": channels}
        response = self.session.post(f"{self.base_url}/devices/virtual", json=data)
        response.raise_for_status()
        return response.json()["device_id"]
    
    # Routing Management API
    def create_routing(self, source_id: int, destination_id: int, volume: float = 1.0):
        """Create a routing connection"""
        data = {"source_id": source_id, "destination_id": destination_id, "volume": volume}
        response = self.session.post(f"{self.base_url}/routing", json=data)
        response.raise_for_status()
        return response.json()
    
    def remove_routing(self, source_id: int, destination_id: int):
        """Remove a routing connection"""
        response = self.session.delete(f"{self.base_url}/routing/{source_id}/{destination_id}")
        response.raise_for_status()
        return response.json()
    
    def set_routing_volume(self, source_id: int, destination_id: int, volume: float):
        """Set volume for a routing connection"""
        data = {"source_id": source_id, "destination_id": destination_id, "volume": volume}
        response = self.session.put(f"{self.base_url}/routing/volume", json=data)
        response.raise_for_status()
        return response.json()
    
    def get_routing_matrix(self) -> Dict:
        """Get current routing matrix"""
        response = self.session.get(f"{self.base_url}/routing")
        response.raise_for_status()
        return response.json()
    
    # Engine Control API
    def start_engine(self):
        """Start the audio engine"""
        response = self.session.post(f"{self.base_url}/engine/start")
        response.raise_for_status()
        return response.json()
    
    def stop_engine(self):
        """Stop the audio engine"""
        response = self.session.post(f"{self.base_url}/engine/stop")
        response.raise_for_status()
        return response.json()
    
    def get_engine_status(self) -> Dict:
        """Get engine status"""
        response = self.session.get(f"{self.base_url}/engine/status")
        response.raise_for_status()
        return response.json()
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        response = self.session.get(f"{self.base_url}/performance")
        response.raise_for_status()
        return response.json()
    
    # WebSocket Event Handling
    async def connect_websocket(self):
        """Connect to WebSocket for real-time events"""
        ws_url = self.base_url.replace("http", "ws") + "/ws/events"
        try:
            self.websocket = await websockets.connect(ws_url)
            # Start message handling in background
            asyncio.create_task(self._handle_websocket_messages())
        except Exception as e:
            print(f"Failed to connect WebSocket: {e}")
    
    def register_event_callback(self, event_type: str, callback: Callable):
        """Register callback for specific event type"""
        self.event_callbacks[event_type] = callback
    
    async def _handle_websocket_messages(self):
        """Handle incoming WebSocket messages"""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    event_type = data.get("type")
                    if event_type in self.event_callbacks:
                        callback = self.event_callbacks[event_type]
                        if asyncio.iscoroutinefunction(callback):
                            await callback(data.get("data"))
                        else:
                            callback(data.get("data"))
                except Exception as e:
                    print(f"WebSocket message error: {e}")
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket connection closed")
        except Exception as e:
            print(f"WebSocket error: {e}")
    
    def _on_websocket_error(self, ws, error):
        """Handle WebSocket error"""
        print(f"WebSocket error: {error}")
    
    def _on_websocket_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close"""
        print("WebSocket connection closed")

# Usage Example
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Initialize client
        client = PyAudioEngineClient()
        
        # Check engine status
        status = client.get_engine_status()
        print(f"Engine status: {status}")
        
        # Get available devices
        devices = client.get_devices()
        print(f"Found {len(devices)} devices")
        
        # Create a virtual input device
        virtual_input_id = client.create_virtual_device("My Mic Input", "input", 2)
        print(f"Created virtual input with ID: {virtual_input_id}")
        
        # Create a virtual output device
        virtual_output_id = client.create_virtual_device("My Speakers", "output", 2)
        print(f"Created virtual output with ID: {virtual_output_id}")
        
        # Create routing connection
        client.create_routing(virtual_input_id, virtual_output_id, 0.8)
        print("Created routing connection")
        
        # Set up real-time performance monitoring
        def on_performance_update(data):
            print(f"CPU: {data['cpu_usage']:.1f}%, Latency: {data['latency_ms']:.1f}ms")
        
        client.register_event_callback("performance_stats", on_performance_update)
        
        # Connect WebSocket and keep running
        await client.connect_websocket()
        
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("Shutting down...")
            if client.websocket:
                await client.websocket.close()
    
    # Run the async main function
    asyncio.run(main())
