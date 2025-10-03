from typing import Dict, Tuple, List
from .models import RoutingConnection, AudioChannel, RoutingState
from tonesphere.utils.logger import logger

class AudioRoutingMatrix:
    """Manages audio routing between inputs and outputs"""
    
    def __init__(self):
        self.connections: Dict[Tuple[int, int], RoutingConnection] = {}
        self.channels: Dict[int, AudioChannel] = {}
        self.solo_active = False
        
    def add_channel(self, channel: AudioChannel):
        """Add an audio channel"""
        self.channels[channel.id] = channel
        
    def create_routing(self, source_id: int, destination_id: int, volume: float = 1.0) -> tuple[bool, str]:
        """Create a routing connection"""
        key = (source_id, destination_id)
        if key not in self.connections:
            connection = RoutingConnection(
                source_id=source_id,
                destination_id=destination_id,
                state=RoutingState.ENABLED,
                volume=volume
            )
            self.connections[key] = connection
            logger.info(f"Created routing: {source_id} -> {destination_id}")
            return True, "✅ Routing created successfully"
        return False, "⚠️ Routing already exists, skipping command"
    
    def remove_routing(self, source_id: int, destination_id: int) -> bool:
        """Remove a routing connection"""
        key = (source_id, destination_id)
        if key in self.connections:
            del self.connections[key]
            logger.info(f"Removed routing: {source_id} -> {destination_id}")
            return True
        return False
    
    def set_routing_volume(self, source_id: int, destination_id: int, volume: float):
        """Set volume for a routing connection"""
        key = (source_id, destination_id)
        if key in self.connections:
            self.connections[key].volume = max(0.0, min(2.0, volume))
    
    def toggle_mute(self, source_id: int, destination_id: int):
        """Toggle mute for a routing connection"""
        key = (source_id, destination_id)
        if key in self.connections:
            self.connections[key].muted = not self.connections[key].muted
    
    def set_solo(self, channel_id: int, solo: bool):
        """Set solo state for a channel"""
        if channel_id in self.channels:
            self.channels[channel_id].solo = solo
            self.solo_active = any(ch.solo for ch in self.channels.values())
    
    def get_active_routings_for_output(self, output_id: int) -> List[RoutingConnection]:
        """Get all active routings for an output"""
        active = []
        for connection in self.connections.values():
            if connection.destination_id == output_id and connection.state == RoutingState.ENABLED:
                if not connection.muted:
                    # Check solo logic
                    if self.solo_active:
                        source_channel = self.channels.get(connection.source_id)
                        if source_channel and source_channel.solo:
                            active.append(connection)
                    else:
                        active.append(connection)
        return active