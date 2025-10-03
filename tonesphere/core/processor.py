from .models import AudioEffect, EffectType
import numpy as np
from typing import Dict, List

class AudioProcessor:
    """Core audio processing engine with real-time effects"""
    
    def __init__(self, sample_rate: int = 48000, buffer_size: int = 128):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.effects_chain = {}
        
    def apply_eq(self, audio_data: np.ndarray, params: Dict) -> np.ndarray:
        """Apply parametric EQ to audio data"""
        # Simplified EQ implementation - in production, use proper filters
        low_gain = params.get('low_gain', 0.0)
        mid_gain = params.get('mid_gain', 0.0)
        high_gain = params.get('high_gain', 0.0)
        
        # Apply basic gain adjustments (simplified)
        processed = audio_data.copy()
        if abs(low_gain) > 0.01:
            processed *= (1.0 + low_gain * 0.1)
        
        return np.clip(processed, -1.0, 1.0)
    
    def apply_compressor(self, audio_data: np.ndarray, params: Dict) -> np.ndarray:
        """Apply dynamic range compression"""
        threshold = params.get('threshold', -20.0)  # dB
        ratio = params.get('ratio', 4.0)
        attack = params.get('attack', 0.003)  # seconds
        release = params.get('release', 0.1)  # seconds
        
        # Simplified compressor - convert to dB, apply compression
        db_audio = 20 * np.log10(np.abs(audio_data) + 1e-10)
        compressed = np.where(
            db_audio > threshold,
            threshold + (db_audio - threshold) / ratio,
            db_audio
        )
        
        # Convert back to linear
        return np.sign(audio_data) * np.power(10, compressed / 20)
    
    def apply_reverb(self, audio_data: np.ndarray, params: Dict) -> np.ndarray:
        """Apply simple reverb effect"""
        room_size = params.get('room_size', 0.5)
        damping = params.get('damping', 0.5)
        wet_level = params.get('wet_level', 0.3)
        
        # Simplified reverb using delay lines
        delay_samples = int(room_size * self.sample_rate * 0.05)  # Max 50ms
        if delay_samples > 0 and delay_samples < len(audio_data):
            delayed = np.roll(audio_data, delay_samples) * damping
            return audio_data + delayed * wet_level
        
        return audio_data
    
    def process_effects_chain(self, audio_data: np.ndarray, effects: List[AudioEffect]) -> np.ndarray:
        """Process audio through effects chain"""
        processed = audio_data.copy()
        
        for effect in effects:
            if not effect.enabled:
                continue
                
            if effect.effect_type == EffectType.EQ:
                processed = self.apply_eq(processed, effect.parameters)
            elif effect.effect_type == EffectType.COMPRESSOR:
                processed = self.apply_compressor(processed, effect.parameters)
            elif effect.effect_type == EffectType.REVERB:
                processed = self.apply_reverb(processed, effect.parameters)
            # Add more effects as needed
            
        return processed