"""
Sample Rate Converter
Handles sample rate conversion between devices
"""

import numpy as np
from typing import Optional
from tonesphere.utils.logger import logger


class SampleRateConverter:
    """
    Sample rate converter using linear interpolation
    Supports real-time sample rate conversion
    """
    
    def __init__(self, input_rate: int, output_rate: int, channels: int):
        self.input_rate = input_rate
        self.output_rate = output_rate
        self.channels = channels
        self.ratio = output_rate / input_rate
        
        # Buffer for leftover samples
        self.buffer = np.zeros((1, channels), dtype=np.float32)
        self.buffer_position = 0.0
        
        logger.info(f"Sample rate converter: {input_rate}Hz -> {output_rate}Hz (ratio: {self.ratio:.4f})")
    
    def convert(self, input_data: np.ndarray) -> np.ndarray:
        """
        Convert sample rate of input audio
        
        Args:
            input_data: Input audio (frames, channels)
            
        Returns:
            Resampled audio
        """
        if self.input_rate == self.output_rate:
            return input_data
        
        input_frames = input_data.shape[0]
        output_frames = int(input_frames * self.ratio)
        
        # Create output array
        output_data = np.zeros((output_frames, self.channels), dtype=np.float32)
        
        # Linear interpolation
        for out_idx in range(output_frames):
            # Calculate input position
            in_pos = out_idx / self.ratio
            in_idx = int(in_pos)
            frac = in_pos - in_idx
            
            if in_idx < input_frames - 1:
                # Interpolate between samples
                output_data[out_idx] = (
                    input_data[in_idx] * (1.0 - frac) +
                    input_data[in_idx + 1] * frac
                )
            elif in_idx < input_frames:
                # Last sample
                output_data[out_idx] = input_data[in_idx]
        
        return output_data
    
    def convert_sinc(self, input_data: np.ndarray, quality: int = 4) -> np.ndarray:
        """
        High-quality sample rate conversion using sinc interpolation
        
        Args:
            input_data: Input audio (frames, channels)
            quality: Quality factor (higher = better quality, slower)
            
        Returns:
            Resampled audio
        """
        if self.input_rate == self.output_rate:
            return input_data
        
        input_frames = input_data.shape[0]
        output_frames = int(input_frames * self.ratio)
        
        # Create output array
        output_data = np.zeros((output_frames, self.channels), dtype=np.float32)
        
        # Sinc interpolation
        for out_idx in range(output_frames):
            in_pos = out_idx / self.ratio
            
            for ch in range(self.channels):
                sample_sum = 0.0
                
                for i in range(-quality, quality + 1):
                    in_idx = int(in_pos) + i
                    
                    if 0 <= in_idx < input_frames:
                        x = in_pos - in_idx
                        
                        # Sinc function
                        if abs(x) < 1e-6:
                            sinc_val = 1.0
                        else:
                            sinc_val = np.sin(np.pi * x) / (np.pi * x)
                        
                        # Hann window
                        window = 0.5 * (1.0 + np.cos(np.pi * x / quality))
                        
                        sample_sum += input_data[in_idx, ch] * sinc_val * window
                
                output_data[out_idx, ch] = sample_sum
        
        return output_data
    
    def update_rates(self, input_rate: int, output_rate: int):
        """Update sample rates"""
        self.input_rate = input_rate
        self.output_rate = output_rate
        self.ratio = output_rate / input_rate
        logger.info(f"Updated sample rate converter: {input_rate}Hz -> {output_rate}Hz")


class SampleRateManager:
    """
    Manages sample rate conversion for multiple devices
    """
    
    def __init__(self, master_sample_rate: int = 48000):
        self.master_sample_rate = master_sample_rate
        self.converters: dict[tuple[int, int, int], SampleRateConverter] = {}
    
    def get_converter(self, input_rate: int, output_rate: int, channels: int) -> SampleRateConverter:
        """Get or create a sample rate converter"""
        key = (input_rate, output_rate, channels)
        
        if key not in self.converters:
            self.converters[key] = SampleRateConverter(input_rate, output_rate, channels)
        
        return self.converters[key]
    
    def convert(self, audio_data: np.ndarray, input_rate: int, 
                output_rate: int, quality: str = 'linear') -> np.ndarray:
        """
        Convert audio sample rate
        
        Args:
            audio_data: Input audio
            input_rate: Input sample rate
            output_rate: Output sample rate
            quality: 'linear' or 'sinc'
            
        Returns:
            Converted audio
        """
        if input_rate == output_rate:
            return audio_data
        
        channels = audio_data.shape[1] if len(audio_data.shape) > 1 else 1
        converter = self.get_converter(input_rate, output_rate, channels)
        
        if quality == 'sinc':
            return converter.convert_sinc(audio_data)
        else:
            return converter.convert(audio_data)
    
    def convert_to_master(self, audio_data: np.ndarray, input_rate: int) -> np.ndarray:
        """Convert audio to master sample rate"""
        return self.convert(audio_data, input_rate, self.master_sample_rate)
    
    def convert_from_master(self, audio_data: np.ndarray, output_rate: int) -> np.ndarray:
        """Convert audio from master sample rate"""
        return self.convert(audio_data, self.master_sample_rate, output_rate)
    
    def set_master_sample_rate(self, sample_rate: int):
        """Set master sample rate"""
        self.master_sample_rate = sample_rate
        # Clear converter cache
        self.converters.clear()
        logger.info(f"Master sample rate set to {sample_rate}Hz")
    
    def clear_cache(self):
        """Clear converter cache"""
        self.converters.clear()
        logger.info("Sample rate converter cache cleared")
