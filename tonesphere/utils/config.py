from tonesphere.utils.logger import logger
from pathlib import Path
from typing import Dict
import yaml

class ConfigManager:
    """Manages configuration and presets"""
    
    def __init__(self, config_path: str = "audio_engine_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_default_config()
        
    def _load_default_config(self) -> Dict:
        """Load default configuration"""
        return {
            'engine': {
                'sample_rate': 48000,
                'buffer_size': 128,
                'master_volume': 1.0
            },
            'virtual_devices': {
                'default_inputs': 3,
                'default_outputs': 3,
                'default_channels': 2
            },
            'effects': {
                'enabled': True,
                'presets': {}
            },
            'api': {
                'host': '127.0.0.1',
                'port': 8080,
                'cors_enabled': True
            }
        }
    
    def load_config(self) -> Dict:
        """Load configuration from file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    # Merge with defaults
                    self.config.update(loaded_config)
            return self.config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self.config
    
    def save_config(self, config: Dict = None):
        """Save configuration to file"""
        try:
            config_to_save = config or self.config
            with open(self.config_path, 'w') as f:
                yaml.safe_dump(config_to_save, f, default_flow_style=False)
            logger.info("Configuration saved")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")