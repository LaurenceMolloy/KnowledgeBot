from pathlib import Path
from typing import Optional, Dict
import os
from dataclasses import dataclass
from functools import lru_cache

@dataclass
class ConfigValue:
    """Container for a config value with source information"""
    name: str          # The name of the config value (e.g., "SLACK_TOKEN")
    value: str         # The actual config value  
    source: str        # Where it came from: 'docker', 'env', 'dotenv', 'default'

class ConfigResolver:
    """Elegant config management with priority support"""
    
    def __init__(self, secrets_dir: Path = Path("/run/secrets")):
        self.secrets_dir = secrets_dir
    
    @lru_cache(maxsize=32)
    def get(self, name: str, default: Optional[str] = None) -> ConfigValue:
        """Get a config value with source information"""
        
        # Try Docker secrets first
        docker_secret = self._get_docker_secret(name)
        if docker_secret is not None:
            return ConfigValue(name=name, value=docker_secret, source="docker")
        
        # Try environment variable
        env_value = os.environ.get(name)
        if env_value is not None:
            return ConfigValue(name=name, value=env_value, source=".env")
        
        # Fallback to default
        if default is not None:
            return ConfigValue(name=name, value=default, source="default")
        
        raise ValueError(f"Config value {name!r} not set and no default provided")
    
    def _get_docker_secret(self, name: str) -> Optional[str]:
        """Read secret from Docker secrets directory"""
        secret_file = self.secrets_dir / name
        try:
            if secret_file.exists():
                return secret_file.read_text().strip()
        except OSError:
            pass
        return None
    
    def __getitem__(self, name: str) -> str:
        """Convenience method to get config value value directly"""
        return self.get(name).value