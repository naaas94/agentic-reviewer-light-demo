"""Configuration loader for unified settings management."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from core.logging_config import get_logger

logger = get_logger(__name__)


class ConfigLoader:
    """Loads and manages configuration from YAML files."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize config loader.

        Args:
            config_path: Path to config YAML file. If None, uses default location.
        """
        if config_path is None:
            # Default: configs/config.yaml relative to project root
            project_root = Path(__file__).parent.parent
            config_path_obj = project_root / "configs" / "config.yaml"
        else:
            config_path_obj = Path(config_path)

        self.config_path = config_path_obj
        self._config: Optional[Dict[str, Any]] = None
        self._load_config()

    def _load_config(self):
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}, using defaults")
            self._config = self._get_default_config()
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f) or {}
            logger.debug(f"Loaded config from {self.config_path}")
        except Exception as e:
            logger.warning(f"Failed to load config from {self.config_path}: {e}, using defaults")
            self._config = self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration if file is missing."""
        return {
            "model": {
                "default": "mistral",
                "ollama_url": "http://localhost:11434",
                "preferences": [
                    ["mistral", 1, "Fast, good instruction following"],
                    ["llama3.2", 2, "Latest Llama, excellent quality"],
                ],
            },
            "performance": {
                "max_concurrent": 1,
                "max_retries": 3,
                "timeout": 180,
                "base_delay": 1.0,
                "num_predict": 200,
                "temperature": 0.1,
            },
            "cache": {
                "enable": True,
                "persistent": True,
                "cache_dir": ".cache",
                "ttl_hours": 168,
            },
            "demo": {
                "default_samples": 12,
                "warmup": True,
                "use_compact_prompt": False,
                "strict_validation": True,
            },
            "presets": {
                "demo_fast": {
                    "max_concurrent": 1,
                    "num_predict": 150,
                    "timeout": 120,
                    "samples": 8,
                    "temperature": 0.1,
                },
                "benchmark": {
                    "max_concurrent": 1,
                    "num_predict": 200,
                    "timeout": 180,
                    "samples": 12,
                    "temperature": 0.0,
                },
            },
        }

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get config value by dot-separated path.

        Args:
            key_path: Dot-separated path (e.g., "model.default", "performance.timeout")
            default: Default value if key not found

        Returns:
            Config value or default
        """
        if self._config is None:
            return default

        keys = key_path.split(".")
        value = self._config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def get_model_default(self) -> str:
        """Get default model name."""
        return str(self.get("model.default", "mistral"))

    def get_ollama_url(self) -> str:
        """Get Ollama URL."""
        return str(self.get("model.ollama_url", "http://localhost:11434"))

    def get_model_preferences(self) -> List[Tuple[str, int, str]]:
        """Get model preferences list.

        Returns:
            List of (keyword, priority, description) tuples
        """
        prefs = self.get("model.preferences", [])
        # Convert list format to tuples
        return [tuple(p) if isinstance(p, list) else p for p in prefs]

    def get_performance_config(self) -> Dict[str, Any]:
        """Get all performance settings."""
        result = self.get("performance", {})
        return result if isinstance(result, dict) else {}

    def get_cache_config(self) -> Dict[str, Any]:
        """Get all cache settings."""
        result = self.get("cache", {})
        return result if isinstance(result, dict) else {}

    def get_demo_config(self) -> Dict[str, Any]:
        """Get all demo settings."""
        result = self.get("demo", {})
        return result if isinstance(result, dict) else {}

    def get_preset(self, preset_name: str) -> Dict[str, Any]:
        """Get preset configuration.

        Args:
            preset_name: Name of preset (e.g., "demo_fast", "benchmark")

        Returns:
            Preset configuration dict
        """
        result = self.get(f"presets.{preset_name}", {})
        return result if isinstance(result, dict) else {}


# Global config instance (lazy-loaded)
_config_instance: Optional[ConfigLoader] = None


def get_config(config_path: Optional[str] = None) -> ConfigLoader:
    """Get global config instance (singleton pattern).

    Args:
        config_path: Optional path to config file (only used on first call)

    Returns:
        ConfigLoader instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigLoader(config_path)
    return _config_instance


def reload_config(config_path: Optional[str] = None):
    """Reload configuration (useful for testing or dynamic updates)."""
    global _config_instance
    _config_instance = ConfigLoader(config_path)

