from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_yaml_config(filename: str) -> dict:
    """Load a YAML config file from the configs directory."""
    config_path = CONFIG_DIR / filename
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
