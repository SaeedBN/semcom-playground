import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def ensure_dir(path: str | Path) -> Path:
    """Creating a new directory (if necessary)"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    return path


def save_config(cfg: DictConfig, output_dir: str | Path) -> None:
    """Saving whole config into output directory"""

    output_dir = ensure_dir(output_dir)
    OmegaConf.save(cfg, output_dir / "compiled_config.yaml")


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Saving the data dictioanry into a json file"""
    path = Path(path)
    ensure_dir(path.parent)

    if path.exists() and path.is_dir():
        if any(path.iterdir()):
            raise IsADirectoryError(f"Cannot write JSON file to directory path: {path}")

        path.rmdir()

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
