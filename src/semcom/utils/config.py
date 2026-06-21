from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


def load_config(config_name: str = "default") -> DictConfig:
    config_dir = get_project_root_dir() / "configs"

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=config_name)

    return cfg


def load_config_from_path(config_path: str | Path) -> DictConfig:
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"This configuration does not exits.\n Given path: {config_path}"
        )

    cfg = OmegaConf.load(config_path)

    if not isinstance(cfg, DictConfig):
        raise TypeError(
            f"Expected a DictConfig at {config_path}, got {type(cfg).__name__}"
        )

    return cfg


def get_project_root_dir() -> Path:
    return Path(__file__).resolve().parents[3]
