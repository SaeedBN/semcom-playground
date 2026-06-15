from pathlib import Path

from omegaconf import DictConfig, OmegaConf

def ensure_dict(path: str | Path) -> Path:
    """ Creating a new directory (if necessary) """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    return path

def save_config(cfg: DictConfig, output_dir: str | Path) -> None:
    """ Saving whole config into output directory """

    output_dir = ensure_dict(output_dir)
    OmegaConf.save(cfg, output_dir / "compiled_config.yaml")