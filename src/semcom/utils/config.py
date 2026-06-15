from pathlib import Path

from hydra import compose, initialize_config_dir  
from omegaconf import DictConfig  


def load_config(config_name: str = "default") -> DictConfig:
    config_dir = Path(__file__).resolve().parents[3] / "configs"

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=config_name)
    
    return cfg