from omegaconf import DictConfig
from torch import nn

from semcom.channels.awgn import AWGNChannel


def create_channel(cfg: DictConfig) -> nn.Module:
    """Creating channel model from the configuration file"""
    if cfg.name == "awgn":
        return AWGNChannel(snr_db=cfg.snr_db)

    raise ValueError(f"Channel Model {cfg.name} is not supported")
