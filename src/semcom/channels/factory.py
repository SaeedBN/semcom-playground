from omegaconf import DictConfig
from torch import nn

from semcom.channels.awgn import AWGNChannel
from semcom.channels.rayleigh import RayleighChannel
from semcom.channels.rician import RicianChannel


def create_channel(cfg: DictConfig) -> nn.Module:
    """Creating channel model from the configuration file"""
    if cfg.name == "awgn":
        return AWGNChannel(snr_db=cfg.snr_db)

    if cfg.name == "rayleigh":
        return RayleighChannel(snr_db=cfg.snr_db)

    if cfg.name == "rician":
        return RicianChannel(
            snr_db=cfg.snr_db,
            k_factor=float(cfg.k_factor),
        )

    raise ValueError(f"Channel Model {cfg.name} is not supported")
