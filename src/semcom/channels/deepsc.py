import math

import torch
from torch import nn


def snr_db_to_noise_std(snr_db: float) -> float:
    snr_linear = 10 ** (float(snr_db) / 10.0)
    return 1.0 / math.sqrt(2.0 * snr_linear)


def power_normalize_deepsc(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    power = torch.sqrt(torch.mean(x * x))

    if bool(power > 1.0):
        return x / (power + eps)

    return x


class DeepSCAWGNChannel(nn.Module):
    def __init__(self, noise_std: float) -> None:
        super().__init__()
        self.noise_std = float(noise_std)

    def forward(self, transmitted_symbols: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(transmitted_symbols) * self.noise_std
        return transmitted_symbols + noise


def create_deepsc_awgn_channel_from_snr(snr_db: float) -> DeepSCAWGNChannel:
    return DeepSCAWGNChannel(noise_std=snr_db_to_noise_std(snr_db))
