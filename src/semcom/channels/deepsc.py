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


class DeepSCRayleighChannel(nn.Module):
    def __init__(self, noise_std: float) -> None:
        super().__init__()
        self.noise_std = float(noise_std)

    def forward(self, transmitted_symbols: torch.Tensor) -> torch.Tensor:
        fading = _rayleigh_fading_like(transmitted_symbols)
        noise = torch.randn_like(transmitted_symbols) * self.noise_std
        return fading * transmitted_symbols + noise


class DeepSCRicianChannel(nn.Module):
    def __init__(
        self,
        noise_std: float,
        k_factor: float = 1.0,
    ) -> None:
        super().__init__()

        if k_factor < 0:
            raise ValueError("k_factor must be non-negative.")

        self.noise_std = float(noise_std)
        self.k_factor = float(k_factor)

    def forward(self, transmitted_symbols: torch.Tensor) -> torch.Tensor:
        fading = _rician_fading_like(
            transmitted_symbols,
            k_factor=self.k_factor,
        )
        noise = torch.randn_like(transmitted_symbols) * self.noise_std
        return fading * transmitted_symbols + noise


def create_deepsc_awgn_channel_from_snr(snr_db: float) -> DeepSCAWGNChannel:
    return DeepSCAWGNChannel(noise_std=snr_db_to_noise_std(snr_db))


def create_deepsc_rayleigh_channel_from_snr(snr_db: float) -> DeepSCRayleighChannel:
    return DeepSCRayleighChannel(noise_std=snr_db_to_noise_std(snr_db))


def create_deepsc_rician_channel_from_snr(
    snr_db: float,
    k_factor: float = 1.0,
) -> DeepSCRicianChannel:
    return DeepSCRicianChannel(
        noise_std=snr_db_to_noise_std(snr_db),
        k_factor=k_factor,
    )


def create_deepsc_channel_from_snr(
    channel_name: str,
    snr_db: float,
    rician_k_factor: float = 1.0,
) -> nn.Module:
    if channel_name == "awgn":
        return create_deepsc_awgn_channel_from_snr(snr_db)

    if channel_name == "rayleigh":
        return create_deepsc_rayleigh_channel_from_snr(snr_db)

    if channel_name == "rician":
        return create_deepsc_rician_channel_from_snr(
            snr_db=snr_db,
            k_factor=rician_k_factor,
        )

    raise ValueError(f"DeepSC channel {channel_name} is not supported.")


def _rayleigh_fading_like(x: torch.Tensor) -> torch.Tensor:
    real = torch.randn_like(x) / math.sqrt(2.0)
    imag = torch.randn_like(x) / math.sqrt(2.0)
    return torch.sqrt(real * real + imag * imag)


def _rician_fading_like(
    x: torch.Tensor,
    k_factor: float,
) -> torch.Tensor:
    los_scale = math.sqrt(k_factor / (k_factor + 1.0))
    scatter_scale = math.sqrt(1.0 / (2.0 * (k_factor + 1.0)))

    real = los_scale + scatter_scale * torch.randn_like(x)
    imag = scatter_scale * torch.randn_like(x)

    return torch.sqrt(real * real + imag * imag)
