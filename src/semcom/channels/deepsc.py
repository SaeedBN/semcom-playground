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
        original_shape = transmitted_symbols.shape
        symbols_per_batch = transmitted_symbols[0].numel()

        if symbols_per_batch % 2 != 0:
            raise ValueError(
                "DeepSC Rayleigh channel expects an even number of transmitted "
                "features per batch item."
            )

        real = torch.randn(
            (),
            device=transmitted_symbols.device,
            dtype=transmitted_symbols.dtype,
        ) / math.sqrt(2.0)
        imag = torch.randn(
            (),
            device=transmitted_symbols.device,
            dtype=transmitted_symbols.dtype,
        ) / math.sqrt(2.0)
        fading = torch.stack(
            (
                torch.stack((real, -imag)),
                torch.stack((imag, real)),
            )
        )

        paired_symbols = transmitted_symbols.reshape(original_shape[0], -1, 2)
        faded_symbols = torch.matmul(paired_symbols, fading)
        noise = torch.randn_like(faded_symbols) * self.noise_std
        received_symbols = faded_symbols + noise
        equalized_symbols = torch.matmul(received_symbols, torch.inverse(fading))

        return equalized_symbols.reshape(original_shape)


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
        original_shape = transmitted_symbols.shape
        symbols_per_batch = transmitted_symbols[0].numel()

        if symbols_per_batch % 2 != 0:
            raise ValueError(
                "DeepSC Rician channel expects an even number of transmitted "
                "features per batch item."
            )

        los_scale = math.sqrt(self.k_factor / (self.k_factor + 1.0))
        scatter_scale = math.sqrt(1.0 / (2.0 * (self.k_factor + 1.0)))
        real = los_scale + scatter_scale * torch.randn(
            (),
            device=transmitted_symbols.device,
            dtype=transmitted_symbols.dtype,
        )
        imag = scatter_scale * torch.randn(
            (),
            device=transmitted_symbols.device,
            dtype=transmitted_symbols.dtype,
        )
        fading = torch.stack(
            (
                torch.stack((real, -imag)),
                torch.stack((imag, real)),
            )
        )

        paired_symbols = transmitted_symbols.reshape(original_shape[0], -1, 2)
        faded_symbols = torch.matmul(paired_symbols, fading)
        noise = torch.randn_like(faded_symbols) * self.noise_std
        received_symbols = faded_symbols + noise
        equalized_symbols = torch.matmul(received_symbols, torch.inverse(fading))

        return equalized_symbols.reshape(original_shape)


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
