import math

import torch
from torch import nn


class RicianChannel(nn.Module):
    def __init__(
        self,
        snr_db: float = 10,
        k_factor: float = 1.0,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()

        if k_factor < 0:
            raise ValueError("k_factor must be non-negative.")

        self.snr_db = float(snr_db)
        self.k_factor = float(k_factor)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fading = self._sample_fading(x)
        noise = self._sample_noise(x)

        return fading * x + noise

    def _sample_fading(self, x: torch.Tensor) -> torch.Tensor:
        los_scale = math.sqrt(self.k_factor / (self.k_factor + 1.0))
        scatter_scale = math.sqrt(1.0 / (2.0 * (self.k_factor + 1.0)))

        real = los_scale + scatter_scale * torch.randn_like(x.real)
        imag = scatter_scale * torch.randn_like(x.real)

        if torch.is_complex(x):
            return torch.complex(real, imag)

        return torch.sqrt(real * real + imag * imag)

    def _sample_noise(self, x: torch.Tensor) -> torch.Tensor:
        noise_power = self._calculate_noise_power(x)

        if torch.is_complex(x):
            noise_std = torch.sqrt(noise_power / 2.0)
            return noise_std * (
                torch.randn_like(x.real) + 1j * torch.randn_like(x.imag)
            )

        noise_std = torch.sqrt(noise_power)
        return noise_std * torch.randn_like(x)

    def _calculate_signal_power(self, x: torch.Tensor) -> torch.Tensor:
        signal_power = torch.mean(torch.abs(x.detach()) ** 2)
        return torch.clamp(signal_power, self.eps)

    def _calculate_noise_power(self, x: torch.Tensor) -> torch.Tensor:
        signal_power = self._calculate_signal_power(x)
        snr_linear = 10 ** (self.snr_db / 10.0)

        return signal_power / snr_linear
