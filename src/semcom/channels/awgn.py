import torch
from torch import nn


class AWGNChannel(nn.Module):
    """Simple AWGN channel"""

    def __init__(self, snr_db: float = 10, eps: float = 1e-12) -> None:
        super().__init__()

        self.snr_db = float(snr_db)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies AWGN to the input tensor x"""

        noise_power = self._calculate_noise_power(x)

        if torch.is_complex(x):
            noise_std = torch.sqrt(noise_power / 2.0)

            noise = noise_std * (
                torch.randn_like(x.real) + 1j * torch.randn_like(x.imag)
            )

        else:
            noise_std = torch.sqrt(noise_power)
            noise = noise_std * torch.randn_like(x)

        return x + noise

    def _calculate_signal_power(self, x: torch.Tensor) -> torch.Tensor:
        """Calculating the signal power"""
        signal_power = torch.mean(torch.abs(x.detach()) ** 2)
        return torch.clamp(signal_power, self.eps)

    def _calculate_noise_power(self, x: torch.Tensor) -> torch.Tensor:
        """Calculating noise power from signal power and snr level"""
        signal_power = self._calculate_signal_power(x)
        snr_linear = 10 ** (self.snr_db / 10.0)

        return signal_power / snr_linear
