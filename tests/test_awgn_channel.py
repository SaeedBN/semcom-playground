import torch

from semcom.channels.awgn import AWGNChannel


def test_awgn_channel_preserves_shape() -> None:
    channel = AWGNChannel(snr_db=10.0)

    x = torch.ones(10, 4, 16)
    y = channel(x)

    assert y.shape == x.shape


def test_awgn_channel_changes_signal() -> None:
    channel = AWGNChannel(snr_db=10.0)

    x = torch.ones(10, 4, 16)
    y = channel(x)

    assert not torch.allclose(x, y)


def test_awgn_higher_snr_lower_noise_power() -> None:
    x = torch.ones(10000)

    low_snr_channel = AWGNChannel(snr_db=0.0)
    high_snr_channel = AWGNChannel(snr_db=20.0)

    torch.manual_seed(42)
    y_low_snr = low_snr_channel(x)

    torch.manual_seed(42)
    y_high_snr = high_snr_channel(x)

    low_snr_noise_power = torch.mean((y_low_snr - x) ** 2)
    high_snr_noise_power = torch.mean((y_high_snr - x) ** 2)

    assert low_snr_noise_power > high_snr_noise_power


def test_awgn_channel_handles_complex_input() -> None:
    channel = AWGNChannel(snr_db=10.0)

    x = torch.ones(4, 8, dtype=torch.complex64)
    y = channel(x)

    assert y.shape == x.shape
    assert torch.is_complex(y)
