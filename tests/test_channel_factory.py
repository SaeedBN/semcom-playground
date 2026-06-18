import pytest
from omegaconf import OmegaConf

from semcom.channels.awgn import AWGNChannel
from semcom.channels.factory import create_channel


def test_create_awgn_from_config() -> None:
    cfg = OmegaConf.create(
        {
            "name": "awgn",
            "snr_db": 10.0,
        }
    )

    channel = create_channel(cfg)

    assert isinstance(channel, AWGNChannel)
    assert channel.snr_db == 10.0


def test_channel_factory_rejects_unsupported_channel_model() -> None:
    cfg = OmegaConf.create(
        {
            "name": "unknown_channel",
            "snr_db": 10.0,
        }
    )

    with pytest.raises(ValueError):
        create_channel(cfg)
