from pathlib import Path

import pytest

from semcom.utils.config import load_config, load_config_from_path


def test_default_config_loads() -> None:
    cfg = load_config()

    assert cfg.experiment.name == "debug_experiment"
    assert cfg.model.name == "debug_text_semantic_autoencoder"
    assert cfg.channel.name == "awgn"
    assert cfg.dataset.name == "toy_text"
    assert cfg.training.epochs == 1
    assert cfg.tracking.backend == "offline"


def test_experiment_config_loads() -> None:
    config_path = Path("experiments/001_text_awgn_debug/config.yaml")

    cfg = load_config_from_path(config_path)

    assert cfg.experiment.name == "001_text_awgn_debug"
    assert cfg.modality.name == "text"
    assert cfg.model.name == "debug_text_semantic_autoencoder"
    assert cfg.channel.name == "awgn"
    assert cfg.dataset.name == "toy_text"
    assert cfg.training.epochs == 50


def test_config_throws_error_invalid_file() -> None:
    config_path = Path("experiments/someInvalidFile.yaml")
    with pytest.raises(FileNotFoundError):
        load_config_from_path(config_path)
