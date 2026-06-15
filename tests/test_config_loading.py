from semcom.utils.config import load_config


def test_default_config_loads() -> None:
    cfg = load_config()

    assert cfg.experiment.name == "debug_experiment"
    assert cfg.model.name == "debug_model"
    assert cfg.channel.name == "awgn"
    assert cfg.dataset.name == "toy_text"
    assert cfg.training.epochs == 1
    assert cfg.tracking.backend == "offline"