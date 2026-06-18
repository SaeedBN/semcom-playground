import pytest
from omegaconf import OmegaConf

from semcom.models.factory import create_model
from semcom.models.text.semantic_autoencoder import TextSemanticAutoencoder


def test_create_text_semantic_autoencoder_from_config() -> None:
    cfg = OmegaConf.create(
        {
            "name": "debug_text_semantic_autoencoder",
            "d_model": 32,
            "latent_dim": 16,
            "n_heads": 4,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "dropout": 0.1,
        }
    )

    model = create_model(
        cfg=cfg,
        vocab_size=50,
        max_length=16,
        pad_id=0,
    )

    assert isinstance(model, TextSemanticAutoencoder)


def test_create_model_rejects_unknown_model() -> None:
    cfg = OmegaConf.create(
        {
            "name": "unknown_model",
            "d_model": 32,
            "latent_dim": 16,
            "n_heads": 4,
            "num_encoder_layers": 1,
            "num_decoder_layers": 1,
            "dropout": 0.1,
        }
    )

    with pytest.raises(ValueError):
        create_model(
            cfg=cfg,
            vocab_size=50,
            max_length=16,
            pad_id=0,
        )
