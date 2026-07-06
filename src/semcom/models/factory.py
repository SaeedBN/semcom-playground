from omegaconf import DictConfig
from torch import nn

from semcom.models.text.deepsc import DeepSCTextModel
from semcom.models.text.semantic_autoencoder import TextSemanticAutoencoder


def create_model(
    cfg: DictConfig,
    vocab_size: int,
    max_length: int,
    pad_id: int,
) -> nn.Module:
    """Creating the model from the config file"""
    if cfg.name == "debug_text_semantic_autoencoder":
        return TextSemanticAutoencoder(
            vocab_size=vocab_size,
            max_length=max_length,
            pad_id=pad_id,
            d_model=cfg.d_model,
            latent_dim=cfg.latent_dim,
            n_heads=cfg.n_heads,
            num_encoder_layers=cfg.num_encoder_layers,
            num_decoder_layers=cfg.num_decoder_layers,
            dropout=cfg.dropout,
        )
    elif cfg.name == "deepsc_text_paper":
        return DeepSCTextModel(
            vocab_size=vocab_size,
            max_length=max_length,
            pad_id=pad_id,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            num_encoder_layers=cfg.num_encoder_layers,
            num_decoder_layers=cfg.num_decoder_layers,
            channel_encoder_hidden_dim=cfg.channel_encoder_hidden_dim,
            channel_symbols_dim=cfg.channel_symbols_dim,
            channel_decoder_hidden_dim=cfg.channel_decoder_hidden_dim,
            dropout=cfg.dropout,
        )

    raise ValueError(f"Model {cfg.name} is not supported.")
