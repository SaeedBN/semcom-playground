import pytest
import torch

from semcom.channels.awgn import AWGNChannel
from semcom.models.text.semantic_autoencoder import TextSemanticAutoencoder


def test_text_semantic_autoencoder_forward_shape() -> None:
    model = TextSemanticAutoencoder(
        vocab_size=50,
        max_length=16,
        pad_id=0,
        d_model=32,
        latent_dim=16,
        n_heads=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dropout=0.1,
    )

    input_ids = torch.randint(0, 50, (4, 16))
    attention_mask = torch.ones(4, 16, dtype=torch.long)

    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    assert logits.shape == (4, 16, 50)


def test_text_semantic_autoencoder_forward_with_channel() -> None:
    model = TextSemanticAutoencoder(
        vocab_size=50,
        max_length=16,
        pad_id=0,
        d_model=32,
        latent_dim=16,
        n_heads=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dropout=0.1,
    )

    channel = AWGNChannel(snr_db=10.0)

    input_ids = torch.randint(0, 50, (4, 16))
    attention_mask = torch.ones(4, 16, dtype=torch.long)

    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        channel=channel,
    )

    assert logits.shape == (4, 16, 50)


def test_text_semantic_autoencoder_rejects_invalid_heads() -> None:
    with pytest.raises(ValueError):
        TextSemanticAutoencoder(
            vocab_size=50,
            max_length=16,
            pad_id=0,
            d_model=30,
            latent_dim=16,
            n_heads=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dropout=0.1,
        )


def test_text_semantic_autoencoder_rejects_too_long_sequence() -> None:
    model = TextSemanticAutoencoder(
        vocab_size=50,
        max_length=16,
        pad_id=0,
        d_model=32,
        latent_dim=16,
        n_heads=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dropout=0.1,
    )

    input_ids = torch.randint(0, 50, (4, 20))

    with pytest.raises(ValueError):
        model(input_ids)
