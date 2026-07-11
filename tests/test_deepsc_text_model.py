import torch

from semcom.channels.awgn import AWGNChannel
from semcom.models.text.deepsc import DeepSCTextModel


def test_deepsc_text_model_forward_shape() -> None:
    model = DeepSCTextModel(
        vocab_size=100,
        max_length=32,
        pad_id=0,
        d_model=128,
        n_heads=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        d_feedforward=512,
        channel_encoder_hidden_dim=256,
        channel_symbols_dim=16,
        channel_decoder_hidden_dim=512,
        dropout=0.1,
    )

    input_ids = torch.randint(0, 100, (2, 32))
    decoder_input_ids = torch.randint(0, 100, (2, 32))
    attention_mask = torch.ones(2, 32, dtype=torch.long)
    decoder_attention_mask = torch.ones(2, 32, dtype=torch.long)

    channel = AWGNChannel(snr_db=10.0)

    logits = model(
        input_ids=input_ids,
        decoder_input_ids=decoder_input_ids,
        attention_mask=attention_mask,
        decoder_attention_mask=decoder_attention_mask,
        channel=channel,
    )

    assert logits.shape == (2, 32, 100)
