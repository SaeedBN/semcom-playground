import torch
from torch import nn


class TextSemanticAutoencoder(nn.Module):
    """Simple Transformer-like text semantic communication model.

    Operations:

        token IDs
        -> token + positional embeddings
        -> Transformer encoder
        -> semantic latent representation
        -> optional wireless channel
        -> latent-to-model projection
        -> Transformer decoder/refiner
        -> vocabulary logits

    This is just an simple start for the implementation.
    """

    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        pad_id: int,
        d_model: int,
        latent_dim: int,
        n_heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")

        self.vocab_size = vocab_size
        self.max_length = max_length
        self.pad_id = pad_id
        self.d_model = d_model
        self.latent_dim = latent_dim

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_id,
        )

        self.position_embedding = nn.Embedding(
            num_embeddings=max_length,
            embedding_dim=d_model,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.semantic_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
            enable_nested_tensor=False,
        )

        self.to_latent = nn.Linear(d_model, latent_dim)
        self.from_latent = nn.Linear(latent_dim, d_model)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.semantic_decoder = nn.TransformerEncoder(
            encoder_layer=decoder_layer,
            num_layers=num_decoder_layers,
            enable_nested_tensor=False,
        )

        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        channel: nn.Module | None = None,
    ) -> torch.Tensor:
        """Feeds input forward going through encoder, channel, and decoder

        Args:
            input_ids: Token IDs with shape (B x max_length).
            attention_mask: Mask with 1 for real tokens and 0 for PAD tokens.
            channel: Optional wireless channel module applied to the encoder's output.

        Returns:
            Prediction logits:

                (batch_size, max_length, vocab_size)
        """
        batch_size, sequence_length = input_ids.shape

        if sequence_length > self.max_length:
            raise ValueError(
                f"sequence_length={sequence_length} exceeds "
                f"max_length={self.max_length}."
            )

        positions = torch.arange(
            sequence_length,
            device=input_ids.device,
        ).unsqueeze(0)

        positions = positions.expand(batch_size, sequence_length)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        if attention_mask is None:
            key_padding_mask = input_ids == self.pad_id
        else:
            key_padding_mask = attention_mask == 0

        encoded = self.semantic_encoder(
            x,
            src_key_padding_mask=key_padding_mask,
        )

        latent = self.to_latent(encoded)

        if channel is not None:
            latent = channel(latent)

        decoded_features = self.from_latent(latent)

        decoded = self.semantic_decoder(
            decoded_features,
            src_key_padding_mask=key_padding_mask,
        )

        logits = self.output_projection(decoded)

        return logits
