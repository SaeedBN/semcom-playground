import torch
from torch import nn


class DeepSCTextModel(nn.Module):
    """DeepSC-style semantic communication model for text.

    Architecture based on the DeepSC paper's semantic network setup:
    Transformer semantic encoder, dense channel encoder, wireless channel,
    dense channel decoder, Transformer semantic decoder, prediction layer.
    """

    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        pad_id: int,
        d_model: int,
        n_heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        channel_encoder_hidden_dim: int,
        channel_symbols_dim: int,
        channel_decoder_hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")

        self.vocab_size = vocab_size
        self.max_length = max_length
        self.pad_id = pad_id

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

        self.channel_encoder = nn.Sequential(
            nn.Linear(d_model, channel_encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(channel_encoder_hidden_dim, channel_symbols_dim),
            nn.ReLU(),
        )

        self.channel_decoder = nn.Sequential(
            nn.Linear(channel_symbols_dim, channel_decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(channel_decoder_hidden_dim, d_model),
            nn.ReLU(),
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.semantic_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_decoder_layers,
        )

        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
        channel: nn.Module | None = None,
    ) -> torch.Tensor:
        recovered_memory, source_key_padding_mask = self.encode_channel(
            input_ids=input_ids,
            attention_mask=attention_mask,
            channel=channel,
        )

        batch_size, target_length = decoder_input_ids.shape

        target_positions = self._make_positions(
            batch_size=batch_size,
            sequence_length=target_length,
            device=decoder_input_ids.device,
        )

        target_embeddings = self.token_embedding(
            decoder_input_ids
        ) + self.position_embedding(target_positions)

        target_key_padding_mask = self._make_key_padding_mask(
            token_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
        )

        target_mask = self._make_causal_mask(
            target_length,
            device=decoder_input_ids.device,
        )

        decoded = self.semantic_decoder(
            tgt=target_embeddings,
            memory=recovered_memory,
            tgt_mask=target_mask,
            tgt_key_padding_mask=target_key_padding_mask,
            memory_key_padding_mask=source_key_padding_mask,
        )

        logits = self.output_projection(decoded)

        return logits

    def encode_channel(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        channel: nn.Module | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        batch_size, sequence_length = input_ids.shape

        if sequence_length > self.max_length:
            raise ValueError(
                f"sequence_length={sequence_length} \
                exceeds max_length={self.max_length}."
            )

        source_positions = self._make_positions(
            batch_size=batch_size,
            sequence_length=sequence_length,
            device=input_ids.device,
        )

        source_embeddings = self.token_embedding(input_ids) + self.position_embedding(
            source_positions
        )

        source_key_padding_mask = self._make_key_padding_mask(
            token_ids=input_ids,
            attention_mask=attention_mask,
        )

        memory = self.semantic_encoder(
            source_embeddings,
            src_key_padding_mask=source_key_padding_mask,
        )

        transmitted_symbols = self.channel_encoder(memory)

        if channel is not None:
            received_symbols = channel(transmitted_symbols)
        else:
            received_symbols = transmitted_symbols

        recovered_memory = self.channel_decoder(received_symbols)

        return recovered_memory, source_key_padding_mask

    def decode_with_memory(
        self,
        recovered_memory: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, target_length = decoder_input_ids.shape

        target_positions = self._make_positions(
            batch_size=batch_size,
            sequence_length=target_length,
            device=decoder_input_ids.device,
        )

        target_embeddings = self.token_embedding(
            decoder_input_ids
        ) + self.position_embedding(target_positions)

        target_key_padding_mask = decoder_input_ids == self.pad_id

        target_mask = self._make_causal_mask(
            target_length,
            device=decoder_input_ids.device,
        )

        decoded = self.semantic_decoder(
            tgt=target_embeddings,
            memory=recovered_memory,
            tgt_mask=target_mask,
            tgt_key_padding_mask=target_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        logits = self.output_projection(decoded)

        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        channel: nn.Module | None,
        bos_id: int,
        eos_id: int,
        max_length: int,
    ) -> torch.Tensor:

        self.eval()

        recovered_memory, source_key_padding_mask = self.encode_channel(
            input_ids=input_ids,
            attention_mask=attention_mask,
            channel=channel,
        )

        batch_size = input_ids.shape[0]
        device = input_ids.device

        generated = torch.full(
            size=(batch_size, 1),
            fill_value=bos_id,
            dtype=torch.long,
            device=device,
        )

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_length - 1):
            logits = self.decode_with_memory(
                recovered_memory=recovered_memory,
                decoder_input_ids=generated,
                memory_key_padding_mask=source_key_padding_mask,
            )

            next_token = torch.argmax(logits[:, -1, :], dim=-1)

            next_token = torch.where(
                finished,
                torch.full_like(next_token, eos_id),
                next_token,
            )

            generated = torch.cat(
                [generated, next_token.unsqueeze(1)],
                dim=1,
            )

            finished = finished | (next_token == eos_id)

            if finished.all():
                break

        if generated.shape[1] < max_length:
            pad_length = max_length - generated.shape[1]
            padding = torch.full(
                size=(batch_size, pad_length),
                fill_value=self.pad_id,
                dtype=torch.long,
                device=device,
            )
            generated = torch.cat([generated, padding], dim=1)

        return generated

    def _make_positions(
        self,
        batch_size: int,
        sequence_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        positions = torch.arange(sequence_length, device=device).unsqueeze(0)
        return positions.expand(batch_size, sequence_length)

    @staticmethod
    def _make_causal_mask(sequence_length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=device,
            ),
            diagonal=1,
        )

    def _make_key_padding_mask(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if attention_mask is None:
            return token_ids == self.pad_id

        return attention_mask == 0
