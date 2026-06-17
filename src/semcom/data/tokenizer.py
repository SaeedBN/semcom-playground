from dataclasses import dataclass


@dataclass(frozen=True)
class SpecialTokens:
    pad: str = "<pad>"
    bos: str = "<bos>"
    eos: str = "<eos>"
    unk: str = "<unk>"


class SimpleTextTokenizer:
    """Deterministic + Word-level"""

    def __init__(self, texts: list[str]) -> None:
        self.special_tokens = SpecialTokens()

        special_order = [
            self.special_tokens.pad,
            self.special_tokens.bos,
            self.special_tokens.eos,
            self.special_tokens.unk,
        ]

        vocabulary = set(special_order)

        for text in texts:
            vocabulary.update(self._basic_tokenize(text))

        regular_tokens = sorted(
            token for token in vocabulary if token not in special_order
        )

        self.id_to_token = special_order + regular_tokens
        self.token_to_id = {
            token: token_id for token_id, token in enumerate(self.id_to_token)
        }

    @property
    def pad_id(self) -> int:
        """PAD Token ID"""
        return self.token_to_id[self.special_tokens.pad]

    @property
    def bos_id(self) -> int:
        """BOS Token ID"""
        return self.token_to_id[self.special_tokens.bos]

    @property
    def eos_id(self) -> int:
        """EOS Token ID"""
        return self.token_to_id[self.special_tokens.eos]

    @property
    def unk_id(self) -> int:
        """UNK Token ID"""
        return self.token_to_id[self.special_tokens.unk]

    @property
    def vocab_size(self) -> int:
        """Vocab. set size"""
        return len(self.id_to_token)

    def encode(self, text: str, max_length: int) -> list[int]:
        """Encode based on produced tokenizer:
        - Too short: padd the sentence to reach the fixed size
        - Too long: Truncate the sentence to the fixed size keeping the last
        token to <eos>
        """
        if max_length < 2:
            raise ValueError("max_length must be at least 2 to fit BOS and EOS.")

        words = self._basic_tokenize(text)

        max_content_length = max_length - 2
        words = words[:max_content_length]

        tokens = [
            self.special_tokens.bos,
            *words,
            self.special_tokens.eos,
        ]

        token_ids = [self.token_to_id.get(token, self.unk_id) for token in tokens]

        padding_length = max_length - len(token_ids)
        token_ids.extend([self.pad_id] * padding_length)

        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back into a sentence.

        PAD, BOS, and EOS are ignored.
        """
        ignored_tokens = {
            self.special_tokens.pad,
            self.special_tokens.bos,
            self.special_tokens.eos,
        }

        decoded_tokens = []

        for token_id in token_ids:
            token = self.id_to_token[token_id]

            if token not in ignored_tokens:
                decoded_tokens.append(token)

        return " ".join(decoded_tokens)

    @staticmethod
    def _basic_tokenize(text: str) -> list[str]:
        return text.lower().strip().split()
