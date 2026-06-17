import pytest

from semcom.data.tokenizer import SimpleTextTokenizer


def test_tokenizer_encodes_to_fixed_length() -> None:
    texts = ["semantic communication preserves meaning"]
    tokenizer = SimpleTextTokenizer(texts)

    token_ids = tokenizer.encode(
        "semantic communication preserves meaning",
        max_length=8,
    )

    assert len(token_ids) == 8
    assert token_ids[0] == tokenizer.bos_id
    assert tokenizer.eos_id in token_ids


def test_tokenizer_decode_ignores_special_tokens() -> None:
    texts = ["semantic communication"]
    tokenizer = SimpleTextTokenizer(texts)

    token_ids = tokenizer.encode(
        "semantic communication",
        max_length=6,
    )

    decoded = tokenizer.decode(token_ids)

    assert decoded == "semantic communication"


def test_tokenizer_uses_unknown_token_for_unseen_words() -> None:
    texts = ["semantic communication"]
    tokenizer = SimpleTextTokenizer(texts)

    token_ids = tokenizer.encode(
        "semantic channel",
        max_length=5,
    )

    assert tokenizer.unk_id in token_ids


def test_tokenizer_rejects_too_short_max_length() -> None:
    tokenizer = SimpleTextTokenizer(["semantic communication"])

    with pytest.raises(ValueError):
        tokenizer.encode(
            "semantic communication",
            max_length=1,
        )
