import json
import pickle
from pathlib import Path

from semcom.data.prepare_europarl import (
    build_vocab,
    encode_sentence,
    keep_deepsc_sentence,
    normalize_text,
    prepare_europarl,
    tokenize,
)


def test_normalize_text_matches_deepsc_style() -> None:
    text = "<SPEAKER> Café, olympic-2020 is important!"

    normalized = normalize_text(text)

    assert normalized == "cafe olympic is important !"


def test_keep_deepsc_sentence() -> None:
    assert keep_deepsc_sentence("This sentence has enough word to pass.")
    assert not keep_deepsc_sentence("This one not")


def test_prepare_europarl_filters_4_to_30_words(tmp_path: Path) -> None:
    input_path = tmp_path / "raw.txt"

    input_path.write_text(
        "\n".join(
            [
                "too short",
                "this sentence has exactly six words.",
                "this is another valid sentence.",
                "<SPEAKER> Another valid sentence appears here.",
            ]
        ),
        encoding="utf-8",
    )

    preparation_stats = prepare_europarl(
        input_dir=tmp_path,
        output_path=tmp_path,
    )

    cleanedup_sentences = [
        "this sentence has exactly six words .",
        "this is another valid sentence .",
        "Another valid sentence appears here .",
    ]

    assert preparation_stats["num_sentences"] == 3
    assert (tmp_path / "encoded_data.pkl").exists()
    assert (tmp_path / "vocab.json").exists()

    with (tmp_path / "encoded_data.pkl").open("rb") as file:
        encoded_data = pickle.load(file)

    with (tmp_path / "vocab.json").open("rb") as file:
        token_to_idx = json.load(file)["token_to_idx"]

    tokenized_sentences = []
    for sents in cleanedup_sentences:
        tokens = tokenize(sents)
        token_ids = [token_to_idx.get(token, token_to_idx["<unk>"]) for token in tokens]

        tokenized_sentences.append(token_ids)

    assert isinstance(encoded_data, list)
    assert encoded_data == tokenized_sentences


def test_build_vocab_and_encode_sentence() -> None:
    sentences = ["hello world", "hello semantic world"]

    vocab = build_vocab(sentences)
    encoded = encode_sentence("hello world", vocab)

    assert vocab["<pad>"] == 0
    assert vocab["<bos>"] == 1
    assert vocab["<eos>"] == 2
    assert vocab["<unk>"] == 3
    assert encoded[0] == vocab["<bos>"]
    assert encoded[-1] == vocab["<eos>"]
