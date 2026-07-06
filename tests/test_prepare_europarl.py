from pathlib import Path

from semcom.data.prepare_europarl import (
    clean_line,
    count_words,
    keep_deepsc_sentence,
    prepare_europarl,
)


def test_word_counting() -> None:
    assert count_words("This sentence must have six words") == 6


def test_keep_deepsc_sentence() -> None:
    assert keep_deepsc_sentence("This sentence has enough word to pass.")
    assert not keep_deepsc_sentence("This one not")


def test_prepare_europarl_filters_4_to_30_words(tmp_path: Path) -> None:
    input_path = tmp_path / "raw.txt"
    output_path = tmp_path / "processed.txt"

    input_path.write_text(
        "\n".join(
            [
                "too short",
                "this sentence has exactly six words",
                "this is another valid sentence",
            ]
        ),
        encoding="utf-8",
    )

    num_sentences_written = prepare_europarl(
        input_dir=tmp_path,
        output_path=output_path,
    )

    assert num_sentences_written == 2

    lines = output_path.read_text(encoding="utf-8").splitlines()

    assert lines == [
        "this sentence has exactly six words",
        "this is another valid sentence",
    ]


def test_clean_line_removes_markup() -> None:
    cleaned = clean_line("<SPEAKER ID='1'> This is a valid sentence.")

    assert cleaned == "This is a valid sentence."
