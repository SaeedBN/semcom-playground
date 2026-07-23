import argparse
import json
import pickle
import re
import unicodedata
from pathlib import Path

from w3lib.html import remove_tags

SPECIAL_TOKENS = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "<unk>": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Europarl English text for DeepSC experiments.",
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to extracted Europarl directory.",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default="dataset/processed/europarl/",
        help="Path to processed output text file.",
    )

    return parser.parse_args()


def find_files(input_dir: str | Path) -> list[Path]:
    input_dir = Path(input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    text_files = sorted(input_dir.rglob("*.txt"))

    if text_files:
        return text_files

    raise FileNotFoundError(
        f"No Europarl files found under {input_dir}. Expected txt files"
    )


def unicode_to_ascii(text: str) -> str:
    """Converting unicode to ASCII"""
    return "".join(
        char
        for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
    )


def normalize_text(text: str) -> str:
    """noramlizing the text, unicode to ascii, removing tags,
    adding whitespace before punctuations, removing non-english letters,
    normalizing whitespaces"""

    text = unicode_to_ascii(text)
    text = remove_tags(text)

    text = re.sub(r"([!.?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.!?]+", r" ", text)
    text = re.sub(r"\s+", r" ", text)

    return text.lower().strip()


def keep_deepsc_sentence(sentence: str) -> bool:
    num_words = len(sentence.split())
    return 4 <= num_words <= 30


def process_file(path: str | Path) -> list[str]:
    file_path = Path(path)

    with file_path.open("r", encoding="utf-8", errors="ignore") as file:
        raw_lines = file.read().strip().split("\n")

    sentences_after_cleanup = []

    for line in raw_lines:
        sentence = normalize_text(line)

        if keep_deepsc_sentence(sentence):
            sentences_after_cleanup.append(" ".join(sentence.split()))

    return sentences_after_cleanup


def tokenize(sentence: str) -> list[str]:
    tokens = sentence.lower().split(" ")
    tokens.insert(0, "<bos>")
    tokens.append("<eos>")

    return tokens


def build_vocab(sentences: list[str]) -> dict[str, int]:

    token_to_idx = dict(SPECIAL_TOKENS)
    token_to_count: dict[str, int] = {}

    for sents in sentences:
        tokens = sents.split(" ")

        for token in tokens:
            token_to_count[token] = token_to_count.get(token, 0) + 1

    for token, _ in sorted(token_to_count.items()):
        if token not in token_to_idx:
            token_to_idx[token] = len(token_to_idx)

    return token_to_idx


def encode_sentence(sentence: str, token_to_idx: dict[str, int]) -> list[int]:
    tokens = tokenize(sentence)

    return [token_to_idx.get(token, token_to_idx["<unk>"]) for token in tokens]


def prepare_europarl(input_dir: str | Path, output_path: str | Path) -> dict[str, int]:
    txt_files = find_files(input_dir)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sentences = []

    for file in txt_files:
        sentences.extend(process_file(file))

    # removing duplicate sentences
    sentences = list(dict.fromkeys(sentences))

    token_to_idx = build_vocab(sentences)

    encoded_data = [encode_sentence(sents, token_to_idx) for sents in sentences]

    with (output_path / "encoded_data.pkl").open("wb") as file:
        pickle.dump(encoded_data, file)

    with (output_path / "vocab.json").open("w", encoding="utf-8") as file:
        json.dump({"token_to_idx": token_to_idx}, file, indent=2)

    with (output_path / "sentences.txt").open("w", encoding="utf-8") as file:
        for sents in sentences:
            file.write(sents + "\n")

    return {
        "num_files": len(txt_files),
        "num_sentences": len(sentences),
        "vocab_size": len(token_to_idx),
    }


def main() -> None:
    args = parse_args()

    preparation_stats = prepare_europarl(
        input_dir=args.input_dir,
        output_path=args.output_path,
    )

    print("Europarl dataset preprocessing completed.")
    print(f"Number of files: {preparation_stats['num_files']}")
    print(f"Number of sentences: {preparation_stats['num_sentences']}")
    print(f"Vocabulary size: {preparation_stats['vocab_size']}")
    print(f"Output directory: {args.output_path}")


if __name__ == "__main__":
    main()
