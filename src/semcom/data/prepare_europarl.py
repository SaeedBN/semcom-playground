import argparse
import re
from pathlib import Path


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
        default="dataset/processed/europarl/europarl_en.txt",
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


def clean_line(line: str) -> str | None:
    line = line.strip()

    if not line:
        return None

    line = re.sub(r"<[^>]+>", " ", line)

    line = " ".join(line.split())

    if not line:
        return None

    return line


def count_words(sentence: str) -> int:
    return len(sentence.split())


def keep_deepsc_sentence(sentence: str) -> bool:
    num_words = count_words(sentence)
    return 4 <= num_words <= 30


def prepare_europarl(input_dir: str | Path, output_path: str | Path) -> int:
    txt_files = find_files(input_dir)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(
            f"Processed dataset exists: {output_path}\n"
            + "If preprocessing is required, delete the file and run again.",
        )
        return sum(1 for _ in output_path.open("r", encoding="utf-8"))

    num_sentences_written = 0

    with output_path.open("w", encoding="utf-8") as output_file:
        for file_path in txt_files:
            with file_path.open("r", encoding="utf-8", errors="ignore") as input_file:
                for line in input_file:
                    sentence = clean_line(line)

                    if sentence is None:
                        continue

                    if keep_deepsc_sentence(sentence):
                        output_file.write(sentence + "\n")
                        num_sentences_written += 1

    return num_sentences_written


def main() -> None:
    args = parse_args()

    num_sentences_written = prepare_europarl(
        input_dir=args.input_dir,
        output_path=args.output_path,
    )

    print(f"Processed sentences: {num_sentences_written}")
    print(f"Output file: {args.output_path}")


if __name__ == "__main__":
    main()
