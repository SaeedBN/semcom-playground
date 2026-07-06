from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from semcom.data.tokenizer import SimpleTextTokenizer


def load_europarl_sentences(
    text_path: str | Path,
    min_words: int,
    max_words: int,
    max_samples: int | None = None,
) -> list[str]:
    # Loading preprocessed text file
    text_path = Path(text_path)

    if not text_path.exists():
        raise FileNotFoundError(f"Europarl processed text file not found: {text_path}")

    sentences: list[str] = []

    with text_path.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            sentence = " ".join(line.strip().split())

            if not sentence:
                continue

            num_words = len(sentence.split())

            if min_words <= num_words <= max_words:
                sentences.append(sentence)

            if max_samples is not None and len(sentences) >= max_samples:
                break

    if not sentences:
        raise ValueError("File did not have valid sentences.")

    return sentences


class EuroparlTextDataset(Dataset):
    def __init__(
        self,
        text_path: str | Path,
        min_words: int,
        max_words: int,
        max_length: int,
        max_samples: int | None = None,
    ) -> None:
        self.sentences = load_europarl_sentences(
            text_path=text_path,
            min_words=min_words,
            max_words=max_words,
            max_samples=max_samples,
        )
        self.max_length = max_length
        self.tokenizer = SimpleTextTokenizer(self.sentences)

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sentence = self.sentences[index]

        token_ids = self.tokenizer.encode(
            sentence,
            max_length=self.max_length,
        )

        input_ids = torch.tensor(token_ids, dtype=torch.long)
        target_ids = input_ids.clone()

        decoder_input_ids = torch.empty_like(target_ids)
        decoder_input_ids[0] = self.tokenizer.bos_id
        decoder_input_ids[1:] = target_ids[:-1]

        attention_mask = (input_ids != self.tokenizer.pad_id).long()
        decoder_attention_mask = (decoder_input_ids != self.tokenizer.pad_id).long()

        return {
            "input_ids": input_ids,
            "decoder_input_ids": decoder_input_ids,
            "target_ids": target_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
        }


def create_europarl_dataloaders(
    text_path: str | Path,
    min_words: int,
    max_words: int,
    max_length: int,
    max_samples: int | None,
    train_fraction: float,
    batch_size: int,
    shuffle_train: bool,
    seed: int,
) -> tuple[DataLoader, DataLoader, SimpleTextTokenizer]:

    dataset = EuroparlTextDataset(
        text_path=text_path,
        min_words=min_words,
        max_words=max_words,
        max_length=max_length,
        max_samples=max_samples,
    )

    train_size = int(len(dataset) * train_fraction)
    test_size = len(dataset) - train_size

    if train_size == 0 or test_size == 0:
        raise ValueError("Train/test split produced an empty subset.")

    generator = torch.Generator().manual_seed(seed)

    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader, dataset.tokenizer
