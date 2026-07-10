import json
import pickle
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split


class EuroparlTextDataset(Dataset):
    def __init__(
        self,
        text_path: str | Path,
        max_length: int,
        max_samples: int | None = None,
    ) -> None:

        self.max_length = max_length
        text_path = Path(text_path)
        with (text_path / "encoded_data.pkl").open("rb") as file:
            self.token_ids = pickle.load(file)

        with (text_path / "vocab.json").open("rb") as file:
            self.token_to_idx = json.load(file)["token_to_idx"]

        self.pad_id = self.token_to_idx["<pad>"]
        for seq_idx in range(len(self.token_ids)):
            seq_length = len(self.token_ids[seq_idx])
            self.token_ids[seq_idx].extend(
                [self.pad_id] * (self.max_length - seq_length)
            )

        if max_samples is not None:
            sample_size = min(max_samples, len(self.token_ids))

            indices = torch.randperm(
                len(self.token_ids),
            )[:sample_size].tolist()
            self.token_ids = [self.token_ids[index] for index in indices]
        print(f"length of dataset: {len(self.token_ids)}")

    def __len__(self) -> int:
        return len(self.token_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sentence_token_ids = self.token_ids[index]

        input_ids = torch.tensor(sentence_token_ids, dtype=torch.long)
        decoder_input_ids = input_ids.clone()

        target_ids = torch.empty_like(input_ids)
        target_ids[:-1] = input_ids[1:]
        target_ids[-1] = self.token_to_idx["<pad>"]

        attention_mask = (input_ids != self.token_to_idx["<pad>"]).long()
        decoder_attention_mask = (
            decoder_input_ids != self.token_to_idx["<pad>"]
        ).long()

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
) -> tuple[DataLoader, DataLoader, dict[str, int]]:

    if not max_length >= max_words + 2:
        raise ValueError(
            "Max length must be equal or greater than max "
            "num of words + 2 (for <bos> and <eos>)"
        )

    dataset = EuroparlTextDataset(
        text_path=text_path,
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

    return train_loader, test_loader, dataset.token_to_idx
