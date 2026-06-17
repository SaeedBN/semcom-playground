import torch
from torch.utils.data import DataLoader, Dataset

from semcom.data.tokenizer import SimpleTextTokenizer


TOY_TEXT_SENTENCES = [
    "semantic communication preserves meaning",
    "wireless channels introduce noise",
    "deep learning can learn robust representations",
    "the receiver reconstructs the transmitted message",
    "attention models capture relationships between tokens",
    "semantic similarity is different from bit accuracy",
    "low snr makes communication difficult",
    "channel coding protects information from noise",
    "image transmission can preserve visual meaning",
    "text transmission can preserve sentence meaning",
]


class ToyTextDataset(Dataset):
    def __init__(self, max_length: int) -> None:
        self.sentences = TOY_TEXT_SENTENCES
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
        attention_mask = (input_ids != self.tokenizer.pad_id).long()

        return {
            "input_ids": input_ids,
            "target_ids": input_ids.clone(),
            "attention_mask": attention_mask,
        }


def create_toy_text_dataloader(
    max_length: int,
    batch_size: int,
    shuffle: bool = True,
) -> tuple[DataLoader, SimpleTextTokenizer]:

    dataset = ToyTextDataset(max_length=max_length)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    return dataloader, dataset.tokenizer
