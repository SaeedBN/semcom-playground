from semcom.data.toy_text import ToyTextDataset, create_toy_text_dataloader


def test_toy_text_dataset_item_shapes() -> None:
    dataset = ToyTextDataset(max_length=16)

    sample = dataset[0]

    assert sample["input_ids"].shape == (16,)
    assert sample["target_ids"].shape == (16,)
    assert sample["attention_mask"].shape == (16,)


def test_toy_text_dataset_input_and_target_match_initially() -> None:
    dataset = ToyTextDataset(max_length=16)

    sample = dataset[0]

    assert sample["input_ids"].equal(sample["target_ids"])


def test_toy_text_dataloader_batch_shapes() -> None:
    dataloader, tokenizer = create_toy_text_dataloader(
        max_length=16,
        batch_size=4,
        shuffle=False,
    )

    batch = next(iter(dataloader))

    assert batch["input_ids"].shape == (4, 16)
    assert batch["target_ids"].shape == (4, 16)
    assert batch["attention_mask"].shape == (4, 16)
    assert tokenizer.vocab_size > 4
