from pathlib import Path

import torch

from semcom.channels.factory import create_channel
from semcom.data.toy_text import create_toy_text_dataloader
from semcom.utils.config import load_config
from semcom.utils.io import ensure_dict, save_config
from semcom.utils.seed import set_seed


def main() -> None:
    cfg = load_config()

    set_seed(cfg.experiment.seed)

    output_dir = ensure_dict(Path(cfg.experiment.output_dir))
    save_config(cfg, output_dir)

    dataloader, tokenizer = create_toy_text_dataloader(
        max_length=cfg.dataset.max_length,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
    )

    channel = create_channel(cfg.channel)

    print(f"Experiment: {cfg.experiment.name}")
    print(f"Output directory: {output_dir}")
    print(f"Dataset: {cfg.dataset.name}")
    print(f"channel: {cfg.channel.name}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Number of batches: {len(dataloader)}")

    for epoch in range(cfg.training.epochs):
        print(f"Epoch {epoch + 1}/{cfg.training.epochs}")

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"]
            target_ids = batch["target_ids"]
            attention_mask = batch["attention_mask"]

            # dummy channel input/ouput
            dummy_input = torch.randn(input_ids.shape[0], 8, cfg.model.latent_dim)
            noisy_output = channel(dummy_input)

            print(f"  Batch {batch_idx + 1}")
            print(f"    input_ids shape: {tuple(input_ids.shape)}")
            print(f"    target_ids shape: {tuple(target_ids.shape)}")
            print(f"    attention_mask shape: {tuple(attention_mask.shape)}")
            print(f"    Dummy input shape: {tuple(dummy_input.shape)}")
            print(f"    Noisy output shape: {tuple(noisy_output.shape)}")

    print("Training completed.")


if __name__ == "__main__":
    main()
