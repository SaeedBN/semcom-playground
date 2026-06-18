from pathlib import Path

import torch
from torch import nn

from semcom.channels.factory import create_channel
from semcom.data.toy_text import create_toy_text_dataloader
from semcom.models.factory import create_model
from semcom.utils.config import load_config
from semcom.utils.io import ensure_dict, save_config
from semcom.utils.seed import set_seed


def main() -> None:
    cfg = load_config()

    set_seed(cfg.experiment.seed)

    output_dir = ensure_dict(Path(cfg.experiment.output_dir))
    save_config(cfg, output_dir)

    device = torch.device(cfg.training.device)

    dataloader, tokenizer = create_toy_text_dataloader(
        max_length=cfg.dataset.max_length,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
    )

    channel = create_channel(cfg.channel)

    model = create_model(
        cfg=cfg.model,
        vocab_size=tokenizer.vocab_size,
        max_length=cfg.dataset.max_length,
        pad_id=tokenizer.pad_id,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
    )

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    print(f"Experiment: {cfg.experiment.name}")
    print(f"Output directory: {output_dir}")
    print(f"Dataset: {cfg.dataset.name}")
    print(f"channel: {cfg.channel.name}")
    print(f"SNR: {cfg.channel.snr_db} dB")
    print(f"Model: {cfg.model.name}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Number of batches: {len(dataloader)}")
    print(f"Device: {device}")

    for epoch in range(cfg.training.epochs):
        model.train()
        total_loss = 0.0

        print(f"Epoch {epoch + 1}/{cfg.training.epochs}")

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"]
            target_ids = batch["target_ids"]
            attention_mask = batch["attention_mask"]

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                channel=channel,
            )

            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            print(
                f"  Batch {batch_idx + 1}/{len(dataloader)} "
                f"| loss={loss.item():.4f} "
                f"| logits shape={tuple(logits.shape)}"
            )

        mean_loss = total_loss / len(dataloader)
        print(f"Mean Training Loss: {mean_loss:.4f}")

    print("Training completed.")


if __name__ == "__main__":
    main()
