import argparse
from pathlib import Path

import torch
from torch import nn

from semcom.channels.factory import create_channel
from semcom.data.toy_text import create_toy_text_dataloader
from semcom.evaluation.text_metrics import sequence_accuracy, token_accuracy
from semcom.models.factory import create_model
from semcom.utils.config import load_config, load_config_from_path
from semcom.utils.io import ensure_dir, save_config, save_json
from semcom.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="We will train a semantic communication model."
    )

    parser.add_argument(
        "-c",
        "--config-path",
        dest="config_path",
        type=str,
        default=None,
        help="Path to the configuration file.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.config_path is None:
        cfg = load_config()
    else:
        cfg = load_config_from_path(config_path=args.config_path)

    set_seed(cfg.experiment.seed)

    output_dir = ensure_dir(Path(cfg.experiment.output_dir))
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
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
    print(f"Channel: {cfg.channel.name}")
    print(f"SNR: {cfg.channel.snr_db} dB")
    print(f"Model: {cfg.model.name}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Number of batches: {len(dataloader)}")
    print(f"Device: {device}")

    history: list[dict[str, float | int]] = []

    for epoch in range(cfg.training.epochs):
        model.train()

        total_loss = 0.0
        total_token_accuracy = 0.0
        total_sequence_accuracy = 0.0

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

            batch_token_accuracy = token_accuracy(
                logits=logits.detach(),
                target_ids=target_ids,
                pad_id=tokenizer.pad_id,
            )

            batch_sequence_accuracy = sequence_accuracy(
                logits=logits.detach(),
                target_ids=target_ids,
                pad_id=tokenizer.pad_id,
            )

            total_loss += loss.item()
            total_token_accuracy += batch_token_accuracy
            total_sequence_accuracy += batch_sequence_accuracy

            print(
                f"  Batch {batch_idx + 1}/{len(dataloader)} "
                f"| loss={loss.item():.4f} "
                f"| toekn_acc={batch_token_accuracy:.4f} "
                f"| seq_acc={batch_sequence_accuracy:.4f}"
            )

        mean_loss = total_loss / len(dataloader)
        mean_token_accuracy = total_token_accuracy / len(dataloader)
        mean_sequence_accuracy = total_sequence_accuracy / len(dataloader)

        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": mean_loss,
            "train_token_accuracy": mean_token_accuracy,
            "train_sequence_accuracy": mean_sequence_accuracy,
        }

        history.append(epoch_metrics)

        print(
            f"Mean Training Loss: {mean_loss:.4f} "
            f"| token_acc={mean_token_accuracy:.4f} "
            f"| seq_acc={mean_sequence_accuracy:.4f}"
        )

    checkpoint_path = checkpoint_dir / "model.pt"
    torch.save(
        {
            "model_state_dic": model.state_dict(),
            "optimizer_state_dic": optimizer.state_dict(),
            "config": cfg,
            "vocab_size": tokenizer.vocab_size,
            "pad_id": tokenizer.pad_id,
        },
        checkpoint_path,
    )

    metrics = {
        "experiment_name": cfg.experiment.name,
        "model_name": cfg.model.name,
        "dataset_name": cfg.dataset.name,
        "channel_name": cfg.channel.name,
        "snr_db": cfg.channel.snr_db,
        "epochs": cfg.training.epochs,
        "history": history,
        "final": history[-1],
    }

    save_json(metrics, output_dir / "metrics.json")

    print(f"Saved checkpoint to: {checkpoint_path}")
    print(f"Saved metrics to: {output_dir / 'metrics.json'}")
    print("Training completed.")


if __name__ == "__main__":
    main()
