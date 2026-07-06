import argparse
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from torch import nn

from semcom.channels.factory import create_channel
from semcom.data.europarl import create_europarl_dataloaders
from semcom.evaluation.text_metrics import (
    predictions_to_token_ids,
    sequence_accuracy,
    token_accuracy,
)
from semcom.models.factory import create_model
from semcom.utils.config import load_config_from_path
from semcom.utils.io import ensure_dir, save_config, save_json
from semcom.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DeepSC text experiment.",
    )

    parser.add_argument(
        "-c",
        "--config-path",
        type=str,
        required=True,
        help="Path to experiment YAML config.",
    )

    return parser.parse_args()


def create_awgn_channel_given_snr(snr_db: float) -> nn.Module:
    channel_cfg = OmegaConf.create(
        {
            "name": "awgn",
            "snr_db": snr_db,
        }
    )
    return create_channel(channel_cfg)


def sample_training_snr(
    snr_values_db: list[int | float],
    batch_index: int,
    epoch_index: int,
) -> float:
    """For now, we are sampling cyclically. Will change this to random"""
    index = (epoch_index + batch_index) % len(snr_values_db)
    return float(snr_values_db[index])


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    pad_id: int,
    device: torch.device,
    snr_values_db: list[int | float],
    epoch_index: int,
) -> dict[str, float]:
    model.train()

    total_loss = 0.0
    total_token_accuracy = 0.0
    total_sequence_accuracy = 0.0

    for batch_index, batch in enumerate(dataloader):
        snr_db = sample_training_snr(
            snr_values_db=snr_values_db,
            batch_index=batch_index,
            epoch_index=epoch_index,
        )

        channel = create_awgn_channel_given_snr(snr_db).to(device)

        input_ids = batch["input_ids"].to(device)
        decoder_input_ids = batch["decoder_input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(device)

        logits = model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
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
        total_token_accuracy += token_accuracy(
            logits=logits.detach(),
            target_ids=target_ids,
            pad_id=pad_id,
        )
        total_sequence_accuracy += sequence_accuracy(
            logits=logits.detach(),
            target_ids=target_ids,
            pad_id=pad_id,
        )

    num_batches = len(dataloader)

    return {
        "loss": total_loss / num_batches,
        "token_accuracy": total_token_accuracy / num_batches,
        "sequence_accuracy": total_sequence_accuracy / num_batches,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    channel: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    tokenizer: Any,
    device: torch.device,
    max_examples: int,
) -> dict[str, Any]:

    model.eval()

    total_loss = 0.0
    total_token_accuracy = 0.0
    total_sequence_accuracy = 0.0
    examples = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        decoder_input_ids = batch["decoder_input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(device)

        logits = model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            channel=channel,
        )

        loss = loss_fn(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
        )

        total_loss += loss.item()
        total_token_accuracy += token_accuracy(
            logits=logits,
            target_ids=target_ids,
            pad_id=tokenizer.pad_id,
        )
        total_sequence_accuracy += sequence_accuracy(
            logits=logits,
            target_ids=target_ids,
            pad_id=tokenizer.pad_id,
        )

        predicted_ids = predictions_to_token_ids(logits).cpu()
        target_ids_cpu = target_ids.cpu()

        for original_ids, reconstructed_ids in zip(
            target_ids_cpu,
            predicted_ids,
            strict=False,
        ):
            if len(examples) >= max_examples:
                break

            examples.append(
                {
                    "original": tokenizer.decode(original_ids.tolist()),
                    "reconstructed": tokenizer.decode(reconstructed_ids.tolist()),
                }
            )

    num_batches = len(dataloader)

    return {
        "loss": total_loss / num_batches,
        "token_accuracy": total_token_accuracy / num_batches,
        "sequence_accuracy": total_sequence_accuracy / num_batches,
        "examples": examples,
    }


def main() -> None:

    args = parse_args()
    cfg = load_config_from_path(args.config_path)

    set_seed(cfg.experiment.seed)

    output_dir = ensure_dir(Path(cfg.experiment.output_dir))
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    save_config(cfg, output_dir)

    device = torch.device(cfg.training.device)

    train_loader, test_loader, tokenizer = create_europarl_dataloaders(
        text_path=cfg.dataset.text_path,
        min_words=cfg.dataset.min_words,
        max_words=cfg.dataset.max_words,
        max_length=cfg.dataset.max_length,
        max_samples=cfg.dataset.max_samples,
        train_fraction=cfg.dataset.train_fraction,
        batch_size=cfg.dataset.batch_size,
        shuffle_train=cfg.dataset.shuffle_train,
        seed=cfg.experiment.seed,
    )

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
    print(f"Paper: {cfg.paper.title}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Model: {cfg.model.name}")
    print(f"Channel: {cfg.channel.name}")
    print(f"Training SNR range: {list(cfg.training.snr_values_db)} dB")
    print(f"Evaluation SNR values: {list(cfg.evaluation.snr_values_db)} dB")
    print(f"Device: {device}")

    history = []

    for epoch in range(cfg.training.epochs):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            pad_id=tokenizer.pad_id,
            device=device,
            snr_values_db=list(cfg.training.snr_values_db),
            epoch_index=epoch,
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "train_token_accuracy": train_metrics["token_accuracy"],
                "train_sequence_accuracy": train_metrics["sequence_accuracy"],
            }
        )

        print(
            f"Epoch {epoch + 1:03d}/{cfg.training.epochs} "
            f"| loss={train_metrics['loss']:.4f} "
            f"| token_acc={train_metrics['token_accuracy']:.4f} "
            f"| seq_acc={train_metrics['sequence_accuracy']:.4f}"
        )

    evaluation_results = []

    for snr_db in cfg.evaluation.snr_values_db:
        eval_channel = create_awgn_channel_given_snr(float(snr_db)).to(device)

        result = evaluate(
            model=model,
            channel=eval_channel,
            dataloader=test_loader,
            loss_fn=loss_fn,
            tokenizer=tokenizer,
            device=device,
            max_examples=cfg.evaluation.num_reconstruction_examples,
        )

        result["snr_db"] = float(snr_db)
        evaluation_results.append(result)

        print(
            f"SNR={float(snr_db):>5.1f} dB "
            f"| loss={result['loss']:.4f} "
            f"| token_acc={result['token_accuracy']:.4f} "
            f"| seq_acc={result['sequence_accuracy']:.4f}"
        )

    checkpoint_path = checkpoint_dir / "model.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "vocab_size": tokenizer.vocab_size,
            "pad_id": tokenizer.pad_id,
        },
        checkpoint_path,
    )

    metrics = {
        "experiment_name": cfg.experiment.name,
        "paper_reference": cfg.paper.title,
        "training_snr_range_db": list(cfg.training.snr_values_db),
        "evaluation_snr_values_db": list(cfg.evaluation.snr_values_db),
        "training_history": history,
        "evaluation_results": evaluation_results,
    }

    save_json(metrics, output_dir / "metrics.json")

    print(f"Saved checkpoint to: {checkpoint_path}")
    print(f"Saved metrics to: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
