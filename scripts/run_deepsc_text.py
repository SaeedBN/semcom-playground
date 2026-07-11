import argparse
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from torch import nn

from semcom.channels.factory import create_channel
from semcom.data.europarl import create_europarl_dataloaders
from semcom.evaluation.text_metrics import (
    corpus_bleu_score,
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
    token_to_idx: dict[str, int],
    device: torch.device,
    max_examples: int,
    max_length: int,
) -> dict[str, Any]:

    id_to_token = {idx: token for token, idx in token_to_idx.items()}
    pad_id = token_to_idx["<pad>"]
    bos_id = token_to_idx.get("<bos>")
    eos_id = token_to_idx.get("<eos>")

    if bos_id is None or eos_id is None:
        raise ValueError("Vocabulary must contain BOS/EOS.")

    def decode_idx_to_token(idx_seq: list[int]) -> str:
        ignored_token_ids = {pad_id, bos_id, eos_id}

        decoded_tokens = []

        for token_id in idx_seq:
            token_id = int(token_id)

            if token_id == eos_id:
                break

            if token_id in ignored_token_ids:
                continue

            token = id_to_token.get(int(token_id), "<unk>")
            decoded_tokens.append(token)

        return " ".join(decoded_tokens)

    model.eval()

    all_references = []
    all_hypotheses = []
    examples = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            channel=channel,
            bos_id=bos_id,
            eos_id=eos_id,
            max_length=max_length,
        )

        input_ids_cpu = input_ids.cpu()
        generated_ids_cpu = generated_ids.cpu()

        for reference_ids, hypothesis_ids in zip(
            input_ids_cpu,
            generated_ids_cpu,
            strict=False,
        ):
            reference = decode_idx_to_token(reference_ids.tolist())
            hypothesis = decode_idx_to_token(hypothesis_ids.tolist())

            all_references.append(reference)
            all_hypotheses.append(hypothesis)

            if len(examples) < max_examples:
                examples.append(
                    {
                        "original": reference,
                        "reconstructed": hypothesis,
                    }
                )

    bleu_1 = corpus_bleu_score(
        references=all_references,
        hypotheses=all_hypotheses,
        n_gram=1,
    )

    bleu_2 = corpus_bleu_score(
        references=all_references,
        hypotheses=all_hypotheses,
        n_gram=2,
    )

    bleu_3 = corpus_bleu_score(
        references=all_references,
        hypotheses=all_hypotheses,
        n_gram=3,
    )

    bleu_4 = corpus_bleu_score(
        references=all_references,
        hypotheses=all_hypotheses,
        n_gram=4,
    )

    exact_match = sum(
        reference == hypothesis
        for reference, hypothesis in zip(all_references, all_hypotheses, strict=False)
    ) / len(all_references)

    return {
        "bleu_1": bleu_1,
        "bleu_2": bleu_2,
        "bleu_3": bleu_3,
        "bleu_4": bleu_4,
        "exact_match": exact_match,
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

    train_loader, test_loader, token_to_idx = create_europarl_dataloaders(
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
        vocab_size=len(token_to_idx),
        max_length=cfg.dataset.max_length,
        pad_id=token_to_idx["<pad>"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
    )

    loss_fn = nn.CrossEntropyLoss(ignore_index=token_to_idx["<pad>"])

    print(f"Experiment: {cfg.experiment.name}")
    print(f"Paper: {cfg.paper.title}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Vocabulary size: {len(token_to_idx)}")
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
            pad_id=token_to_idx["<pad>"],
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
            token_to_idx=token_to_idx,
            device=device,
            max_examples=cfg.evaluation.num_reconstruction_examples,
            max_length=cfg.dataset.max_length,
        )

        result["snr_db"] = float(snr_db)
        evaluation_results.append(result)

        print(
            f"SNR={float(snr_db):>5.1f} dB "
            f"| BLEU-1={result['bleu_1']:.4f} "
            f"| BLEU-2={result['bleu_2']:.4f} "
            f"| BLEU-3={result['bleu_3']:.4f} "
            f"| BLEU-4={result['bleu_4']:.4f} "
            f"| exact={result['exact_match']:.4f}"
        )

    checkpoint_path = checkpoint_dir / "model.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "vocab_size": len(token_to_idx),
            "pad_id": token_to_idx["<pad>"],
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
