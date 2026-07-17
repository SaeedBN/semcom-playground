import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "semcom-matplotlib"),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    str(Path(tempfile.gettempdir()) / "semcom-cache"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_metrics(results_dir: str | Path) -> dict[str, Any]:
    results_dir = Path(results_dir)
    metrics_path = results_dir / "metrics.json"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    with metrics_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def plot_deepsc_training_metrics(
    metrics: dict[str, Any],
    output_path: str | Path,
) -> None:
    history = metrics.get("training_history", [])

    if not history:
        raise ValueError("metrics.json does not contain training_history.")

    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    token_accuracy = [item["train_token_accuracy"] for item in history]
    sequence_accuracy = [item["train_sequence_accuracy"] for item in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, train_loss, color="#1f77b4", linewidth=2)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        epochs,
        token_accuracy,
        label="Token accuracy",
        color="#2ca02c",
        linewidth=2,
    )
    axes[1].plot(
        epochs,
        sequence_accuracy,
        label="Sequence accuracy",
        color="#d62728",
        linewidth=2,
    )
    axes[1].set_title("Training Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(_figure_title(metrics, "Training Metrics"))
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_deepsc_evaluation_metrics(
    metrics: dict[str, Any],
    output_path: str | Path,
) -> None:
    evaluation_results = metrics.get("evaluation_results", [])

    if not evaluation_results:
        raise ValueError("metrics.json does not contain evaluation_results.")

    evaluation_results = sorted(
        evaluation_results,
        key=lambda item: item["snr_db"],
    )
    snr_db = [item["snr_db"] for item in evaluation_results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for key, label in [
        ("bleu_1", "BLEU-1"),
        ("bleu_2", "BLEU-2"),
        ("bleu_3", "BLEU-3"),
        ("bleu_4", "BLEU-4"),
    ]:
        axes[0].plot(
            snr_db,
            [item[key] for item in evaluation_results],
            marker="o",
            linewidth=2,
            label=label,
        )

    axes[0].set_title("BLEU vs SNR")
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("BLEU")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        snr_db,
        [item["exact_match"] for item in evaluation_results],
        marker="o",
        color="#9467bd",
        linewidth=2,
    )
    axes[1].set_title("Exact Match vs SNR")
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("Exact match")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(_figure_title(metrics, "Evaluation Metrics"))
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_deepsc_result_directory(
    results_dir: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(results_dir)
    training_path = output_dir / "training_metrics.png"
    evaluation_path = output_dir / "evaluation_metrics.png"

    plot_deepsc_training_metrics(metrics, training_path)
    plot_deepsc_evaluation_metrics(metrics, evaluation_path)

    return training_path, evaluation_path


def _figure_title(metrics: dict[str, Any], title: str) -> str:
    experiment_name = metrics.get("experiment_name", "experiment")
    channel = metrics.get("channel", {}).get("name", "channel")
    return f"{title}: {experiment_name} ({channel})"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot DeepSC training and evaluation figures from metrics.json.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        help="Directory containing metrics.json.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Directory for generated figures. Defaults to results_dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training_path, evaluation_path = plot_deepsc_result_directory(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
    )

    print(f"Saved training figure to: {training_path}")
    print(f"Saved evaluation figure to: {evaluation_path}")


if __name__ == "__main__":
    main()
