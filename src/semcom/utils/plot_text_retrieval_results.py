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


def plot_text_retrieval_evaluation_metrics(
    metrics: dict[str, Any],
    output_path: str | Path,
) -> None:
    evaluation_results = metrics.get("evaluation_results", [])

    if not evaluation_results:
        raise ValueError("metrics.json does not contain evaluation_results.")

    evaluation_results = sorted(
        evaluation_results,
        key=lambda item: item["ebno_db"],
    )
    ebno_db = [item["ebno_db"] for item in evaluation_results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for key, label, color in [
        ("top_1_accuracy", "Top-1", "#1f77b4"),
        ("top_5_accuracy", "Top-5", "#2ca02c"),
        ("top_10_accuracy", "Top-10", "#d62728"),
    ]:
        values = [item[key] for item in evaluation_results]
        ci95 = [item.get(f"{key}_ci95", 0.0) for item in evaluation_results]
        axes[0].plot(
            ebno_db,
            values,
            marker="o",
            linewidth=2,
            color=color,
            label=label,
        )
        axes[0].fill_between(
            ebno_db,
            [max(0.0, value - delta) for value, delta in zip(values, ci95)],
            [min(1.0, value + delta) for value, delta in zip(values, ci95)],
            color=color,
            alpha=0.15,
        )

    axes[0].set_title("Retrieval Accuracy vs Eb/N0")
    axes[0].set_xlabel("Eb/N0 (dB)")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    cosine_values = [item["mean_cosine_similarity"] for item in evaluation_results]
    cosine_ci95 = [
        item.get("mean_cosine_similarity_ci95", 0.0) for item in evaluation_results
    ]
    axes[1].plot(
        ebno_db,
        cosine_values,
        marker="o",
        linewidth=2,
        color="#9467bd",
    )
    axes[1].fill_between(
        ebno_db,
        [max(0.0, value - delta) for value, delta in zip(cosine_values, cosine_ci95)],
        [min(1.0, value + delta) for value, delta in zip(cosine_values, cosine_ci95)],
        color="#9467bd",
        alpha=0.15,
    )
    axes[1].set_title("Mean Cosine Similarity vs Eb/N0")
    axes[1].set_xlabel("Eb/N0 (dB)")
    axes[1].set_ylabel("Cosine similarity")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(_figure_title(metrics, "Evaluation Metrics"))
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_text_retrieval_result_directory(
    results_dir: str | Path,
    output_dir: str | Path | None,
) -> Path:
    results_dir = Path(results_dir)
    output_dir = results_dir if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(results_dir)
    evaluation_path = output_dir / "evaluation_metrics.png"

    plot_text_retrieval_evaluation_metrics(metrics, evaluation_path)

    return evaluation_path


def _figure_title(metrics: dict[str, Any], title: str) -> str:
    experiment_name = metrics.get("experiment_name", "experiment")
    num_trials = metrics.get("num_trials")
    if num_trials is None:
        return f"{title}: {experiment_name}"
    return f"{title}: {experiment_name} ({num_trials} trials, 95% CI)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot text retrieval evaluation figure from metrics.json.",
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
    evaluation_path = plot_text_retrieval_result_directory(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
    )

    print(f"Saved evaluation figure to: {evaluation_path}")


if __name__ == "__main__":
    main()
