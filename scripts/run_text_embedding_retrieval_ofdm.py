import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sionna.phy.channel import RayleighBlockFading

from semcom.channels.sionna_ofdm import DigitalTextOFDMLink, IdentityChannelModel
from semcom.data.europarl import load_europarl_sentences
from semcom.utils.config import load_config_from_path
from semcom.utils.io import ensure_dir, save_config, save_json
from semcom.utils.quantization import (
    bits2integer,
    dequantize_embedding,
    integer2bits,
    quantize_embedding,
)
from semcom.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Running digital text embedding retrieval OFDM experiment.",
    )
    parser.add_argument(
        "-c",
        "--config-path",
        type=str,
        required=True,
        help="Path to experiment YAML config.",
    )
    return parser.parse_args()


def evaluate_retrieval(
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    target_indices: torch.Tensor,
    top_k: list[int],
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    similarity = torch.matmul(query_embeddings, candidate_embeddings.transpose(0, 1))
    ranked_indices = torch.argsort(similarity, dim=1, descending=True)
    target_similarity = torch.gather(
        similarity,
        dim=1,
        index=target_indices.unsqueeze(1),
    ).squeeze(1)

    results = {}
    for k in top_k:
        correct = (
            (ranked_indices[:, :k] == target_indices.unsqueeze(1))
            .any(dim=1)
            .float()
            .mean()
        )
        results[f"top_{k}_accuracy"] = correct.item()

    results["mean_cosine_similarity"] = target_similarity.mean().item()
    return results, ranked_indices, similarity


def build_examples(
    query_sentences: list[str],
    candidate_sentences: list[str],
    target_indices: torch.Tensor,
    ranked_indices: torch.Tensor,
    similarity: torch.Tensor,
    num_examples: int,
) -> list[dict[str, object]]:
    examples = []
    target_index_list = target_indices.tolist()

    for idx in range(min(num_examples, len(query_sentences))):
        ranking = ranked_indices[idx].tolist()
        target_idx = target_index_list[idx]
        target_rank = ranking.index(target_idx) + 1
        top_prediction_idx = ranking[0]

        examples.append(
            {
                "query": query_sentences[idx],
                "target": candidate_sentences[target_idx],
                "top_prediction": candidate_sentences[top_prediction_idx],
                "target_rank": target_rank,
                "top1_correct": top_prediction_idx == target_idx,
                "cosine_to_target": float(similarity[idx, target_idx].item()),
            }
        )

    return examples


def pack_bitstreams_into_blocks(
    bitstreams: torch.Tensor,
    block_length: int,
) -> tuple[torch.Tensor, int]:
    total_bits = bitstreams.shape[1]
    num_blocks = (total_bits + block_length - 1) // block_length
    padded_bits = num_blocks * block_length

    if padded_bits > total_bits:
        pad = torch.zeros(
            bitstreams.shape[0],
            padded_bits - total_bits,
            dtype=bitstreams.dtype,
            device=bitstreams.device,
        )
        bitstreams = torch.cat([bitstreams, pad], dim=1)

    return bitstreams.reshape(-1, block_length), total_bits


def unpack_blocks_into_bitstreams(
    blocks: torch.Tensor,
    num_bitstreams: int,
    total_bits: int,
) -> torch.Tensor:
    padded_bits = blocks.shape[0] * blocks.shape[1] // num_bitstreams
    bitstreams = blocks.reshape(num_bitstreams, padded_bits)

    return bitstreams[:, :total_bits]


def aggregate_trial_metrics(
    trial_metrics: list[dict[str, float]],
) -> dict[str, float]:
    if not trial_metrics:
        raise ValueError("trial_metrics must not be empty.")

    aggregated = {}
    num_trials = len(trial_metrics)
    metric_names = list(trial_metrics[0].keys())

    for metric_name in metric_names:
        values = torch.tensor(
            [metrics[metric_name] for metrics in trial_metrics],
            dtype=torch.float32,
        )
        mean = values.mean().item()
        std = values.std(unbiased=True).item() if num_trials > 1 else 0.0
        ci95 = 1.96 * std / (num_trials**0.5) if num_trials > 1 else 0.0

        aggregated[metric_name] = mean
        aggregated[f"{metric_name}_std"] = std
        aggregated[f"{metric_name}_ci95"] = ci95

    return aggregated


def main() -> None:
    args = parse_args()
    cfg = load_config_from_path(args.config_path)
    set_seed(int(cfg.experiment.seed))
    device = torch.device(str(cfg.experiment.device))
    device_str = str(device)

    if str(cfg.channel.name) != "ofdm_digital":
        raise ValueError(
            f"Unsupported channel.name={cfg.channel.name}. "
            "This script expects channel.name=ofdm_digital."
        )

    output_dir = ensure_dir(Path(cfg.experiment.output_dir))
    save_config(cfg, output_dir)

    sentences = load_europarl_sentences(
        text_path=cfg.dataset.text_path,
        max_sentences=int(cfg.dataset.max_sentences),
    )

    query_count = max(1, int(len(sentences) * float(cfg.dataset.query_fraction)))
    query_sentences = sentences[:query_count]
    candidate_sentences = sentences
    target_indices = torch.arange(query_count, device=device)

    model = SentenceTransformer(cfg.semantic.model_name, device=device_str)
    embeddings = model.encode(sentences, convert_to_tensor=True).to(
        device=device,
        dtype=torch.float32,
    )
    query_embeddings = embeddings[:query_count]
    candidate_embeddings = embeddings

    if bool(cfg.semantic.normalize):
        query_embeddings = F.normalize(query_embeddings, dim=-1)
        candidate_embeddings = F.normalize(candidate_embeddings, dim=-1)

    quantized = quantize_embedding(
        query_embeddings,
        bits_per_dim=int(cfg.quantizer.bits_per_dim),
        clip_value=float(cfg.quantizer.clip_value),
    )
    source_bits = integer2bits(
        quantized,
        bits_per_dim=int(cfg.quantizer.bits_per_dim),
    )
    packet_bits, sentence_bit_length = pack_bitstreams_into_blocks(
        source_bits,
        block_length=int(cfg.phy.coding.k),
    )

    if str(cfg.phy.channel_model.type) == "awgn":
        channel_model = IdentityChannelModel(device=device_str)
    elif str(cfg.phy.channel_model.type) == "rayleigh":
        channel_model = RayleighBlockFading(
            num_rx=1,
            num_rx_ant=1,
            num_tx=1,
            num_tx_ant=1,
            device=device_str,
        )
    else:
        raise ValueError(
            f"Unsupported phy.channel_model.type={cfg.phy.channel_model.type}. "
            "Supported options are awgn and rayleigh."
        )

    link = DigitalTextOFDMLink(
        k=int(cfg.phy.coding.k),
        n=int(cfg.phy.coding.n),
        num_bits_per_symbol=int(cfg.phy.modulation.num_bits_per_symbol),
        fft_size=int(cfg.phy.resource_grid.fft_size),
        num_ofdm_symbols=int(cfg.phy.resource_grid.num_ofdm_symbols),
        subcarrier_spacing=float(cfg.phy.resource_grid.subcarrier_spacing_hz),
        cyclic_prefix_length=int(cfg.phy.resource_grid.cyclic_prefix_length),
        pilot_pattern=str(cfg.phy.resource_grid.pilot_pattern),
        pilot_ofdm_symbol_indices=list(
            cfg.phy.resource_grid.get("pilot_ofdm_symbol_indices", [])
        )
        or None,
        dc_null=bool(cfg.phy.resource_grid.dc_null),
        num_guard_carriers=tuple(cfg.phy.resource_grid.num_guard_carriers),
        channel_model=channel_model,
        normalize_channel=bool(cfg.phy.channel_model.get("normalize_channel", True)),
        device=device,
    )

    evaluation_results = []
    ebno_db_values = list(cfg.evaluation.ebno_db_values)
    num_trials = int(cfg.evaluation.get("num_trials", 1))
    metric_order = [f"top_{k}_accuracy" for k in cfg.evaluation.top_k] + [
        "mean_cosine_similarity"
    ]

    for step_idx, ebno_db in enumerate(ebno_db_values, start=1):
        trial_metrics = []
        examples = None

        for trial_idx in range(num_trials):
            recovered_packets = link.transmit(
                bits=packet_bits,
                ebno_db=float(ebno_db),
                perfect_csi=bool(cfg.phy.estimation.perfect_csi),
            )
            recovered_bits = unpack_blocks_into_bitstreams(
                recovered_packets,
                num_bitstreams=source_bits.shape[0],
                total_bits=sentence_bit_length,
            )

            recovered_quantized = bits2integer(
                recovered_bits,
                bits_per_dim=int(cfg.quantizer.bits_per_dim),
            )
            recovered_embeddings = dequantize_embedding(
                recovered_quantized,
                bits_per_dim=int(cfg.quantizer.bits_per_dim),
                clip_value=float(cfg.quantizer.clip_value),
            )
            recovered_embeddings = recovered_embeddings.to(
                device=device, dtype=torch.float32
            )
            recovered_embeddings = F.normalize(recovered_embeddings, dim=-1)

            metrics, ranked_indices, similarity = evaluate_retrieval(
                recovered_embeddings,
                candidate_embeddings,
                target_indices=target_indices,
                top_k=list(cfg.evaluation.top_k),
            )
            trial_metrics.append(metrics)

            if examples is None:
                examples = build_examples(
                    query_sentences=query_sentences,
                    candidate_sentences=candidate_sentences,
                    target_indices=target_indices,
                    ranked_indices=ranked_indices,
                    similarity=similarity,
                    num_examples=int(cfg.evaluation.num_examples),
                )

            metric_parts = [f"{name}={metrics[name]:.4f}" for name in metric_order]
            print(
                f"[eval {step_idx}/{len(ebno_db_values)} | "
                f"trial {trial_idx + 1}/{num_trials}] "
                f"Eb/N0 = {float(ebno_db):.1f} dB | " + " | ".join(metric_parts)
            )

        aggregated_metrics = aggregate_trial_metrics(trial_metrics)
        summary_parts = [
            (
                f"{name}={aggregated_metrics[name]:.4f} "
                f"+/- {aggregated_metrics[f'{name}_ci95']:.4f}"
            )
            for name in metric_order
        ]
        print(
            f"[summary {step_idx}/{len(ebno_db_values)}] "
            f"Eb/N0 = {float(ebno_db):.1f} dB | " + " | ".join(summary_parts)
        )

        evaluation_results.append(
            {
                "ebno_db": float(ebno_db),
                "num_trials": num_trials,
                **aggregated_metrics,
                "trial_metrics": trial_metrics,
                "examples": examples,
                "example_trial_index": 0,
            }
        )

    save_json(
        {
            "experiment_name": cfg.experiment.name,
            "embedding_model": cfg.semantic.model_name,
            "num_queries": len(query_sentences),
            "num_candidates": len(candidate_sentences),
            "num_trials": num_trials,
            "evaluation_results": evaluation_results,
        },
        output_dir / "metrics.json",
    )


if __name__ == "__main__":
    main()
