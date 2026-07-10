import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


def token_accuracy(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    pad_id: int,
) -> float:
    """Calculating Token-wise Accuracy (PADs are ignored)"""

    predictions = torch.argmax(logits, dim=-1)
    valid_mask = target_ids != pad_id

    valid_token_count = valid_mask.sum()

    if valid_token_count == 0:
        return 0.0

    correct_tokens = (predictions == target_ids) & valid_mask

    return (correct_tokens.sum() / valid_token_count).item()


def sequence_accuracy(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    pad_id: int,
) -> float:
    """Calculating sequence-wise accuracy"""

    predictions = torch.argmax(logits, dim=-1)
    valid_mask = target_ids != pad_id

    token_matches = (predictions == target_ids) | ~valid_mask
    sequence_matches = token_matches.all(dim=1)

    return sequence_matches.float().mean().item()


def predictions_to_token_ids(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=-1)


def sentence_bleu_score(
    reference: str,
    hypothesis: str,
    n_gram: int = 4,
) -> float:

    reference_tokens = reference.split()
    hypothesis_tokens = hypothesis.split()

    if not reference_tokens or not hypothesis_tokens:
        return 0.0

    if n_gram == 1:
        weights = (1.0, 0.0, 0.0, 0.0)
    elif n_gram == 2:
        weights = (0.5, 0.5, 0.0, 0.0)
    elif n_gram == 3:
        weights = (1 / 3, 1 / 3, 1 / 3, 0.0)
    elif n_gram == 4:
        weights = (0.25, 0.25, 0.25, 0.25)
    else:
        raise ValueError("n_gram must be one of {1, 2, 3, 4}.")

    smoothing = SmoothingFunction().method1

    return float(
        sentence_bleu(
            [reference_tokens],
            hypothesis_tokens,
            weights=weights,
            smoothing_function=smoothing,
        )
    )


def corpus_bleu_score(
    references: list[str],
    hypotheses: list[str],
    n_gram: int = 4,
) -> float:
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length.")

    if not references:
        return 0.0

    scores = [
        sentence_bleu_score(
            reference=reference,
            hypothesis=hypothesis,
            n_gram=n_gram,
        )
        for reference, hypothesis in zip(references, hypotheses, strict=False)
    ]

    return sum(scores) / len(scores)
