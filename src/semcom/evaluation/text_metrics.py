import torch


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
