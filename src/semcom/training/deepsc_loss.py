from typing import NamedTuple

import torch
from torch import nn

from semcom.models.mine import (
    mutual_information_lower_bound,
    sample_joint_and_marginal,
)


class DeepSCLossOutput(NamedTuple):
    total_loss: torch.Tensor
    ce_loss: torch.Tensor
    mi_lower_bound: torch.Tensor | None
    mi_loss: torch.Tensor | None


def masked_cross_entropy(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    pad_id: int,
    criterion: nn.Module,
) -> torch.Tensor:
    vocab_size = logits.shape[-1]

    loss = criterion(
        logits.reshape(-1, vocab_size),
        target_ids.reshape(-1),
    )

    mask = target_ids.reshape(-1) != pad_id
    loss = loss * mask.to(loss.dtype)

    return loss.mean()


def deepsc_total_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    pad_id: int,
    criterion: nn.Module,
    transmitted_symbols: torch.Tensor | None = None,
    received_symbols: torch.Tensor | None = None,
    mine_net: nn.Module | None = None,
    mi_weight: float = 0.0009,
) -> DeepSCLossOutput:

    ce_loss = masked_cross_entropy(
        logits=logits,
        target_ids=target_ids,
        pad_id=pad_id,
        criterion=criterion,
    )

    if mine_net is None:
        return DeepSCLossOutput(
            total_loss=ce_loss,
            ce_loss=ce_loss,
            mi_lower_bound=None,
            mi_loss=None,
        )

    if transmitted_symbols is None or received_symbols is None:
        raise ValueError("transmitted_symbols and received_symbols are missing.")

    joint, marginal = sample_joint_and_marginal(
        transmitted_symbols=transmitted_symbols,
        received_symbols=received_symbols,
    )

    mi_lower_bound = mutual_information_lower_bound(
        joint=joint,
        marginal=marginal,
        mine_net=mine_net,
    )

    mi_loss = -mi_lower_bound
    total_loss = ce_loss + mi_weight * mi_loss

    return DeepSCLossOutput(
        total_loss=total_loss,
        ce_loss=ce_loss,
        mi_lower_bound=mi_lower_bound,
        mi_loss=mi_loss,
    )
