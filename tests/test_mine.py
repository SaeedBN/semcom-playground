import torch

from semcom.models.mine import (
    Mine,
    mutual_information_lower_bound,
    sample_joint_and_marginal,
)


def test_mine_output_shape() -> None:
    mine = Mine(in_dim=2, hidden_size=10)

    inputs = torch.randn(32, 2)
    outputs = mine(inputs)

    assert outputs.shape == (32, 1)


def test_sample_joint_and_marginal_shapes() -> None:
    tx = torch.randn(4, 8, 16)
    rx = torch.randn(4, 8, 16)

    joint, marginal = sample_joint_and_marginal(
        transmitted_symbols=tx,
        received_symbols=rx,
    )

    assert joint.shape[1] == 2
    assert marginal.shape[1] == 2
    assert joint.shape == marginal.shape


def test_mutual_information_lower_bound_is_scalar() -> None:
    mine = Mine(in_dim=2, hidden_size=10)

    joint = torch.randn(32, 2)
    marginal = torch.randn(32, 2)

    mi = mutual_information_lower_bound(
        joint=joint,
        marginal=marginal,
        mine_net=mine,
    )

    assert mi.ndim == 0
