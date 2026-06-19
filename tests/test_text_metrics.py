import torch

from semcom.evaluation.text_metrics import (
    predictions_to_token_ids,
    sequence_accuracy,
    token_accuracy,
)


def test_predictions_to_token_ids() -> None:
    logits = torch.tensor(
        [
            [
                [0.1, 0.9, 0.0],
                [0.8, 0.1, 0.1],
            ]
        ]
    )

    predictions = predictions_to_token_ids(logits)

    assert predictions.tolist() == [[1, 0]]


def test_token_accuracy_ignores_padding() -> None:
    logits = torch.tensor(
        [
            [
                [0.1, 0.9, 0.0],
                [0.8, 0.1, 0.1],
                [0.2, 0.7, 0.1],
            ]
        ]
    )

    target_ids = torch.tensor([[1, 0, 2]])
    pad_id = 0

    accuracy = token_accuracy(
        logits=logits,
        target_ids=target_ids,
        pad_id=pad_id,
    )

    # only one out of two non-PAD token is correct
    assert accuracy == 0.5


def test_sequence_accuracy_ignores_padding() -> None:
    logits = torch.tensor(
        [
            [
                [0.1, 0.9, 0.0],
                [0.8, 0.1, 0.1],
                [0.2, 0.1, 0.7],
            ],
            [
                [0.1, 0.9, 0.0],
                [0.8, 0.1, 0.1],
                [0.2, 0.7, 0.1],
            ],
        ]
    )

    target_ids = torch.tensor(
        [
            [1, 0, 2],
            [1, 0, 2],
        ]
    )

    pad_id = 0

    accuracy = sequence_accuracy(
        logits=logits,
        target_ids=target_ids,
        pad_id=pad_id,
    )

    # 1 sequence correct, the other one is not
    assert accuracy == 0.5


def test_token_accuracy_returns_zero_when_only_padding() -> None:
    logits = torch.randn(2, 4, 10)
    target_ids = torch.zeros(2, 4, dtype=torch.long)

    accuracy = token_accuracy(
        logits=logits,
        target_ids=target_ids,
        pad_id=0,
    )

    assert accuracy == 0.0
