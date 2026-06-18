import numpy as np
import torch

from semcom.utils.seed import set_seed


def test_seed_reproducibility() -> None:
    set_seed(42)
    a_torch = torch.randn(3)
    a_numpy = np.random.normal(0, 1, size=(3,))

    set_seed(42)
    b_torch = torch.randn(3)
    b_numpy = np.random.normal(0, 1, size=(3,))

    assert torch.allclose(a_torch, b_torch)
    assert np.allclose(a_numpy, b_numpy)
