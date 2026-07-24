import torch


def quantize_embedding(
    x: torch.Tensor,
    bits_per_dim: int,
    clip_value: float,
) -> torch.Tensor:
    levels = 2**bits_per_dim - 1
    x = torch.clamp(x, -clip_value, clip_value)
    x = (x + clip_value) / (2 * clip_value)
    q = torch.round(x * levels).to(torch.int64)

    return q


def dequantize_embedding(
    q: torch.Tensor,
    bits_per_dim: int,
    clip_value: float,
) -> torch.Tensor:
    levels = 2**bits_per_dim - 1
    x = q.to(torch.float32) / levels
    x = x * (2 * clip_value) - clip_value

    return x


def integer2bits(q: torch.Tensor, bits_per_dim: int) -> torch.Tensor:
    bit_positions = torch.arange(bits_per_dim - 1, -1, -1, device=q.device)
    bits = ((q.unsqueeze(-1) >> bit_positions) & 1).to(torch.float32)

    return bits.reshape(q.shape[0], -1)


def bits2integer(bits: torch.Tensor, bits_per_dim: int) -> torch.Tensor:
    reshaped_bits = bits.reshape(bits.shape[0], -1, bits_per_dim).to(torch.int64)
    weights = 2 ** torch.arange(bits_per_dim - 1, -1, -1, device=bits.device)

    return torch.sum(reshaped_bits * weights, dim=-1)
