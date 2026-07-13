import torch
from torch import nn


class Mine(nn.Module):
    def __init__(
        self,
        in_dim: int = 2,
        hidden_size: int = 10,
    ) -> None:
        super().__init__()

        self.dense1 = self._custom_linear(in_dim, hidden_size)
        self.dense2 = self._custom_linear(hidden_size, hidden_size)
        self.dense3 = self._custom_linear(hidden_size, 1)

        self.relu = nn.ReLU()

    @staticmethod
    def _custom_linear(
        in_dim: int,
        out_dim: int,
        bias: bool = True,
    ) -> nn.Linear:
        layer = nn.Linear(in_dim, out_dim, bias=bias)

        with torch.no_grad():
            layer.weight.normal_(mean=0.0, std=0.02)

            if bias:
                layer.bias.zero_()

        return layer

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.dense1(inputs))
        x = self.relu(self.dense2(x))
        return self.dense3(x)


def sample_joint_and_marginal(
    transmitted_symbols: torch.Tensor,
    received_symbols: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:

    tx = transmitted_symbols.reshape(-1, 1)
    rx = received_symbols.reshape(-1, 1)

    num_samples = min(tx.shape[0], rx.shape[0])

    if num_samples < 2:
        raise ValueError("Need at least two flattened samples for MINE.")

    if num_samples % 2 != 0:
        num_samples -= 1

    tx = tx[:num_samples]
    rx = rx[:num_samples]

    half = num_samples // 2

    tx_1 = tx[:half]
    rx_1 = rx[:half]
    rx_2 = rx[half:]

    joint = torch.cat([tx_1, rx_1], dim=1)
    marginal = torch.cat([tx_1, rx_2], dim=1)

    return joint, marginal


def mutual_information_lower_bound(
    joint: torch.Tensor,
    marginal: torch.Tensor,
    mine_net: nn.Module,
) -> torch.Tensor:

    joint_score = mine_net(joint)
    marginal_score = mine_net(marginal)

    return torch.mean(joint_score) - torch.log(
        torch.mean(torch.exp(marginal_score)) + 1e-8
    )


def train_mi_one_batch(
    model: nn.Module,
    mine_net: nn.Module,
    channel: nn.Module,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    gradient_clip_norm: float,
    device: torch.device,
) -> float:

    model.eval()
    mine_net.train()

    optimizer.zero_grad()

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    with torch.no_grad():
        (
            _recovered_memory,
            _source_key_padding_mask,
            transmitted_symbols,
            received_symbols,
        ) = model.encode_channel(
            input_ids=input_ids,
            attention_mask=attention_mask,
            channel=channel,
            return_tx_rx_symbols=True,
        )

    joint, marginal = sample_joint_and_marginal(
        transmitted_symbols=transmitted_symbols.detach(),
        received_symbols=received_symbols.detach(),
    )

    mi_lower_bound = mutual_information_lower_bound(
        joint=joint,
        marginal=marginal,
        mine_net=mine_net,
    )

    mi_loss = -mi_lower_bound
    mi_loss.backward()

    torch.nn.utils.clip_grad_norm_(
        mine_net.parameters(),
        max_norm=gradient_clip_norm,
    )

    optimizer.step()

    return float(mi_lower_bound.detach().cpu())
