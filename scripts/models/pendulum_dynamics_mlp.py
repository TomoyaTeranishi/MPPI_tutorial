#NN本体

import torch
from torch import nn


class PendulumDynamicsMLP(nn.Module):
    """Pendulumの次状態を予測する全結合型ニューラルネットワーク。"""

    INPUT_SIZE = 4
    OUTPUT_SIZE = 3

    def __init__(
        self,
        hidden_size: int = 32,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                self.INPUT_SIZE,
                hidden_size,
            ),
            nn.Tanh(),
            nn.Linear(
                hidden_size,
                hidden_size,
            ),
            nn.Tanh(),
            nn.Linear(
                hidden_size,
                self.OUTPUT_SIZE,
            ),
        )

    def forward(
        self,
        standardized_input: torch.Tensor,
    ) -> torch.Tensor:
        """標準化済み入力から標準化空間の次状態を予測する。"""

        if standardized_input.shape[-1] != self.INPUT_SIZE:
            raise ValueError(
                "NN入力の最後の次元は"
                f"{self.INPUT_SIZE}である必要があります。"
                f"実際の形状: {tuple(standardized_input.shape)}"
            )

        return self.network(
            standardized_input
        )