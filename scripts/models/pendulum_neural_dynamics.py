#最終的に状態遷移モデルとして使えるようにする
from pathlib import Path

import torch
from torch import nn

from scripts.models.pendulum_dynamics_mlp import PendulumDynamicsMLP
from scripts.models.pendulum_normalization import PendulumNormalization


class PendulumNeuralDynamics(nn.Module):
    """MLPを用いてPendulumの次状態を予測する状態遷移モデル。"""

    def __init__(
        self,
        normalization_path: Path | str,
        hidden_size: int = 32,
        max_speed_abs: float = 8.0,
        unit_circle_epsilon: float = 1.0e-8,
    ) -> None:
        super().__init__()

        if max_speed_abs <= 0.0:
            raise ValueError(
                "max_speed_absには正の値が必要です。"
            )

        if unit_circle_epsilon <= 0.0:
            raise ValueError(
                "unit_circle_epsilonには正の値が必要です。"
            )

        self.model = PendulumDynamicsMLP(
            hidden_size=hidden_size,
        )
        self.normalization = (
            PendulumNormalization.from_npz(
                normalization_path
            )
        )

        self.max_speed_abs = float(
            max_speed_abs
        )
        self.unit_circle_epsilon = float(
            unit_circle_epsilon
        )

    def predict_encoded_from_encoded_state(
        self,
        encoded_state: torch.Tensor,
        torque: torch.Tensor,
    ) -> torch.Tensor:
        """符号化状態とトルクから次の符号化状態を予測する。"""

        if encoded_state.shape[-1] != 3:
            raise ValueError(
                "encoded_stateの最後の次元は"
                "3である必要があります。"
                f"実際の形状: {tuple(encoded_state.shape)}"
            )

        predicted_sin_input = encoded_state[..., 0]
        predicted_cos_input = encoded_state[..., 1]
        theta_dot_input = encoded_state[..., 2]

        predicted_sin_input, predicted_cos_input = (
            self.normalize_sin_cos(
                sin_theta=predicted_sin_input,
                cos_theta=predicted_cos_input,
            )
        )

        predicted_sin_input, torque = torch.broadcast_tensors(
            predicted_sin_input,
            torque,
        )

        predicted_cos_input = torch.broadcast_to(
            predicted_cos_input,
            predicted_sin_input.shape,
        )

        theta_dot_input = torch.broadcast_to(
            theta_dot_input,
            predicted_sin_input.shape,
        )

        physical_input = torch.stack(
            (
                predicted_sin_input,
                predicted_cos_input,
                theta_dot_input,
                torque,
            ),
            dim=-1,
        )

        standardized_input = (
            self.normalization.standardize_input(
                physical_input
            )
        )

        standardized_output = self.model(
            standardized_input
        )

        physical_output = (
            self.normalization.inverse_standardize_output(
                standardized_output
            )
        )

        predicted_sin = physical_output[..., 0]
        predicted_cos = physical_output[..., 1]
        predicted_theta_dot = physical_output[..., 2]

        predicted_sin, predicted_cos = (
            self.normalize_sin_cos(
                sin_theta=predicted_sin,
                cos_theta=predicted_cos,
            )
        )

        predicted_theta_dot = torch.clamp(
            predicted_theta_dot,
            min=-self.max_speed_abs,
            max=self.max_speed_abs,
        )

        return torch.stack(
            (
                predicted_sin,
                predicted_cos,
                predicted_theta_dot,
            ),
            dim=-1,
        )   

    def predict_encoded_next_state(
        self,
        theta: torch.Tensor,
        theta_dot: torch.Tensor,
        torque: torch.Tensor,
    ) -> torch.Tensor:
        """次状態を[sin(theta), cos(theta), theta_dot]で返す。"""

        theta, theta_dot, torque = torch.broadcast_tensors(
            theta,
            theta_dot,
            torque,
        )

        encoded_state = torch.stack(
            (
                torch.sin(theta),
                torch.cos(theta),
                theta_dot,
            ),
            dim=-1,
        )

        return self.predict_encoded_from_encoded_state(
            encoded_state=encoded_state,
            torque=torque,
        )

    def forward(
        self,
        theta: torch.Tensor,
        theta_dot: torch.Tensor,
        torque: torch.Tensor,
    ) -> torch.Tensor:
        """現在状態とトルクから次の[theta, theta_dot]を予測する。"""

        encoded_next_state = (
            self.predict_encoded_next_state(
                theta=theta,
                theta_dot=theta_dot,
                torque=torque,
            )
        )

        predicted_sin = encoded_next_state[..., 0]
        predicted_cos = encoded_next_state[..., 1]
        predicted_theta_dot = encoded_next_state[..., 2]

        predicted_theta = torch.atan2(
            predicted_sin,
            predicted_cos,
        )

        return torch.stack(
            (
                predicted_theta,
                predicted_theta_dot,
            ),
            dim=-1,
        )

    def normalize_sin_cos(
        self,
        sin_theta: torch.Tensor,
        cos_theta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """予測したsinとcosを単位円上へ正規化する。"""

        vector_norm = torch.sqrt(
            sin_theta.square()
            + cos_theta.square()
        )

        vector_norm = torch.clamp(
            vector_norm,
            min=self.unit_circle_epsilon,
        )

        return (
            sin_theta / vector_norm,
            cos_theta / vector_norm,
        )