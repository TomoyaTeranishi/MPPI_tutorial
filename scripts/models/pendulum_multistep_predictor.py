import torch
from torch import nn

from scripts.models.pendulum_neural_dynamics import PendulumNeuralDynamics


class PendulumMultistepPredictor(nn.Module):
    """PendulumのNN状態遷移モデルを連続して使用する。"""

    def __init__(
        self,
        dynamics_model: PendulumNeuralDynamics,
    ) -> None:
        super().__init__()

        self.dynamics_model = dynamics_model

    def forward(
        self,
        initial_state: torch.Tensor,
        torque_sequence: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """初期状態からトルク系列に沿って複数ステップ予測する。"""

        if initial_state.shape[-1] != 2:
            raise ValueError(
                "initial_stateの最後の次元は"
                "2である必要があります。"
                f"実際の形状: {tuple(initial_state.shape)}"
            )

        if torque_sequence.ndim < 1:
            raise ValueError(
                "torque_sequenceには少なくとも"
                "1次元が必要です。"
            )

        if (
            initial_state.shape[:-1]
            != torque_sequence.shape[:-1]
        ):
            raise ValueError(
                "initial_stateのバッチ形状と"
                "torque_sequenceのバッチ形状が一致しません。\n"
                f"initial_state: {tuple(initial_state.shape)}\n"
                f"torque_sequence: {tuple(torque_sequence.shape)}"
            )

        horizon = torque_sequence.shape[-1]

        if horizon <= 0:
            raise ValueError(
                "予測ホライゾンは1以上である必要があります。"
            )

        theta = initial_state[..., 0]
        theta_dot = initial_state[..., 1]

        encoded_state = torch.stack(
            (
                torch.sin(theta),
                torch.cos(theta),
                theta_dot,
            ),
            dim=-1,
        )

        predicted_encoded_states = []
        predicted_physical_states = []

        for step in range(horizon):
            encoded_state = (
                self.dynamics_model
                .predict_encoded_from_encoded_state(
                    encoded_state=encoded_state,
                    torque=torque_sequence[..., step],
                )
            )

            predicted_theta = torch.atan2(
                encoded_state[..., 0],
                encoded_state[..., 1],
            )

            physical_state = torch.stack(
                (
                    predicted_theta,
                    encoded_state[..., 2],
                ),
                dim=-1,
            )

            predicted_encoded_states.append(
                encoded_state
            )

            predicted_physical_states.append(
                physical_state
            )

        return {
            "encoded_states": torch.stack(
                predicted_encoded_states,
                dim=-2,
            ),
            "physical_states": torch.stack(
                predicted_physical_states,
                dim=-2,
            ),
        }