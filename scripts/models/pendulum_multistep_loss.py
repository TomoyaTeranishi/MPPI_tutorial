# NNの複数ステップ予測が正解状態からどれだけずれているかを計算
import torch
from torch import nn


class PendulumMultistepLoss(nn.Module):
    """Pendulumの複数ステップ予測誤差を計算する。"""

    def __init__(
        self,
        sin_weight: float = 1.0,
        cos_weight: float = 1.0,
        angular_velocity_weight: float = 1.0,
    ) -> None:
        super().__init__()

        weights = (
            sin_weight,
            cos_weight,
            angular_velocity_weight,
        )

        if any(weight <= 0.0 for weight in weights):
            raise ValueError(
                "すべての損失重みには正の値が必要です。"
            )

        self.sin_weight = float(sin_weight)
        self.cos_weight = float(cos_weight)
        self.angular_velocity_weight = float(
            angular_velocity_weight
        )

    def forward(
        self,
        predicted_encoded_states: torch.Tensor,
        target_physical_states: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """複数ステップ予測と正解状態から損失を計算する。"""

        if predicted_encoded_states.shape[-1] != 3:
            raise ValueError(
                "predicted_encoded_statesの最後の次元は"
                "3である必要があります。"
                f"実際の形状: "
                f"{tuple(predicted_encoded_states.shape)}"
            )

        if target_physical_states.shape[-1] != 2:
            raise ValueError(
                "target_physical_statesの最後の次元は"
                "2である必要があります。"
                f"実際の形状: "
                f"{tuple(target_physical_states.shape)}"
            )

        if (
            predicted_encoded_states.shape[:-1]
            != target_physical_states.shape[:-1]
        ):
            raise ValueError(
                "予測系列と正解系列のバッチ・時間形状が"
                "一致していません。\n"
                f"prediction: "
                f"{tuple(predicted_encoded_states.shape)}\n"
                f"target: "
                f"{tuple(target_physical_states.shape)}"
            )

        target_theta = target_physical_states[..., 0]
        target_theta_dot = target_physical_states[..., 1]

        target_sin = torch.sin(target_theta)
        target_cos = torch.cos(target_theta)

        predicted_sin = predicted_encoded_states[..., 0]
        predicted_cos = predicted_encoded_states[..., 1]
        predicted_theta_dot = predicted_encoded_states[..., 2]

        sin_mse = torch.mean(
            (predicted_sin - target_sin).square()
        )

        cos_mse = torch.mean(
            (predicted_cos - target_cos).square()
        )

        angular_velocity_mse = torch.mean(
            (
                predicted_theta_dot
                - target_theta_dot
            ).square()
        )

        total_loss = (
            self.sin_weight * sin_mse
            + self.cos_weight * cos_mse
            + self.angular_velocity_weight
            * angular_velocity_mse
        )

        return {
            "loss": total_loss,
            "sin_mse": sin_mse,
            "cos_mse": cos_mse,
            "angular_velocity_mse": angular_velocity_mse,
            "sin_rmse": torch.sqrt(sin_mse),
            "cos_rmse": torch.sqrt(cos_mse),
            "angular_velocity_rmse": torch.sqrt(
                angular_velocity_mse
            ),
        }