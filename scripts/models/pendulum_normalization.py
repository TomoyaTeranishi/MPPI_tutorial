#保存した平均・標準化を使って，入力の標準化と出力の逆標準化を行うモジュール
from pathlib import Path

import numpy as np
import torch
from torch import nn


class PendulumNormalization(nn.Module):
    """Pendulum NN用の標準化・逆標準化処理。"""

    INPUT_NAMES = (
        "sin_theta",
        "cos_theta",
        "theta_dot",
        "torque_applied",
    )

    OUTPUT_NAMES = (
        "sin_next_theta",
        "cos_next_theta",
        "next_theta_dot",
    )

    def __init__(
        self,
        input_mean: torch.Tensor,
        input_std: torch.Tensor,
        output_mean: torch.Tensor,
        output_std: torch.Tensor,
    ) -> None:
        super().__init__()

        self._validate_shape(
            tensor=input_mean,
            expected_size=4,
            tensor_name="input_mean",
        )
        self._validate_shape(
            tensor=input_std,
            expected_size=4,
            tensor_name="input_std",
        )
        self._validate_shape(
            tensor=output_mean,
            expected_size=3,
            tensor_name="output_mean",
        )
        self._validate_shape(
            tensor=output_std,
            expected_size=3,
            tensor_name="output_std",
        )

        if torch.any(input_std <= 0.0):
            raise ValueError(
                "input_stdには正の値が必要です。"
            )

        if torch.any(output_std <= 0.0):
            raise ValueError(
                "output_stdには正の値が必要です。"
            )

        # モデルと一緒にCPU・GPU間を移動できるようbufferとして保持する
        self.register_buffer(
            "input_mean",
            input_mean.to(dtype=torch.float32),
        )
        self.register_buffer(
            "input_std",
            input_std.to(dtype=torch.float32),
        )
        self.register_buffer(
            "output_mean",
            output_mean.to(dtype=torch.float32),
        )
        self.register_buffer(
            "output_std",
            output_std.to(dtype=torch.float32),
        )

    @classmethod
    def from_npz(
        cls,
        normalization_path: Path | str,
    ) -> "PendulumNormalization":
        """NPZから標準化情報を読み込む。"""

        normalization_path = Path(
            normalization_path
        )

        if not normalization_path.exists():
            raise FileNotFoundError(
                "標準化情報が見つかりません。\n"
                f"確認した場所: {normalization_path}"
            )

        with np.load(normalization_path) as data:
            required_keys = {
                "input_names",
                "input_mean",
                "input_std",
                "output_names",
                "output_mean",
                "output_std",
            }

            missing_keys = (
                required_keys - set(data.files)
            )

            if missing_keys:
                raise KeyError(
                    "NPZに必要なキーがありません: "
                    + ", ".join(sorted(missing_keys))
                )

            input_names = tuple(
                data["input_names"].tolist()
            )
            output_names = tuple(
                data["output_names"].tolist()
            )

            if input_names != cls.INPUT_NAMES:
                raise ValueError(
                    "入力特徴量の順序が想定と異なります。\n"
                    f"想定: {cls.INPUT_NAMES}\n"
                    f"実際: {input_names}"
                )

            if output_names != cls.OUTPUT_NAMES:
                raise ValueError(
                    "出力特徴量の順序が想定と異なります。\n"
                    f"想定: {cls.OUTPUT_NAMES}\n"
                    f"実際: {output_names}"
                )

            input_mean = torch.from_numpy(
                data["input_mean"].copy()
            )
            input_std = torch.from_numpy(
                data["input_std"].copy()
            )
            output_mean = torch.from_numpy(
                data["output_mean"].copy()
            )
            output_std = torch.from_numpy(
                data["output_std"].copy()
            )

        return cls(
            input_mean=input_mean,
            input_std=input_std,
            output_mean=output_mean,
            output_std=output_std,
        )

    def standardize_input(
        self,
        physical_input: torch.Tensor,
    ) -> torch.Tensor:
        """物理量の入力4項目を標準化する。"""

        self._validate_last_dimension(
            tensor=physical_input,
            expected_size=4,
            tensor_name="physical_input",
        )

        return (
            physical_input - self.input_mean
        ) / self.input_std

    def inverse_standardize_input(
        self,
        standardized_input: torch.Tensor,
    ) -> torch.Tensor:
        """標準化入力を元の物理量へ戻す。"""

        self._validate_last_dimension(
            tensor=standardized_input,
            expected_size=4,
            tensor_name="standardized_input",
        )

        return (
            standardized_input * self.input_std
            + self.input_mean
        )

    def standardize_output(
        self,
        physical_output: torch.Tensor,
    ) -> torch.Tensor:
        """物理量の正解出力3項目を標準化する。"""

        self._validate_last_dimension(
            tensor=physical_output,
            expected_size=3,
            tensor_name="physical_output",
        )

        return (
            physical_output - self.output_mean
        ) / self.output_std

    def inverse_standardize_output(
        self,
        standardized_output: torch.Tensor,
    ) -> torch.Tensor:
        """NNの標準化出力を元の物理量へ戻す。"""

        self._validate_last_dimension(
            tensor=standardized_output,
            expected_size=3,
            tensor_name="standardized_output",
        )

        return (
            standardized_output * self.output_std
            + self.output_mean
        )

    @staticmethod
    def _validate_shape(
        tensor: torch.Tensor,
        expected_size: int,
        tensor_name: str,
    ) -> None:
        """平均・標準偏差の配列形状を確認する。"""

        if tensor.ndim != 1 or tensor.shape[0] != expected_size:
            raise ValueError(
                f"{tensor_name}は形状"
                f"({expected_size},)である必要があります。"
                f"実際の形状: {tuple(tensor.shape)}"
            )

    @staticmethod
    def _validate_last_dimension(
        tensor: torch.Tensor,
        expected_size: int,
        tensor_name: str,
    ) -> None:
        """入力されたTensorの最後の次元を確認する。"""

        if tensor.shape[-1] != expected_size:
            raise ValueError(
                f"{tensor_name}の最後の次元は"
                f"{expected_size}である必要があります。"
                f"実際の形状: {tuple(tensor.shape)}"
            )