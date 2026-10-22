import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.models.pendulum_multistep_predictor import (
    PendulumMultistepPredictor,
)
from scripts.models.pendulum_neural_dynamics import (
    PendulumNeuralDynamics,
)
from scripts.models.pendulum_sequence_dataset import (
    PendulumSequenceDataset,
)


def calculate_wrapped_angle_error(
    predicted_theta: torch.Tensor,
    target_theta: torch.Tensor,
) -> torch.Tensor:
    """角度境界を考慮した最短角度誤差を計算する。"""

    angle_difference = (
        predicted_theta - target_theta
    )

    return torch.atan2(
        torch.sin(angle_difference),
        torch.cos(angle_difference),
    )


def load_predictor(
    model_path: Path,
    normalization_path: Path,
    device: torch.device,
) -> tuple[PendulumMultistepPredictor, dict]:
    """保存済みチェックポイントからモデルを復元する。"""

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    config = checkpoint["config"]

    dynamics_model = PendulumNeuralDynamics(
        normalization_path=normalization_path,
        hidden_size=int(config["hidden_size"]),
    )

    predictor = PendulumMultistepPredictor(
        dynamics_model=dynamics_model,
    ).to(device)

    predictor.load_state_dict(
        checkpoint["model_state_dict"]
    )

    predictor.eval()

    return predictor, checkpoint


def evaluate_model(
    predictor: PendulumMultistepPredictor,
    data_loader: DataLoader,
    horizon: int,
    device: torch.device,
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    """評価用データに対して複数ステップ予測を行う。"""

    angle_squared_error_sum = 0.0
    angular_velocity_squared_error_sum = 0.0
    maximum_absolute_angle_error = 0.0
    maximum_absolute_angular_velocity_error = 0.0

    step_angle_squared_error_sum = np.zeros(
        horizon,
        dtype=np.float64,
    )
    step_angular_velocity_squared_error_sum = (
        np.zeros(
            horizon,
            dtype=np.float64,
        )
    )
    step_sample_count = np.zeros(
        horizon,
        dtype=np.int64,
    )

    number_of_sequences = 0
    number_of_state_predictions = 0
    number_of_invalid_sequences = 0

    with torch.no_grad():
        for batch in data_loader:
            initial_state = batch[
                "initial_state"
            ].to(device)

            torque_sequence = batch[
                "torque_sequence"
            ].to(device)

            target_states = batch[
                "target_state_sequence"
            ].to(device)

            prediction = predictor(
                initial_state=initial_state,
                torque_sequence=torque_sequence,
            )

            predicted_states = prediction[
                "physical_states"
            ]

            finite_mask = torch.isfinite(
                predicted_states
            ).all(dim=-1).all(dim=-1)

            number_of_invalid_sequences += int(
                (~finite_mask).sum().item()
            )

            if not torch.all(finite_mask):
                predicted_states = predicted_states[
                    finite_mask
                ]
                target_states = target_states[
                    finite_mask
                ]

            if predicted_states.shape[0] == 0:
                continue

            angle_error = calculate_wrapped_angle_error(
                predicted_theta=predicted_states[..., 0],
                target_theta=target_states[..., 0],
            )

            angular_velocity_error = (
                predicted_states[..., 1]
                - target_states[..., 1]
            )

            batch_size = predicted_states.shape[0]

            angle_squared_error_sum += float(
                angle_error.square().sum().item()
            )

            angular_velocity_squared_error_sum += float(
                angular_velocity_error.square().sum().item()
            )

            maximum_absolute_angle_error = max(
                maximum_absolute_angle_error,
                float(
                    angle_error.abs().max().item()
                ),
            )

            maximum_absolute_angular_velocity_error = max(
                maximum_absolute_angular_velocity_error,
                float(
                    angular_velocity_error.abs().max().item()
                ),
            )

            step_angle_squared_error_sum += (
                angle_error.square()
                .sum(dim=0)
                .cpu()
                .numpy()
            )

            step_angular_velocity_squared_error_sum += (
                angular_velocity_error.square()
                .sum(dim=0)
                .cpu()
                .numpy()
            )

            step_sample_count += batch_size

            number_of_sequences += batch_size
            number_of_state_predictions += (
                batch_size * horizon
            )

    if number_of_state_predictions == 0:
        raise ValueError(
            "有効な予測結果がありません。"
        )

    angle_rmse = np.sqrt(
        angle_squared_error_sum
        / number_of_state_predictions
    )

    angular_velocity_rmse = np.sqrt(
        angular_velocity_squared_error_sum
        / number_of_state_predictions
    )

    per_step_results = []

    for step_index in range(horizon):
        angle_step_rmse = np.sqrt(
            step_angle_squared_error_sum[step_index]
            / step_sample_count[step_index]
        )

        angular_velocity_step_rmse = np.sqrt(
            step_angular_velocity_squared_error_sum[
                step_index
            ]
            / step_sample_count[step_index]
        )

        per_step_results.append(
            {
                "prediction_step": step_index + 1,
                "prediction_time_s": (
                    (step_index + 1) * 0.02
                ),
                "angle_rmse_rad": float(
                    angle_step_rmse
                ),
                "angular_velocity_rmse_rad_s": float(
                    angular_velocity_step_rmse
                ),
            }
        )

    summary = {
        "horizon": horizon,
        "prediction_delta_t": 0.02,
        "prediction_time_s": horizon * 0.02,
        "number_of_sequences": (
            number_of_sequences
        ),
        "number_of_state_predictions": (
            number_of_state_predictions
        ),
        "number_of_invalid_sequences": (
            number_of_invalid_sequences
        ),
        "angle_rmse_rad": float(angle_rmse),
        "angular_velocity_rmse_rad_s": float(
            angular_velocity_rmse
        ),
        "maximum_absolute_angle_error_rad": float(
            maximum_absolute_angle_error
        ),
        "maximum_absolute_angular_velocity_error_rad_s": float(
            maximum_absolute_angular_velocity_error
        ),
    }

    return summary, per_step_results


def save_results(
    output_directory: Path,
    summary: dict,
    per_step_results: list[dict],
) -> None:
    """評価結果をJSONとCSVへ保存する。"""

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        output_directory / "evaluation_summary.json"
    )

    with summary_path.open(
        mode="w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            summary,
            json_file,
            ensure_ascii=False,
            indent=2,
        )

    per_step_path = (
        output_directory / "per_step_rmse.csv"
    )

    with per_step_path.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(
                per_step_results[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(per_step_results)

    print(
        f"[INFO] Saved summary: {summary_path}"
    )
    print(
        f"[INFO] Saved per-step results: "
        f"{per_step_path}"
    )


def parse_arguments() -> argparse.Namespace:
    """コマンドライン引数を読み込む。"""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Pendulum multistep "
            "prediction model."
        )
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Pendulumデータセットフォルダー",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="評価するbest_model.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="評価結果の保存先",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=50,
        help="評価する予測ステップ数。既定値: 50",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="評価時のバッチサイズ。既定値: 64",
    )

    return parser.parse_args()


def main() -> None:
    """評価用データで複数ステップ予測性能を確認する。"""

    arguments = parse_arguments()

    if arguments.horizon <= 0:
        raise ValueError(
            "horizonには正の整数が必要です。"
        )

    split_directory = (
        arguments.dataset_dir / "splits"
    )

    test_csv_path = (
        split_directory / "test.csv"
    )

    normalization_path = (
        split_directory / "normalization.npz"
    )

    device = torch.device("cpu")

    predictor, checkpoint = load_predictor(
        model_path=arguments.model_path,
        normalization_path=normalization_path,
        device=device,
    )

    test_dataset = PendulumSequenceDataset(
        csv_path=test_csv_path,
        horizon=arguments.horizon,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=0,
    )

    summary, per_step_results = evaluate_model(
        predictor=predictor,
        data_loader=test_loader,
        horizon=arguments.horizon,
        device=device,
    )

    summary.update(
        {
            "dataset_directory": str(
                arguments.dataset_dir
            ),
            "model_path": str(
                arguments.model_path
            ),
            "model_epoch": int(
                checkpoint["epoch"]
            ),
            "model_validation_loss": float(
                checkpoint["validation_loss"]
            ),
        }
    )

    save_results(
        output_directory=arguments.output_dir,
        summary=summary,
        per_step_results=per_step_results,
    )

    print()
    print("[RESULT] Multistep prediction evaluation")
    print(
        "Model epoch                    : "
        f"{summary['model_epoch']}"
    )
    print(
        "Horizon                        : "
        f"{summary['horizon']} steps"
    )
    print(
        "Prediction time                : "
        f"{summary['prediction_time_s']:.2f} s"
    )
    print(
        "Number of sequences            : "
        f"{summary['number_of_sequences']}"
    )
    print(
        "Invalid sequences              : "
        f"{summary['number_of_invalid_sequences']}"
    )
    print(
        "Angle RMSE                     : "
        f"{summary['angle_rmse_rad']:.6f} rad"
    )
    print(
        "Angular velocity RMSE          : "
        f"{summary['angular_velocity_rmse_rad_s']:.6f} rad/s"
    )
    print(
        "Maximum absolute angle error   : "
        f"{summary['maximum_absolute_angle_error_rad']:.6f} rad"
    )
    print(
        "Maximum angular velocity error : "
        f"{summary['maximum_absolute_angular_velocity_error_rad_s']:.6f} rad/s"
    )


if __name__ == "__main__":
    main()