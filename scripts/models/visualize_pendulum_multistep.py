import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.models.evaluate_pendulum_multistep import (
    calculate_wrapped_angle_error,
    load_predictor,
)
from scripts.models.pendulum_sequence_dataset import (
    PendulumSequenceDataset,
)


def collect_sequence_results(
    predictor: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> list[dict]:
    """各予測区間の予測値・正解値・RMSEを収集する."""

    sequence_results = []

    with torch.no_grad():
        for batch in data_loader:
            initial_state = batch["initial_state"].to(device)
            torque_sequence = batch["torque_sequence"].to(device)
            target_states = batch["target_state_sequence"].to(device)

            prediction = predictor(
                initial_state=initial_state,
                torque_sequence=torque_sequence,
            )
            predicted_states = prediction["physical_states"]

            angle_error = calculate_wrapped_angle_error(
                predicted_theta=predicted_states[..., 0],
                target_theta=target_states[..., 0],
            )
            angular_velocity_error = (
                predicted_states[..., 1] - target_states[..., 1]
            )

            angle_rmse = torch.sqrt(
                torch.mean(angle_error.square(), dim=1)
            )
            angular_velocity_rmse = torch.sqrt(
                torch.mean(
                    angular_velocity_error.square(),
                    dim=1,
                )
            )

            for batch_index in range(initial_state.shape[0]):
                sequence_results.append(
                    {
                        "trajectory_id": int(
                            batch["trajectory_id"][batch_index].item()
                        ),
                        "start_step": int(
                            batch["start_step"][batch_index].item()
                        ),
                        "initial_state": (
                            initial_state[batch_index].cpu().numpy()
                        ),
                        "torque_sequence": (
                            torque_sequence[batch_index].cpu().numpy()
                        ),
                        "target_states": (
                            target_states[batch_index].cpu().numpy()
                        ),
                        "predicted_states": (
                            predicted_states[batch_index].cpu().numpy()
                        ),
                        "angle_error": (
                            angle_error[batch_index].cpu().numpy()
                        ),
                        "angular_velocity_error": (
                            angular_velocity_error[
                                batch_index
                            ].cpu().numpy()
                        ),
                        "angle_rmse": float(
                            angle_rmse[batch_index].item()
                        ),
                        "angular_velocity_rmse": float(
                            angular_velocity_rmse[
                                batch_index
                            ].item()
                        ),
                    }
                )

    return sequence_results


def select_representative_sequence(
    sequence_results: list[dict],
) -> dict:
    """誤差が中央値に近い予測区間を代表例として選択する。"""

    angle_rmse_values = np.array(
        [
            result["angle_rmse"]
            for result in sequence_results
        ],
        dtype=float,
    )
    angular_velocity_rmse_values = np.array(
        [
            result["angular_velocity_rmse"]
            for result in sequence_results
        ],
        dtype=float,
    )

    angle_scale = max(
        float(np.median(angle_rmse_values)),
        1.0e-12,
    )
    angular_velocity_scale = max(
        float(np.median(angular_velocity_rmse_values)),
        1.0e-12,
    )

    combined_scores = (
        angle_rmse_values / angle_scale
        + angular_velocity_rmse_values
        / angular_velocity_scale
    )

    median_score = float(np.median(combined_scores))
    representative_index = int(
        np.argmin(np.abs(combined_scores - median_score))
    )

    representative = sequence_results[representative_index].copy()
    representative["selection_index"] = representative_index
    representative["combined_score"] = float(
        combined_scores[representative_index]
    )
    representative["median_combined_score"] = median_score

    return representative


def plot_representative_sequence(
    representative: dict,
    delta_t: float,
    output_path: Path,
    dataset_name: str,
) -> None:
    """代表区間の正解軌道とNN予測軌道を描画する。"""

    target_states = representative["target_states"]
    predicted_states = representative["predicted_states"]
    angle_error = representative["angle_error"]
    angular_velocity_error = representative[
        "angular_velocity_error"
    ]

    horizon = target_states.shape[0]
    prediction_time = (
        np.arange(1, horizon + 1, dtype=float) * delta_t
    )

    target_theta_unwrapped = np.unwrap(
        target_states[:, 0]
    )

    # 角度境界をまたいでも線が不自然に飛ばないよう、
    # 正解角度に最短角度誤差を加えて予測角度を表示する
    predicted_theta_unwrapped = (
        target_theta_unwrapped + angle_error
    )

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(10, 10),
        sharex=True,
    )

    axes[0].plot(
        prediction_time,
        target_theta_unwrapped,
        label="Physics model",
        linewidth=2.0,
    )
    axes[0].plot(
        prediction_time,
        predicted_theta_unwrapped,
        label="Neural Network",
        linewidth=2.0,
    )
    axes[0].set_ylabel("Angle [rad]")
    axes[0].set_title(
        f"{dataset_name}: Representative multistep prediction"
    )
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(
        prediction_time,
        target_states[:, 1],
        label="Physics model",
        linewidth=2.0,
    )
    axes[1].plot(
        prediction_time,
        predicted_states[:, 1],
        label="Neural Network",
        linewidth=2.0,
    )
    axes[1].set_ylabel("Angular velocity [rad/s]")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(
        prediction_time,
        angle_error,
        label="Angle error [rad]",
        linewidth=2.0,
    )
    axes[2].plot(
        prediction_time,
        angular_velocity_error,
        label="Angular velocity error [rad/s]",
        linewidth=2.0,
    )
    axes[2].axhline(
        0.0,
        color="black",
        linewidth=1.0,
    )
    axes[2].set_xlabel("Prediction time [s]")
    axes[2].set_ylabel("Prediction error")
    axes[2].grid(True)
    axes[2].legend()

    figure.suptitle(
        "Trajectory ID "
        f"{representative['trajectory_id']}, "
        "start step "
        f"{representative['start_step']}, "
        "angle RMSE "
        f"{representative['angle_rmse']:.4f} rad, "
        "angular velocity RMSE "
        f"{representative['angular_velocity_rmse']:.4f} rad/s"
    )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def load_per_step_rmse(
    csv_path: Path,
) -> dict[str, np.ndarray]:
    """ステップ別RMSEを読み込む。"""

    with csv_path.open(
        mode="r",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        rows = list(csv.DictReader(csv_file))

    return {
        "prediction_time_s": np.array(
            [
                float(row["prediction_time_s"])
                for row in rows
            ],
            dtype=float,
        ),
        "angle_rmse_rad": np.array(
            [
                float(row["angle_rmse_rad"])
                for row in rows
            ],
            dtype=float,
        ),
        "angular_velocity_rmse_rad_s": np.array(
            [
                float(
                    row["angular_velocity_rmse_rad_s"]
                )
                for row in rows
            ],
            dtype=float,
        ),
    }


def plot_per_step_rmse(
    per_step_results: dict[str, np.ndarray],
    output_path: Path,
    dataset_name: str,
) -> None:
    """予測時間に対するステップ別RMSEを描画する。"""

    prediction_time = per_step_results[
        "prediction_time_s"
    ]

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(9, 8),
        sharex=True,
    )

    axes[0].plot(
        prediction_time,
        per_step_results["angle_rmse_rad"],
        linewidth=2.0,
    )
    axes[0].set_ylabel("Angle RMSE [rad]")
    axes[0].set_title(
        f"{dataset_name}: RMSE by prediction time"
    )
    axes[0].grid(True)

    axes[1].plot(
        prediction_time,
        per_step_results[
            "angular_velocity_rmse_rad_s"
        ],
        linewidth=2.0,
    )
    axes[1].set_xlabel("Prediction time [s]")
    axes[1].set_ylabel(
        "Angular velocity RMSE [rad/s]"
    )
    axes[1].grid(True)

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_representative_metadata(
    representative: dict,
    output_path: Path,
) -> None:
    """選択した代表区間の情報を保存する。"""

    metadata = {
        "selection_method": (
            "sequence closest to the median combined "
            "angle and angular velocity RMSE score"
        ),
        "selection_index": representative[
            "selection_index"
        ],
        "trajectory_id": representative[
            "trajectory_id"
        ],
        "start_step": representative["start_step"],
        "angle_rmse_rad": representative["angle_rmse"],
        "angular_velocity_rmse_rad_s": representative[
            "angular_velocity_rmse"
        ],
        "combined_score": representative[
            "combined_score"
        ],
        "median_combined_score": representative[
            "median_combined_score"
        ],
    }

    with output_path.open(
        mode="w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            metadata,
            json_file,
            ensure_ascii=False,
            indent=2,
        )


def parse_arguments() -> argparse.Namespace:
    """コマンドライン引数を読み込む。"""

    parser = argparse.ArgumentParser(
        description=(
            "Visualize representative Pendulum "
            "multistep prediction."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--delta-t",
        type=float,
        default=0.02,
    )

    return parser.parse_args()


def main() -> None:
    """代表軌道とステップ別RMSEを可視化する。"""

    arguments = parse_arguments()

    split_directory = arguments.dataset_dir / "splits"
    test_csv_path = split_directory / "test.csv"
    normalization_path = (
        split_directory / "normalization.npz"
    )
    per_step_rmse_path = (
        arguments.evaluation_dir / "per_step_rmse.csv"
    )

    required_paths = (
        test_csv_path,
        normalization_path,
        arguments.model_path,
        per_step_rmse_path,
    )

    for required_path in required_paths:
        if not required_path.exists():
            raise FileNotFoundError(
                "必要なファイルが見つかりません。\n"
                f"確認した場所: {required_path}"
            )

    arguments.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device("cpu")

    predictor, _ = load_predictor(
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

    sequence_results = collect_sequence_results(
        predictor=predictor,
        data_loader=test_loader,
        device=device,
    )

    representative = select_representative_sequence(
        sequence_results
    )

    plot_representative_sequence(
        representative=representative,
        delta_t=arguments.delta_t,
        output_path=(
            arguments.output_dir
            / "representative_prediction.png"
        ),
        dataset_name=arguments.dataset_dir.name,
    )

    per_step_results = load_per_step_rmse(
        per_step_rmse_path
    )

    plot_per_step_rmse(
        per_step_results=per_step_results,
        output_path=(
            arguments.output_dir
            / "per_step_rmse.png"
        ),
        dataset_name=arguments.dataset_dir.name,
    )

    save_representative_metadata(
        representative=representative,
        output_path=(
            arguments.output_dir
            / "representative_metadata.json"
        ),
    )

    print("[INFO] Visualization completed.")
    print(
        "Representative trajectory ID : "
        f"{representative['trajectory_id']}"
    )
    print(
        "Representative start step    : "
        f"{representative['start_step']}"
    )
    print(
        "Representative angle RMSE    : "
        f"{representative['angle_rmse']:.6f} rad"
    )
    print(
        "Representative velocity RMSE : "
        f"{representative['angular_velocity_rmse']:.6f} rad/s"
    )
    print(
        f"Output directory              : "
        f"{arguments.output_dir}"
    )


if __name__ == "__main__":
    main()