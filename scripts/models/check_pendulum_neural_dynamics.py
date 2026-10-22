import csv
from pathlib import Path

import torch

from scripts.models.pendulum_neural_dynamics import PendulumNeuralDynamics


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIRECTORY = (
    REPOSITORY_ROOT
    / "data"
    / "datasets"
    / "pendulum"
    / "random_hold1_v001"
)

TRAIN_CSV_PATH = (
    DATASET_DIRECTORY
    / "splits"
    / "train.csv"
)

NORMALIZATION_PATH = (
    DATASET_DIRECTORY
    / "splits"
    / "normalization.npz"
)


def load_first_transition(
    csv_path: Path,
) -> dict[str, float | int]:
    """CSVの最初の状態遷移を読み込む。"""

    with csv_path.open(
        mode="r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(csv_file)
        row = next(reader)

    return {
        "trajectory_id": int(row["trajectory_id"]),
        "step": int(row["step"]),
        "theta": float(row["theta"]),
        "theta_dot": float(row["theta_dot"]),
        "torque_applied": float(row["torque_applied"]),
        "next_theta": float(row["next_theta"]),
        "next_theta_dot": float(row["next_theta_dot"]),
    }


def wrapped_angle_error(
    predicted_theta: torch.Tensor,
    target_theta: torch.Tensor,
) -> torch.Tensor:
    """角度境界を考慮した最短角度誤差を求める。"""

    return torch.atan2(
        torch.sin(predicted_theta - target_theta),
        torch.cos(predicted_theta - target_theta),
    )


def main() -> None:
    """実データ1件を未学習モデルへ入力して処理経路を確認する。"""

    if not TRAIN_CSV_PATH.exists():
        raise FileNotFoundError(
            "学習用CSVが見つかりません。\n"
            f"確認した場所: {TRAIN_CSV_PATH}"
        )

    torch.manual_seed(0)

    transition = load_first_transition(
        TRAIN_CSV_PATH
    )

    model = PendulumNeuralDynamics(
        normalization_path=NORMALIZATION_PATH,
    )

    theta = torch.tensor(
        transition["theta"],
        dtype=torch.float32,
    )
    theta_dot = torch.tensor(
        transition["theta_dot"],
        dtype=torch.float32,
    )
    torque = torch.tensor(
        transition["torque_applied"],
        dtype=torch.float32,
    )

    target_next_state = torch.tensor(
        [
            transition["next_theta"],
            transition["next_theta_dot"],
        ],
        dtype=torch.float32,
    )

    predicted_next_state = model(
        theta=theta,
        theta_dot=theta_dot,
        torque=torque,
    )

    encoded_prediction = (
        model.predict_encoded_next_state(
            theta=theta,
            theta_dot=theta_dot,
            torque=torque,
        )
    )

    angle_error = wrapped_angle_error(
        predicted_theta=predicted_next_state[0],
        target_theta=target_next_state[0],
    )

    angular_velocity_error = (
        predicted_next_state[1]
        - target_next_state[1]
    )

    unit_circle_length = torch.sqrt(
        encoded_prediction[0].square()
        + encoded_prediction[1].square()
    )

    print("Dataset               :", DATASET_DIRECTORY.name)
    print("Trajectory ID         :", transition["trajectory_id"])
    print("Step                  :", transition["step"])
    print()
    print("Current theta         :", theta.item())
    print("Current theta dot     :", theta_dot.item())
    print("Applied torque        :", torque.item())
    print()
    print("Target next state     :", target_next_state)
    print("Predicted next state  :", predicted_next_state)
    print()
    print("Wrapped angle error   :", angle_error.item())
    print(
        "Angular velocity error:",
        angular_velocity_error.item(),
    )
    print(
        "Unit circle length    :",
        unit_circle_length.item(),
    )
    print(
        "Prediction is finite  :",
        torch.isfinite(
            predicted_next_state
        ).all().item(),
    )

    print()
    print(
        "[INFO] The model is not trained yet, "
        "so prediction accuracy is not evaluated."
    )


if __name__ == "__main__":
    main()