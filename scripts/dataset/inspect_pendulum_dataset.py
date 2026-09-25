import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


REQUIRED_COLUMNS = (
    "trajectory_id",
    "step",
    "theta",
    "theta_dot",
    "torque_command",
    "torque_applied",
    "next_theta",
    "next_theta_dot",
    "delta_t",
    "seed",
    "noise_std",
    "torque_hold_steps",
    "initial_theta",
    "initial_theta_dot",
)

NUMERIC_COLUMNS = (
    "theta",
    "theta_dot",
    "torque_command",
    "torque_applied",
    "next_theta",
    "next_theta_dot",
    "delta_t",
    "noise_std",
    "initial_theta",
    "initial_theta_dot",
)


def load_transitions(csv_path: Path) -> list[dict[str, str]]:
    """状態遷移CSVを読み込む。"""
    with csv_path.open(
        mode="r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(f"状態遷移データが空です: {csv_path}")

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in rows[0]
    ]

    if missing_columns:
        raise ValueError(
            "必要な列が不足しています: "
            + ", ".join(missing_columns)
        )

    return rows


def check_invalid_values(
    rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    """欠損値、NaN、無限大を検査する。"""
    errors = []

    for row_number, row in enumerate(rows, start=2):
        for column in NUMERIC_COLUMNS:
            value = row[column]

            if value == "":
                errors.append(
                    {
                        "row": row_number,
                        "column": column,
                        "value": value,
                        "reason": "missing",
                    }
                )
                continue

            try:
                number = float(value)
            except ValueError:
                errors.append(
                    {
                        "row": row_number,
                        "column": column,
                        "value": value,
                        "reason": "not_numeric",
                    }
                )
                continue

            if not math.isfinite(number):
                errors.append(
                    {
                        "row": row_number,
                        "column": column,
                        "value": value,
                        "reason": "not_finite",
                    }
                )

    return errors


def group_by_trajectory(
    rows: list[dict[str, str]],
) -> dict[int, list[dict[str, str]]]:
    """軌道IDごとに状態遷移をまとめる。"""
    trajectories: dict[int, list[dict[str, str]]] = {}

    for row in rows:
        trajectory_id = int(row["trajectory_id"])
        trajectories.setdefault(trajectory_id, []).append(row)

    for trajectory in trajectories.values():
        trajectory.sort(key=lambda row: int(row["step"]))

    return trajectories


def check_continuity(
    trajectories: dict[int, list[dict[str, str]]],
    tolerance: float = 1.0e-12,
) -> list[dict[str, object]]:
    """次状態と次行の現在状態が一致するか検査する。"""
    errors = []

    for trajectory_id, trajectory in trajectories.items():
        for current, following in zip(
            trajectory[:-1],
            trajectory[1:],
        ):
            current_step = int(current["step"])
            following_step = int(following["step"])

            if following_step != current_step + 1:
                errors.append(
                    {
                        "trajectory_id": trajectory_id,
                        "step": current_step,
                        "reason": "step_not_continuous",
                    }
                )

            theta_error = abs(
                float(current["next_theta"])
                - float(following["theta"])
            )
            theta_dot_error = abs(
                float(current["next_theta_dot"])
                - float(following["theta_dot"])
            )

            if (
                theta_error > tolerance
                or theta_dot_error > tolerance
            ):
                errors.append(
                    {
                        "trajectory_id": trajectory_id,
                        "step": current_step,
                        "reason": "state_not_continuous",
                        "theta_error": theta_error,
                        "theta_dot_error": theta_dot_error,
                    }
                )

    return errors


def check_ranges(
    rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    """状態と適用トルクの範囲を検査する。"""
    errors = []

    for row_number, row in enumerate(rows, start=2):
        theta = float(row["theta"])
        theta_dot = float(row["theta_dot"])
        torque = float(row["torque_applied"])

        if theta < -math.pi or theta > math.pi:
            errors.append(
                {
                    "row": row_number,
                    "column": "theta",
                    "value": theta,
                }
            )

        if theta_dot < -8.0 or theta_dot > 8.0:
            errors.append(
                {
                    "row": row_number,
                    "column": "theta_dot",
                    "value": theta_dot,
                }
            )

        if torque < -2.0 or torque > 2.0:
            errors.append(
                {
                    "row": row_number,
                    "column": "torque_applied",
                    "value": torque,
                }
            )

    return errors


def check_torque_holding(
    trajectories: dict[int, list[dict[str, str]]],
    tolerance: float = 1.0e-12,
) -> list[dict[str, object]]:
    """設定した保持区間内でトルクが一定か検査する。"""
    errors = []

    for trajectory_id, trajectory in trajectories.items():
        hold_steps = int(trajectory[0]["torque_hold_steps"])

        for start in range(0, len(trajectory), hold_steps):
            block = trajectory[start:start + hold_steps]

            if not block:
                continue

            reference = float(block[0]["torque_command"])

            for row in block[1:]:
                torque = float(row["torque_command"])

                if abs(torque - reference) > tolerance:
                    errors.append(
                        {
                            "trajectory_id": trajectory_id,
                            "step": int(row["step"]),
                            "reason": "torque_changed_in_hold_block",
                        }
                    )

    return errors


def calculate_summary(
    rows: list[dict[str, str]],
    trajectories: dict[int, list[dict[str, str]]],
) -> dict[str, object]:
    """データセットの基本統計を計算する。"""
    theta = [float(row["theta"]) for row in rows]
    theta_dot = [float(row["theta_dot"]) for row in rows]
    torque_command = [
        float(row["torque_command"])
        for row in rows
    ]
    torque_applied = [
        float(row["torque_applied"])
        for row in rows
    ]

    clipped_count = sum(
        abs(command) > 2.0
        for command in torque_command
    )

    return {
        "actual_transitions": len(rows),
        "number_of_trajectories": len(trajectories),
        "trajectory_lengths": {
            str(trajectory_id): len(trajectory)
            for trajectory_id, trajectory
            in trajectories.items()
        },
        "theta_min": min(theta),
        "theta_max": max(theta),
        "theta_dot_min": min(theta_dot),
        "theta_dot_max": max(theta_dot),
        "torque_command_min": min(torque_command),
        "torque_command_max": max(torque_command),
        "torque_applied_min": min(torque_applied),
        "torque_applied_max": max(torque_applied),
        "clipped_count": clipped_count,
        "clipped_rate": clipped_count / len(rows),
    }


def save_figures(
    rows: list[dict[str, str]],
    output_dir: Path,
) -> None:
    """状態と入力の分布図を保存する。"""
    theta = [float(row["theta"]) for row in rows]
    theta_dot = [float(row["theta_dot"]) for row in rows]
    torque = [float(row["torque_applied"]) for row in rows]

    figure, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(theta, bins=30)
    axes[0].set_title("Angle distribution")
    axes[0].set_xlabel("theta [rad]")
    axes[0].set_ylabel("count")

    axes[1].hist(theta_dot, bins=30)
    axes[1].set_title("Angular velocity distribution")
    axes[1].set_xlabel("theta_dot [rad/s]")
    axes[1].set_ylabel("count")

    axes[2].hist(torque, bins=30)
    axes[2].set_title("Applied torque distribution")
    axes[2].set_xlabel("torque [Nm]")
    axes[2].set_ylabel("count")

    figure.tight_layout()
    figure.savefig(
        output_dir / "distributions.png",
        dpi=150,
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(theta, theta_dot, s=8)
    axis.set_title("Visited state distribution")
    axis.set_xlabel("theta [rad]")
    axis.set_ylabel("theta_dot [rad/s]")
    figure.tight_layout()
    figure.savefig(
        output_dir / "state_distribution.png",
        dpi=150,
    )
    plt.close(figure)


def inspect_dataset(dataset_dir: Path) -> bool:
    """データセットを検査し、結果を保存する。"""
    csv_path = dataset_dir / "transitions.csv"

    rows = load_transitions(csv_path)
    trajectories = group_by_trajectory(rows)

    invalid_errors = check_invalid_values(rows)
    continuity_errors = check_continuity(trajectories)
    range_errors = check_ranges(rows)
    holding_errors = check_torque_holding(trajectories)

    summary = calculate_summary(
        rows=rows,
        trajectories=trajectories,
    )

    summary["invalid_error_count"] = len(invalid_errors)
    summary["continuity_error_count"] = len(
        continuity_errors
    )
    summary["range_error_count"] = len(range_errors)
    summary["holding_error_count"] = len(holding_errors)

    is_valid = (
        not invalid_errors
        and not continuity_errors
        and not range_errors
        and not holding_errors
    )
    summary["is_valid"] = is_valid

    inspection_path = dataset_dir / "inspection_summary.json"

    with inspection_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    save_figures(
        rows=rows,
        output_dir=dataset_dir,
    )

    print(f"Dataset: {dataset_dir.name}")
    print(f"Transitions: {len(rows)}")
    print(f"Trajectories: {len(trajectories)}")
    print(f"Invalid errors: {len(invalid_errors)}")
    print(f"Continuity errors: {len(continuity_errors)}")
    print(f"Range errors: {len(range_errors)}")
    print(f"Holding errors: {len(holding_errors)}")
    print(f"Clipped rate: {summary['clipped_rate']:.4f}")
    print(f"Valid: {is_valid}")
    print()

    return is_valid


def parse_arguments() -> argparse.Namespace:
    """コマンドライン引数を取得する。"""
    parser = argparse.ArgumentParser(
        description="Pendulumデータセットを検査する。"
    )
    parser.add_argument(
        "dataset_dirs",
        nargs="+",
        type=Path,
        help="検査するデータセットフォルダー",
    )
    return parser.parse_args()


def main() -> None:
    """指定されたデータセットを検査する。"""
    args = parse_arguments()

    inspection_results = [
        inspect_dataset(dataset_dir)
        for dataset_dir in args.dataset_dirs
    ]

    if not all(inspection_results):
        raise SystemExit(
            "検査に失敗したデータセットがあります。"
        )


if __name__ == "__main__":
    main()