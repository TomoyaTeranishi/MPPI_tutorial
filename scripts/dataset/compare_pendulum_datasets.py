import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_rows(dataset_dir: Path) -> list[dict[str, str]]:
    """状態遷移CSVを読み込む。"""
    csv_path = dataset_dir / "transitions.csv"

    with csv_path.open(
        mode="r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(f"データが空です: {csv_path}")

    return rows


def extract_values(
    rows: list[dict[str, str]],
) -> dict[str, np.ndarray]:
    """比較に使用する数値をNumPy配列へ変換する。"""
    return {
        "theta": np.array(
            [float(row["theta"]) for row in rows],
            dtype=float,
        ),
        "theta_dot": np.array(
            [float(row["theta_dot"]) for row in rows],
            dtype=float,
        ),
        "torque_command": np.array(
            [float(row["torque_command"]) for row in rows],
            dtype=float,
        ),
        "torque_applied": np.array(
            [float(row["torque_applied"]) for row in rows],
            dtype=float,
        ),
    }


def calculate_statistics(
    values: dict[str, np.ndarray],
) -> dict[str, float | int]:
    """データセットの比較用統計量を計算する。"""
    theta = values["theta"]
    theta_dot = values["theta_dot"]
    torque_command = values["torque_command"]
    torque_applied = values["torque_applied"]

    clipped_mask = np.abs(torque_command) > 2.0

    # 角度と角速度が占める長方形領域の大きさ
    state_range_area = (
        (theta.max() - theta.min())
        * (theta_dot.max() - theta_dot.min())
    )

    return {
        "number_of_transitions": int(theta.size),
        "theta_min": float(theta.min()),
        "theta_max": float(theta.max()),
        "theta_range": float(np.ptp(theta)),
        "theta_dot_min": float(theta_dot.min()),
        "theta_dot_max": float(theta_dot.max()),
        "theta_dot_range": float(np.ptp(theta_dot)),
        "torque_applied_mean": float(torque_applied.mean()),
        "torque_applied_std": float(torque_applied.std()),
        "torque_applied_min": float(torque_applied.min()),
        "torque_applied_max": float(torque_applied.max()),
        "clipped_count": int(clipped_mask.sum()),
        "clipped_rate": float(clipped_mask.mean()),
        "state_range_area": float(state_range_area),
    }


def save_distribution_comparison(
    datasets: dict[str, dict[str, np.ndarray]],
    output_dir: Path,
) -> None:
    """角度・角速度・トルクの分布を同じ軸で比較する。"""
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))

    for label, values in datasets.items():
        axes[0].hist(
            values["theta"],
            bins=30,
            alpha=0.5,
            label=label,
        )
        axes[1].hist(
            values["theta_dot"],
            bins=30,
            alpha=0.5,
            label=label,
        )
        axes[2].hist(
            values["torque_applied"],
            bins=30,
            alpha=0.5,
            label=label,
        )

    axes[0].set_title("Angle distribution")
    axes[0].set_xlabel("theta [rad]")
    axes[0].set_ylabel("count")

    axes[1].set_title("Angular velocity distribution")
    axes[1].set_xlabel("theta_dot [rad/s]")
    axes[1].set_ylabel("count")

    axes[2].set_title("Applied torque distribution")
    axes[2].set_xlabel("torque [Nm]")
    axes[2].set_ylabel("count")

    for axis in axes:
        axis.legend()

    figure.tight_layout()
    figure.savefig(
        output_dir / "distribution_comparison.png",
        dpi=150,
    )
    plt.close(figure)


def save_state_space_comparison(
    datasets: dict[str, dict[str, np.ndarray]],
    output_dir: Path,
) -> None:
    """角度と角速度の状態空間を比較する。"""
    figure, axis = plt.subplots(figsize=(7, 6))

    for label, values in datasets.items():
        axis.scatter(
            values["theta"],
            values["theta_dot"],
            s=12,
            alpha=0.6,
            label=label,
        )

    axis.set_title("Visited state comparison")
    axis.set_xlabel("theta [rad]")
    axis.set_ylabel("theta_dot [rad/s]")
    axis.legend()
    axis.grid()

    figure.tight_layout()
    figure.savefig(
        output_dir / "state_space_comparison.png",
        dpi=150,
    )
    plt.close(figure)


def save_time_series_comparison(
    datasets: dict[str, dict[str, np.ndarray]],
    output_dir: Path,
    display_steps: int = 100,
) -> None:
    """先頭軌道の状態と入力の時系列を比較する。"""
    figure, axes = plt.subplots(3, 1, figsize=(10, 9))

    for label, values in datasets.items():
        steps = min(display_steps, values["theta"].size)
        time = np.arange(steps) * 0.02

        axes[0].plot(
            time,
            values["theta"][:steps],
            label=label,
        )
        axes[1].plot(
            time,
            values["theta_dot"][:steps],
            label=label,
        )
        axes[2].step(
            time,
            values["torque_applied"][:steps],
            where="post",
            label=label,
        )

    axes[0].set_ylabel("theta [rad]")
    axes[1].set_ylabel("theta_dot [rad/s]")
    axes[2].set_ylabel("torque [Nm]")
    axes[2].set_xlabel("time [s]")

    for axis in axes:
        axis.legend()
        axis.grid()

    figure.tight_layout()
    figure.savefig(
        output_dir / "time_series_comparison.png",
        dpi=150,
    )
    plt.close(figure)


def compare_datasets(
    hold1_dir: Path,
    hold5_dir: Path,
    output_dir: Path,
) -> None:
    """hold1とhold5を比較し、数値と図を保存する。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dirs = {
        "hold1": hold1_dir,
        "hold5": hold5_dir,
    }

    datasets = {}
    statistics = {}

    for label, dataset_dir in dataset_dirs.items():
        rows = load_rows(dataset_dir)
        values = extract_values(rows)

        datasets[label] = values
        statistics[label] = calculate_statistics(values)

    comparison = {
        "hold1": statistics["hold1"],
        "hold5": statistics["hold5"],
        "hold5_to_hold1_theta_dot_range_ratio": (
            statistics["hold5"]["theta_dot_range"]
            / statistics["hold1"]["theta_dot_range"]
        ),
        "hold5_to_hold1_state_range_area_ratio": (
            statistics["hold5"]["state_range_area"]
            / statistics["hold1"]["state_range_area"]
        ),
    }

    summary_path = output_dir / "comparison_summary.json"

    with summary_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            comparison,
            file,
            ensure_ascii=False,
            indent=2,
        )

    save_distribution_comparison(
        datasets=datasets,
        output_dir=output_dir,
    )
    save_state_space_comparison(
        datasets=datasets,
        output_dir=output_dir,
    )
    save_time_series_comparison(
        datasets=datasets,
        output_dir=output_dir,
    )

    print("hold1 and hold5 comparison")
    print()

    for label, summary in statistics.items():
        print(f"{label}:")
        print(f"  transitions: {summary['number_of_transitions']}")
        print(f"  theta range: {summary['theta_range']:.6f}")
        print(
            "  theta_dot range: "
            f"{summary['theta_dot_range']:.6f}"
        )
        print(
            "  torque standard deviation: "
            f"{summary['torque_applied_std']:.6f}"
        )
        print(f"  clipped rate: {summary['clipped_rate']:.6f}")
        print(
            "  state range area: "
            f"{summary['state_range_area']:.6f}"
        )
        print()

    print(
        "hold5 / hold1 theta_dot range ratio: "
        f"{comparison['hold5_to_hold1_theta_dot_range_ratio']:.6f}"
    )
    print(
        "hold5 / hold1 state range area ratio: "
        f"{comparison['hold5_to_hold1_state_range_area_ratio']:.6f}"
    )
    print(f"results saved to: {output_dir}")


def parse_arguments() -> argparse.Namespace:
    """コマンドライン引数を取得する。"""
    parser = argparse.ArgumentParser(
        description="hold1とhold5のデータセットを比較する。"
    )
    parser.add_argument(
        "--hold1",
        type=Path,
        required=True,
        help="hold1データセットのフォルダー",
    )
    parser.add_argument(
        "--hold5",
        type=Path,
        required=True,
        help="hold5データセットのフォルダー",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="比較結果の保存先",
    )
    return parser.parse_args()


def main() -> None:
    """hold1とhold5を比較する。"""
    args = parse_arguments()

    compare_datasets(
        hold1_dir=args.hold1,
        hold5_dir=args.hold5,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()