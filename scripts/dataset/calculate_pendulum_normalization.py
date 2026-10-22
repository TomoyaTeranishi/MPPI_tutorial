#標準化情報作成

import argparse
import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

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


def get_git_commit() -> str:
    """現在のGitコミットIDを取得する。"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def load_training_data(
    train_csv_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """学習用CSVからNNの入力配列と出力配列を作成する。"""
    if not train_csv_path.exists():
        raise FileNotFoundError(
            f"train.csvが見つかりません: {train_csv_path}"
        )

    with train_csv_path.open(
        mode="r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(
            f"学習用データが空です: {train_csv_path}"
        )

    required_columns = {
        "theta",
        "theta_dot",
        "torque_applied",
        "next_theta",
        "next_theta_dot",
    }

    missing_columns = required_columns - set(rows[0].keys())

    if missing_columns:
        raise ValueError(
            "必要な列がありません: "
            + ", ".join(sorted(missing_columns))
        )

    theta = np.array(
        [float(row["theta"]) for row in rows],
        dtype=np.float64,
    )
    theta_dot = np.array(
        [float(row["theta_dot"]) for row in rows],
        dtype=np.float64,
    )
    torque_applied = np.array(
        [float(row["torque_applied"]) for row in rows],
        dtype=np.float64,
    )
    next_theta = np.array(
        [float(row["next_theta"]) for row in rows],
        dtype=np.float64,
    )
    next_theta_dot = np.array(
        [float(row["next_theta_dot"]) for row in rows],
        dtype=np.float64,
    )

    inputs = np.column_stack(
        (
            np.sin(theta),
            np.cos(theta),
            theta_dot,
            torque_applied,
        )
    )

    outputs = np.column_stack(
        (
            np.sin(next_theta),
            np.cos(next_theta),
            next_theta_dot,
        )
    )

    if not np.all(np.isfinite(inputs)):
        raise ValueError("NN入力にNaNまたは無限大があります。")

    if not np.all(np.isfinite(outputs)):
        raise ValueError("NN出力にNaNまたは無限大があります。")

    return inputs, outputs


def calculate_mean_and_std(
    values: np.ndarray,
    variable_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """各変数の平均値と標準偏差を計算する。"""
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0, ddof=0)

    zero_std_names = [
        variable_names[index]
        for index, std_value in enumerate(std)
        if std_value < 1.0e-12
    ]

    safe_std = std.copy()
    safe_std[safe_std < 1.0e-12] = 1.0

    return mean, safe_std, zero_std_names


def check_standardization(
    values: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> dict[str, list[float]]:
    """標準化後の平均値と標準偏差を確認する。"""
    standardized = (values - mean) / std

    return {
        "standardized_mean": np.mean(
            standardized,
            axis=0,
        ).tolist(),
        "standardized_std": np.std(
            standardized,
            axis=0,
            ddof=0,
        ).tolist(),
    }


def save_normalization(
    dataset_dir: Path,
    inputs: np.ndarray,
    outputs: np.ndarray,
    input_mean: np.ndarray,
    input_std: np.ndarray,
    output_mean: np.ndarray,
    output_std: np.ndarray,
    input_zero_std_names: list[str],
    output_zero_std_names: list[str],
) -> None:
    """標準化情報をJSONとNPZへ保存する。"""
    split_dir = dataset_dir / "splits"
    json_path = split_dir / "normalization.json"
    npz_path = split_dir / "normalization.npz"

    input_check = check_standardization(
        values=inputs,
        mean=input_mean,
        std=input_std,
    )
    output_check = check_standardization(
        values=outputs,
        mean=output_mean,
        std=output_std,
    )

    normalization_data = {
        "dataset_name": dataset_dir.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "source_file": str(
            (split_dir / "train.csv").resolve()
        ),
        "number_of_training_samples": int(inputs.shape[0]),
        "calculation_scope": "training_data_only",
        "standardization_formula": "(value - mean) / std",
        "inverse_formula": "standardized_value * std + mean",
        "input_names": list(INPUT_NAMES),
        "input_mean": input_mean.tolist(),
        "input_std": input_std.tolist(),
        "output_names": list(OUTPUT_NAMES),
        "output_mean": output_mean.tolist(),
        "output_std": output_std.tolist(),
        "input_zero_std_names": input_zero_std_names,
        "output_zero_std_names": output_zero_std_names,
        "input_standardization_check": input_check,
        "output_standardization_check": output_check,
        "git_commit": get_git_commit(),
    }

    with json_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            normalization_data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    np.savez(
        npz_path,
        input_names=np.array(INPUT_NAMES),
        input_mean=input_mean,
        input_std=input_std,
        output_names=np.array(OUTPUT_NAMES),
        output_mean=output_mean,
        output_std=output_std,
    )

    print(f"Dataset: {dataset_dir.name}")
    print(f"Training samples: {inputs.shape[0]}")
    print(f"Input mean: {input_mean}")
    print(f"Input std: {input_std}")
    print(f"Output mean: {output_mean}")
    print(f"Output std: {output_std}")
    print(f"JSON: {json_path}")
    print(f"NPZ: {npz_path}")
    print()


def process_dataset(dataset_dir: Path) -> None:
    """1つのデータセットについて標準化情報を計算する。"""
    train_csv_path = dataset_dir / "splits" / "train.csv"

    inputs, outputs = load_training_data(
        train_csv_path=train_csv_path,
    )

    input_mean, input_std, input_zero_std_names = (
        calculate_mean_and_std(
            values=inputs,
            variable_names=INPUT_NAMES,
        )
    )

    output_mean, output_std, output_zero_std_names = (
        calculate_mean_and_std(
            values=outputs,
            variable_names=OUTPUT_NAMES,
        )
    )

    save_normalization(
        dataset_dir=dataset_dir,
        inputs=inputs,
        outputs=outputs,
        input_mean=input_mean,
        input_std=input_std,
        output_mean=output_mean,
        output_std=output_std,
        input_zero_std_names=input_zero_std_names,
        output_zero_std_names=output_zero_std_names,
    )


def parse_arguments() -> argparse.Namespace:
    """コマンドライン引数を取得する。"""
    parser = argparse.ArgumentParser(
        description=(
            "Pendulumの学習用データから"
            "NN入出力の標準化情報を計算する。"
        )
    )
    parser.add_argument(
        "dataset_dirs",
        nargs="+",
        type=Path,
        help="標準化情報を計算するデータセットフォルダー",
    )
    return parser.parse_args()


def main() -> None:
    """指定された各データセットの標準化情報を計算する。"""
    args = parse_arguments()

    for dataset_dir in args.dataset_dirs:
        process_dataset(dataset_dir)


if __name__ == "__main__":
    main()