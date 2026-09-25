import csv
import json
import subprocess
import sys
import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


# リポジトリのルートディレクトリ
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# scriptsパッケージを読み込めるようにする
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pendulum import Pendulum


# 生成したデータセットの保存先
DATASET_ROOT = PROJECT_ROOT / "data" / "datasets" / "pendulum"

# 状態遷移CSVの列順
TRANSITION_FIELDNAMES = (
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


@dataclass(frozen=True)
class DatasetConfig:
    """Pendulumのランダム軌道収集条件を保持する。"""

    dataset_name: str
    delta_t: float
    initial_theta: float
    initial_theta_dot: float
    number_of_trajectories: int
    trajectory_length: int
    noise_mean: float
    noise_std: float
    torque_hold_steps: int
    max_torque_abs: float
    max_speed_abs: float
    base_seed: int

    @property
    def total_transitions(self) -> int:
        """収集予定の状態遷移数を返す。"""
        return self.number_of_trajectories * self.trajectory_length

    @property
    def output_dir(self) -> Path:
        """データセットの保存先を返す。"""
        return DATASET_ROOT / self.dataset_name


def create_test_config(
    torque_hold_steps: int = 5,
) -> DatasetConfig:
    """小規模な動作確認用の設定を返す。"""
    return DatasetConfig(
        dataset_name=f"random_test_hold{torque_hold_steps}",
        delta_t=0.02,
        initial_theta=float(np.pi),
        initial_theta_dot=0.0,
        number_of_trajectories=2,
        trajectory_length=100,
        noise_mean=0.0,
        noise_std=1.0,
        torque_hold_steps=torque_hold_steps,
        max_torque_abs=2.0,
        max_speed_abs=8.0,
        base_seed=0,
    )


def create_full_config(
    torque_hold_steps: int,
) -> DatasetConfig:
    """正式な初期データ収集用の設定を返す。"""
    task_steps = 375

    return DatasetConfig(
        dataset_name=f"random_hold{torque_hold_steps}_v001",
        delta_t=0.02,
        initial_theta=float(np.pi),
        initial_theta_dot=0.0,
        number_of_trajectories=10,
        trajectory_length=task_steps * 2,
        noise_mean=0.0,
        noise_std=1.0,
        torque_hold_steps=torque_hold_steps,
        max_torque_abs=2.0,
        max_speed_abs=8.0,
        base_seed=0,
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


def save_config(config: DatasetConfig) -> Path:
    """データ収集条件をJSON形式で保存する。"""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = config.output_dir / "config.json"

    config_data = asdict(config)
    config_data["total_transitions"] = config.total_transitions
    config_data["trajectory_seeds"] = [
        config.base_seed + trajectory_id
        for trajectory_id in range(config.number_of_trajectories)
    ]
    config_data["transition_columns"] = list(TRANSITION_FIELDNAMES)
    config_data["git_commit"] = get_git_commit()

    with config_path.open(mode="w", encoding="utf-8") as file:
        json.dump(
            config_data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return config_path


def generate_trajectory(
    config: DatasetConfig,
    trajectory_id: int,
) -> list[dict[str, int | float]]:
    """ランダムトルクを加えた連続軌道を1本生成する。"""
    trajectory_seed = config.base_seed + trajectory_id
    rng = np.random.default_rng(trajectory_seed)

    pendulum = Pendulum(
        max_torque_abs=config.max_torque_abs,
        max_speed_abs=config.max_speed_abs,
        delta_t=config.delta_t,
        visualize=False,
    )

    pendulum.reset(
        init_state=np.array(
            [
                config.initial_theta,
                config.initial_theta_dot,
            ],
            dtype=float,
        )
    )

    transitions: list[dict[str, int | float]] = []
    torque_command = 0.0

    for step in range(config.trajectory_length):
        # 保持区間の先頭で新しいランダムトルクを生成する
        if step % config.torque_hold_steps == 0:
            torque_command = float(
                rng.normal(
                    loc=config.noise_mean,
                    scale=config.noise_std,
                )
            )

        # Pendulumへ実際に適用するトルク
        torque_applied = float(
            np.clip(
                torque_command,
                -config.max_torque_abs,
                config.max_torque_abs,
            )
        )

        # 入力適用前の状態を取得する
        theta, theta_dot = pendulum.get_state()

        # 物理式Pendulumを20 ms進める
        pendulum.update(
            u=np.array([torque_applied], dtype=float),
            delta_t=config.delta_t,
            append_frame=False,
        )

        # 入力適用後の状態を取得する
        next_theta, next_theta_dot = pendulum.get_state()

        transitions.append(
            {
                "trajectory_id": trajectory_id,
                "step": step,
                "theta": float(theta),
                "theta_dot": float(theta_dot),
                "torque_command": torque_command,
                "torque_applied": torque_applied,
                "next_theta": float(next_theta),
                "next_theta_dot": float(next_theta_dot),
                "delta_t": config.delta_t,
                "seed": trajectory_seed,
                "noise_std": config.noise_std,
                "torque_hold_steps": config.torque_hold_steps,
                "initial_theta": config.initial_theta,
                "initial_theta_dot": config.initial_theta_dot,
            }
        )

    return transitions


def generate_dataset(
    config: DatasetConfig,
) -> list[dict[str, int | float]]:
    """設定された本数の軌道を生成する。"""
    all_transitions: list[dict[str, int | float]] = []

    for trajectory_id in range(config.number_of_trajectories):
        trajectory = generate_trajectory(
            config=config,
            trajectory_id=trajectory_id,
        )
        all_transitions.extend(trajectory)

        print(
            f"trajectory {trajectory_id} completed: "
            f"{len(trajectory)} transitions"
        )

    return all_transitions


def save_transitions(
    config: DatasetConfig,
    transitions: list[dict[str, int | float]],
) -> Path:
    """状態遷移をCSVへ保存する。"""
    csv_path = config.output_dir / "transitions.csv"

    with csv_path.open(
        mode="w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=TRANSITION_FIELDNAMES,
        )
        writer.writeheader()
        writer.writerows(transitions)

    return csv_path


def save_metadata(
    config: DatasetConfig,
    actual_transitions: int,
) -> Path:
    """データセットの収集結果をJSON形式で保存する。"""
    metadata_path = config.output_dir / "metadata.json"

    metadata = {
        "dataset_name": config.dataset_name,
        "created_at": datetime.now().astimezone().isoformat(),
        "number_of_trajectories": config.number_of_trajectories,
        "trajectory_length": config.trajectory_length,
        "planned_transitions": config.total_transitions,
        "actual_transitions": actual_transitions,
        "trajectory_seeds": [
            config.base_seed + trajectory_id
            for trajectory_id in range(config.number_of_trajectories)
        ],
        "transition_columns": list(TRANSITION_FIELDNAMES),
        "git_commit": get_git_commit(),
        "status": (
            "completed"
            if actual_transitions == config.total_transitions
            else "incomplete"
        ),
    }

    with metadata_path.open(mode="w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return metadata_path


def print_config(config: DatasetConfig) -> None:
    """設定内容を表示する。"""
    print("Pendulum random dataset configuration")

    for key, value in asdict(config).items():
        print(f"{key}: {value}")

    print(f"total_transitions: {config.total_transitions}")
    print(f"output_dir: {config.output_dir}")

def create_dataset(config: DatasetConfig) -> None:
    """指定された条件でデータセットを生成して保存する。"""
    config.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print_config(config)

    config_path = save_config(config)
    transitions = generate_dataset(config)

    csv_path = save_transitions(
        config=config,
        transitions=transitions,
    )

    metadata_path = save_metadata(
        config=config,
        actual_transitions=len(transitions),
    )

    print(f"config saved: {config_path}")
    print(f"transitions saved: {csv_path}")
    print(f"metadata saved: {metadata_path}")
    print(f"actual transitions: {len(transitions)}")
    print()

def parse_arguments() -> argparse.Namespace:
    """データ収集条件をコマンドラインから取得する。"""
    parser = argparse.ArgumentParser(
        description="Pendulumのランダム軌道データを生成する。"
    )
    parser.add_argument(
        "--mode",
        choices=("test", "full"),
        default="test",
        help="testは小規模確認、fullは正式収集",
    )
    parser.add_argument(
        "--hold-steps",
        type=int,
        choices=(1, 5),
        default=5,
        help="同じランダムトルクを保持するステップ数",
    )

    return parser.parse_args()    

def main() -> None:
    """指定された条件でランダム軌道データセットを生成する。"""
    args = parse_arguments()

    if args.mode == "test":
        config = create_test_config(
            torque_hold_steps=args.hold_steps,
        )
    else:
        config = create_full_config(
            torque_hold_steps=args.hold_steps,
        )

    config.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print_config(config)

    config_path = save_config(config)
    transitions = generate_dataset(config)

    csv_path = save_transitions(
        config=config,
        transitions=transitions,
    )
    metadata_path = save_metadata(
        config=config,
        actual_transitions=len(transitions),
    )

    print(f"config saved: {config_path}")
    print(f"transitions saved: {csv_path}")
    print(f"metadata saved: {metadata_path}")
    print(f"actual transitions: {len(transitions)}")

if __name__ == "__main__":
    main()