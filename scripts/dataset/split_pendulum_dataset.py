import argparse
import csv
import json
import random
import subprocess
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


def load_transitions(
    csv_path: Path,
) -> tuple[list[dict[str, str]], list[str]]:
    """状態遷移CSVを読み込む。"""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"transitions.csvが見つかりません: {csv_path}"
        )

    with csv_path.open(
        mode="r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if not rows:
        raise ValueError(
            f"状態遷移データが空です: {csv_path}"
        )

    if fieldnames is None:
        raise ValueError(
            f"CSVの列名を読み込めません: {csv_path}"
        )

    required_columns = {
        "trajectory_id",
        "step",
    }

    missing_columns = required_columns - set(fieldnames)

    if missing_columns:
        raise ValueError(
            "必要な列がありません: "
            + ", ".join(sorted(missing_columns))
        )

    return rows, fieldnames


def group_by_trajectory(
    rows: list[dict[str, str]],
) -> dict[int, list[dict[str, str]]]:
    """状態遷移を軌道IDごとにまとめる。"""
    trajectories: dict[int, list[dict[str, str]]] = {}

    for row in rows:
        trajectory_id = int(row["trajectory_id"])
        trajectories.setdefault(
            trajectory_id,
            [],
        ).append(row)

    for trajectory in trajectories.values():
        trajectory.sort(
            key=lambda row: int(row["step"])
        )

    return trajectories


def split_trajectory_ids(
    trajectory_ids: list[int],
    split_seed: int,
) -> dict[str, list[int]]:
    """軌道IDを学習用・検証用・評価用へ分割する。"""
    if len(trajectory_ids) < 3:
        raise ValueError(
            "軌道単位で3分割するには、最低3軌道必要です。"
        )

    shuffled_ids = trajectory_ids.copy()

    random_generator = random.Random(split_seed)
    random_generator.shuffle(shuffled_ids)

    number_of_trajectories = len(shuffled_ids)

    # 正式データの10軌道では、7・1・2へ分割する
    number_of_train = int(number_of_trajectories * 0.7)
    number_of_validation = max(
        1,
        int(number_of_trajectories * 0.15),
    )

    number_of_test = (
        number_of_trajectories
        - number_of_train
        - number_of_validation
    )

    if number_of_test < 1:
        number_of_test = 1
        number_of_train -= 1

    train_ids = shuffled_ids[:number_of_train]

    validation_start = number_of_train
    validation_end = (
        validation_start + number_of_validation
    )

    validation_ids = shuffled_ids[
        validation_start:validation_end
    ]
    test_ids = shuffled_ids[validation_end:]

    return {
        "train": sorted(train_ids),
        "validation": sorted(validation_ids),
        "test": sorted(test_ids),
    }


def collect_split_rows(
    trajectories: dict[int, list[dict[str, str]]],
    trajectory_ids: list[int],
) -> list[dict[str, str]]:
    """指定された軌道IDのデータを連結する。"""
    split_rows: list[dict[str, str]] = []

    for trajectory_id in trajectory_ids:
        split_rows.extend(
            trajectories[trajectory_id]
        )

    return split_rows


def save_split_csv(
    output_path: Path,
    rows: list[dict[str, str]],
    fieldnames: list[str],
) -> None:
    """分割後の状態遷移をCSVへ保存する。"""
    with output_path.open(
        mode="w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def check_split_independence(
    split_ids: dict[str, list[int]],
) -> None:
    """同じ軌道が複数の集合へ入っていないか確認する。"""
    train_ids = set(split_ids["train"])
    validation_ids = set(split_ids["validation"])
    test_ids = set(split_ids["test"])

    if train_ids & validation_ids:
        raise ValueError(
            "学習用と検証用に同じ軌道が含まれています。"
        )

    if train_ids & test_ids:
        raise ValueError(
            "学習用と評価用に同じ軌道が含まれています。"
        )

    if validation_ids & test_ids:
        raise ValueError(
            "検証用と評価用に同じ軌道が含まれています。"
        )


def split_dataset(
    dataset_dir: Path,
    split_seed: int,
) -> None:
    """データセットを軌道単位で3分割する。"""
    transitions_path = dataset_dir / "transitions.csv"

    rows, fieldnames = load_transitions(
        transitions_path
    )
    trajectories = group_by_trajectory(rows)

    trajectory_ids = sorted(trajectories.keys())

    split_ids = split_trajectory_ids(
        trajectory_ids=trajectory_ids,
        split_seed=split_seed,
    )

    check_split_independence(split_ids)

    split_dir = dataset_dir / "splits"
    split_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_information = {}

    for split_name, selected_ids in split_ids.items():
        split_rows = collect_split_rows(
            trajectories=trajectories,
            trajectory_ids=selected_ids,
        )

        output_path = (
            split_dir
            / f"{split_name}.csv"
        )

        save_split_csv(
            output_path=output_path,
            rows=split_rows,
            fieldnames=fieldnames,
        )

        split_information[split_name] = {
            "trajectory_ids": selected_ids,
            "number_of_trajectories": len(
                selected_ids
            ),
            "number_of_transitions": len(
                split_rows
            ),
            "file": output_path.name,
        }

    all_selected_ids = (
        split_ids["train"]
        + split_ids["validation"]
        + split_ids["test"]
    )

    metadata = {
        "dataset_name": dataset_dir.name,
        "created_at": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),
        "source_file": str(
            transitions_path.resolve()
        ),
        "split_method": "trajectory_level",
        "split_seed": split_seed,
        "total_number_of_trajectories": len(
            trajectories
        ),
        "total_number_of_transitions": len(rows),
        "all_trajectory_ids": trajectory_ids,
        "all_selected_trajectory_ids": sorted(
            all_selected_ids
        ),
        "splits": split_information,
        "git_commit": get_git_commit(),
    }

    metadata_path = (
        split_dir
        / "split_metadata.json"
    )

    with metadata_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Dataset: {dataset_dir.name}")
    print(
        f"Total trajectories: {len(trajectories)}"
    )
    print(f"Total transitions: {len(rows)}")

    for split_name, information in split_information.items():
        print()
        print(f"{split_name}:")
        print(
            "  trajectory IDs: "
            f"{information['trajectory_ids']}"
        )
        print(
            "  trajectories: "
            f"{information['number_of_trajectories']}"
        )
        print(
            "  transitions: "
            f"{information['number_of_transitions']}"
        )

    print()
    print(f"Saved to: {split_dir}")


def parse_arguments() -> argparse.Namespace:
    """コマンドライン引数を取得する。"""
    parser = argparse.ArgumentParser(
        description=(
            "Pendulumデータセットを"
            "軌道単位で学習・検証・評価用へ分割する。"
        )
    )

    parser.add_argument(
        "dataset_dirs",
        nargs="+",
        type=Path,
        help="分割するデータセットフォルダー",
    )

    parser.add_argument(
        "--split-seed",
        type=int,
        default=2026,
        help="軌道IDを分割する乱数seed",
    )

    return parser.parse_args()


def main() -> None:
    """指定されたデータセットを分割する。"""
    args = parse_arguments()

    for dataset_dir in args.dataset_dirs:
        split_dataset(
            dataset_dir=dataset_dir,
            split_seed=args.split_seed,
        )
        print()


if __name__ == "__main__":
    main()