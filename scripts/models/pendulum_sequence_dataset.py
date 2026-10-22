import csv
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset


class PendulumSequenceDataset(Dataset):
    """連続軌道から複数ステップ予測区間を作るデータセット。"""

    def __init__(
        self,
        csv_path: Path | str,
        horizon: int,
        stride: int = 1,
    ) -> None:
        if horizon <= 0:
            raise ValueError(
                "horizonには正の整数が必要です。"
            )

        if stride <= 0:
            raise ValueError(
                "strideには正の整数が必要です。"
            )

        self.csv_path = Path(csv_path)
        self.horizon = int(horizon)
        self.stride = int(stride)

        if not self.csv_path.exists():
            raise FileNotFoundError(
                "状態遷移CSVが見つかりません。\n"
                f"確認した場所: {self.csv_path}"
            )

        self.trajectories = self._load_trajectories()
        self.sequence_indices = self._create_sequence_indices()

        if not self.sequence_indices:
            raise ValueError(
                "指定したhorizonで作成できる"
                "複数ステップ区間がありません。"
            )

    def _load_trajectories(
        self,
    ) -> dict[int, list[dict[str, float | int]]]:
        """CSVを軌道IDごとの連続状態遷移へ分ける。"""

        trajectories = defaultdict(list)

        with self.csv_path.open(
            mode="r",
            encoding="utf-8-sig",
            newline="",
        ) as csv_file:
            reader = csv.DictReader(csv_file)

            required_columns = {
                "trajectory_id",
                "step",
                "theta",
                "theta_dot",
                "torque_applied",
                "next_theta",
                "next_theta_dot",
            }

            if reader.fieldnames is None:
                raise ValueError(
                    "CSVにヘッダーがありません。"
                )

            missing_columns = (
                required_columns - set(reader.fieldnames)
            )

            if missing_columns:
                raise ValueError(
                    "CSVに必要な列がありません: "
                    + ", ".join(sorted(missing_columns))
                )

            for row in reader:
                trajectory_id = int(
                    row["trajectory_id"]
                )

                trajectories[trajectory_id].append(
                    {
                        "trajectory_id": trajectory_id,
                        "step": int(row["step"]),
                        "theta": float(row["theta"]),
                        "theta_dot": float(row["theta_dot"]),
                        "torque_applied": float(
                            row["torque_applied"]
                        ),
                        "next_theta": float(
                            row["next_theta"]
                        ),
                        "next_theta_dot": float(
                            row["next_theta_dot"]
                        ),
                    }
                )

        sorted_trajectories = {}

        for trajectory_id, rows in trajectories.items():
            sorted_rows = sorted(
                rows,
                key=lambda row: row["step"],
            )

            self._validate_trajectory(
                trajectory_id=trajectory_id,
                rows=sorted_rows,
            )

            sorted_trajectories[
                trajectory_id
            ] = sorted_rows

        return sorted_trajectories

    @staticmethod
    def _validate_trajectory(
        trajectory_id: int,
        rows: list[dict[str, float | int]],
    ) -> None:
        """軌道内のステップ番号と状態遷移の連続性を確認する。"""

        for index, row in enumerate(rows):
            if row["step"] != index:
                raise ValueError(
                    "ステップ番号が連続していません。"
                    f" trajectory_id={trajectory_id},"
                    f" expected={index},"
                    f" actual={row['step']}"
                )

        for index in range(len(rows) - 1):
            current_row = rows[index]
            next_row = rows[index + 1]

            theta_is_continuous = (
                abs(
                    current_row["next_theta"]
                    - next_row["theta"]
                )
                <= 1.0e-9
            )

            theta_dot_is_continuous = (
                abs(
                    current_row["next_theta_dot"]
                    - next_row["theta_dot"]
                )
                <= 1.0e-9
            )

            if not (
                theta_is_continuous
                and theta_dot_is_continuous
            ):
                raise ValueError(
                    "状態遷移が連続していません。"
                    f" trajectory_id={trajectory_id},"
                    f" step={current_row['step']}"
                )

    def _create_sequence_indices(
        self,
    ) -> list[tuple[int, int]]:
        """各複数ステップ区間の軌道IDと開始位置を作る。"""

        sequence_indices = []

        for trajectory_id, rows in self.trajectories.items():
            maximum_start = (
                len(rows) - self.horizon
            )

            for start_index in range(
                0,
                maximum_start + 1,
                self.stride,
            ):
                sequence_indices.append(
                    (
                        trajectory_id,
                        start_index,
                    )
                )

        return sequence_indices

    def __len__(self) -> int:
        return len(self.sequence_indices)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor]:
        trajectory_id, start_index = (
            self.sequence_indices[index]
        )

        rows = self.trajectories[trajectory_id]

        sequence_rows = rows[
            start_index:
            start_index + self.horizon
        ]

        initial_state = torch.tensor(
            [
                sequence_rows[0]["theta"],
                sequence_rows[0]["theta_dot"],
            ],
            dtype=torch.float32,
        )

        torque_sequence = torch.tensor(
            [
                row["torque_applied"]
                for row in sequence_rows
            ],
            dtype=torch.float32,
        )

        target_state_sequence = torch.tensor(
            [
                [
                    row["next_theta"],
                    row["next_theta_dot"],
                ]
                for row in sequence_rows
            ],
            dtype=torch.float32,
        )

        return {
            "initial_state": initial_state,
            "torque_sequence": torque_sequence,
            "target_state_sequence": target_state_sequence,
            "trajectory_id": torch.tensor(
                trajectory_id,
                dtype=torch.long,
            ),
            "start_step": torch.tensor(
                sequence_rows[0]["step"],
                dtype=torch.long,
            ),
        }