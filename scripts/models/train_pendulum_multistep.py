import argparse
import csv
import json
import random
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.models.pendulum_multistep_loss import PendulumMultistepLoss
from scripts.models.pendulum_multistep_predictor import (
    PendulumMultistepPredictor,
)
from scripts.models.pendulum_neural_dynamics import PendulumNeuralDynamics
from scripts.models.pendulum_sequence_dataset import (
    PendulumSequenceDataset,
)


def set_random_seed(seed: int) -> None:
    """Python・NumPy・PyTorchの乱数seedを固定する。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_git_commit() -> str:
    """現在のGitコミットIDを取得する。"""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def create_data_loader(
    csv_path: Path,
    horizon: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """複数ステップ予測用DataLoaderを作成する。"""

    dataset = PendulumSequenceDataset(
        csv_path=csv_path,
        horizon=horizon,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator if shuffle else None,
    )


def run_epoch(
    predictor: PendulumMultistepPredictor,
    data_loader: DataLoader,
    criterion: PendulumMultistepLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip_norm: float = 1.0,
) -> dict[str, float]:
    """学習または検証を1エポック実行する。"""

    is_training = optimizer is not None

    if is_training:
        predictor.train()
    else:
        predictor.eval()

    metric_sums = {
        "loss": 0.0,
        "sin_rmse": 0.0,
        "cos_rmse": 0.0,
        "angular_velocity_rmse": 0.0,
    }
    number_of_samples = 0

    for batch in data_loader:
        initial_state = batch["initial_state"].to(device)
        torque_sequence = batch["torque_sequence"].to(device)
        target_state_sequence = batch[
            "target_state_sequence"
        ].to(device)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            prediction = predictor(
                initial_state=initial_state,
                torque_sequence=torque_sequence,
            )

            loss_result = criterion(
                predicted_encoded_states=prediction[
                    "encoded_states"
                ],
                target_physical_states=target_state_sequence,
            )

            loss = loss_result["loss"]

            if is_training:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    predictor.parameters(),
                    max_norm=gradient_clip_norm,
                )

                optimizer.step()

        batch_size = initial_state.shape[0]
        number_of_samples += batch_size

        for metric_name in metric_sums:
            metric_sums[metric_name] += (
                loss_result[metric_name].detach().item()
                * batch_size
            )

    return {
        metric_name: metric_sum / number_of_samples
        for metric_name, metric_sum in metric_sums.items()
    }


def save_history(
    history: list[dict[str, float | int]],
    output_path: Path,
) -> None:
    """エポックごとの学習履歴をCSVへ保存する。"""

    if not history:
        return

    with output_path.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(history[0].keys()),
        )
        writer.writeheader()
        writer.writerows(history)


def save_checkpoint(
    output_path: Path,
    predictor: PendulumMultistepPredictor,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_loss: float,
    config: dict,
) -> None:
    """学習済みモデルと学習条件を保存する。"""

    torch.save(
        {
            "epoch": epoch,
            "validation_loss": validation_loss,
            "model_state_dict": predictor.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        output_path,
    )


def train(
    dataset_directory: Path,
    output_directory: Path,
    horizon: int,
    hidden_size: int,
    batch_size: int,
    number_of_epochs: int,
    learning_rate: float,
    seed: int,
    patience: int,
    gradient_clip_norm: float,
) -> None:
    """Pendulum NNを複数ステップ誤差で学習する。"""

    split_directory = dataset_directory / "splits"
    train_csv_path = split_directory / "train.csv"
    validation_csv_path = split_directory / "validation.csv"
    normalization_path = split_directory / "normalization.npz"

    required_paths = (
        train_csv_path,
        validation_csv_path,
        normalization_path,
    )

    for required_path in required_paths:
        if not required_path.exists():
            raise FileNotFoundError(
                "必要なファイルが見つかりません。\n"
                f"確認した場所: {required_path}"
            )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    set_random_seed(seed)

    device = torch.device("cpu")

    train_loader = create_data_loader(
        csv_path=train_csv_path,
        horizon=horizon,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )

    validation_loader = create_data_loader(
        csv_path=validation_csv_path,
        horizon=horizon,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )

    dynamics_model = PendulumNeuralDynamics(
        normalization_path=normalization_path,
        hidden_size=hidden_size,
    ).to(device)

    predictor = PendulumMultistepPredictor(
        dynamics_model=dynamics_model,
    ).to(device)

    criterion = PendulumMultistepLoss()

    optimizer = torch.optim.Adam(
        predictor.parameters(),
        lr=learning_rate,
    )

    config = {
        "dataset_directory": str(dataset_directory),
        "normalization_path": str(normalization_path),
        "output_directory": str(output_directory),
        "horizon": horizon,
        "hidden_size": hidden_size,
        "batch_size": batch_size,
        "number_of_epochs": number_of_epochs,
        "learning_rate": learning_rate,
        "seed": seed,
        "patience": patience,
        "gradient_clip_norm": gradient_clip_norm,
        "device": str(device),
        "git_commit": get_git_commit(),
        "created_at": datetime.now().astimezone().isoformat(),
        "loss": (
            "sin_mse + cos_mse + angular_velocity_mse"
        ),
        "teacher_forcing": False,
    }

    with (
        output_directory / "training_config.json"
    ).open(
        mode="w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            config,
            json_file,
            ensure_ascii=False,
            indent=2,
        )

    history = []
    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, number_of_epochs + 1):
        train_metrics = run_epoch(
            predictor=predictor,
            data_loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            gradient_clip_norm=gradient_clip_norm,
        )

        with torch.no_grad():
            validation_metrics = run_epoch(
                predictor=predictor,
                data_loader=validation_loader,
                criterion=criterion,
                device=device,
            )

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "validation_loss": validation_metrics["loss"],
            "train_sin_rmse": train_metrics["sin_rmse"],
            "validation_sin_rmse": (
                validation_metrics["sin_rmse"]
            ),
            "train_cos_rmse": train_metrics["cos_rmse"],
            "validation_cos_rmse": (
                validation_metrics["cos_rmse"]
            ),
            "train_angular_velocity_rmse": (
                train_metrics["angular_velocity_rmse"]
            ),
            "validation_angular_velocity_rmse": (
                validation_metrics["angular_velocity_rmse"]
            ),
        }

        history.append(epoch_result)

        save_history(
            history=history,
            output_path=output_directory / "history.csv",
        )

        print(
            f"Epoch {epoch:04d} | "
            f"train loss={train_metrics['loss']:.6f} | "
            f"validation loss="
            f"{validation_metrics['loss']:.6f} | "
            f"validation theta_dot RMSE="
            f"{validation_metrics['angular_velocity_rmse']:.6f}"
        )

        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            epochs_without_improvement = 0

            save_checkpoint(
                output_path=output_directory / "best_model.pt",
                predictor=predictor,
                optimizer=optimizer,
                epoch=epoch,
                validation_loss=best_validation_loss,
                config=config,
            )

            print(
                "[INFO] Best model updated: "
                f"validation loss={best_validation_loss:.6f}"
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(
                "[INFO] Early stopping: "
                f"no improvement for {patience} epochs."
            )
            break

    final_validation_loss = history[-1][
        "validation_loss"
    ]

    save_checkpoint(
        output_path=output_directory / "final_model.pt",
        predictor=predictor,
        optimizer=optimizer,
        epoch=int(history[-1]["epoch"]),
        validation_loss=float(final_validation_loss),
        config=config,
    )

    print()
    print("[INFO] Training completed.")
    print(f"[INFO] Best validation loss: {best_validation_loss:.6f}")
    print(f"[INFO] Output directory: {output_directory}")


def parse_arguments() -> argparse.Namespace:
    """コマンドライン引数を読み込む。"""

    parser = argparse.ArgumentParser(
        description=(
            "Train Pendulum neural dynamics "
            "with multistep prediction loss."
        )
    )

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Pendulumデータセットフォルダー",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="学習結果の保存先",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=50,
        help="複数ステップ予測長。既定値: 50",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=32,
        help="各中間層のユニット数。既定値: 32",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="バッチサイズ。既定値: 64",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="学習エポック数。既定値: 3",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1.0e-3,
        help="学習率。既定値: 0.001",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="乱数seed。既定値: 0",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stoppingの待機エポック数",
    )
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=1.0,
        help="勾配ノルムの上限。既定値: 1.0",
    )

    return parser.parse_args()


def main() -> None:
    """コマンドライン引数に基づいてNNを学習する。"""

    arguments = parse_arguments()

    train(
        dataset_directory=arguments.dataset_dir,
        output_directory=arguments.output_dir,
        horizon=arguments.horizon,
        hidden_size=arguments.hidden_size,
        batch_size=arguments.batch_size,
        number_of_epochs=arguments.epochs,
        learning_rate=arguments.learning_rate,
        seed=arguments.seed,
        patience=arguments.patience,
        gradient_clip_norm=arguments.gradient_clip_norm,
    )


if __name__ == "__main__":
    main()