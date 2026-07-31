import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 入力CSVと画像保存先
# ============================================================

# このファイルは、
# リポジトリ/scripts/analysis/plot_pendulum_heatmaps.py
# にあるため、parents[2]がリポジトリのルートになる
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

# 集約済みの実験結果
SUMMARY_RESULTS_PATH = (
    REPOSITORY_ROOT
    / "data"
    / "experiment_results"
    / "pendulum"
    / "summary_results.csv"
)

# ヒートマップの保存先
FIGURE_DIRECTORY = (
    REPOSITORY_ROOT
    / "media"
    / "experiment_figures"
    / "pendulum"
)

# ★★今回描画する予測周期★★
# python -m scripts.analysis.plot_pendulum_heatmaps　実行コマンド

TARGET_PREDICTION_DELTA_T = 0.20

# 実時間制約を表す計算時間
REAL_TIME_LIMIT_MS = 20.0


def load_summary_results() -> list:
    """summary_results.csvを読み込む。"""

    if not SUMMARY_RESULTS_PATH.exists():
        raise FileNotFoundError(
            "summary_results.csvが見つかりません。\n"
            f"確認した場所: {SUMMARY_RESULTS_PATH}"
        )

    results = []

    with SUMMARY_RESULTS_PATH.open(
        mode="r",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            result = {
                "environment": row["environment"],
                "K": int(row["K"]),
                "N": int(row["N"]),
                "control_delta_t": float(
                    row["control_delta_t"]
                ),
                "prediction_delta_t": float(
                    row["prediction_delta_t"]
                ),
                "number_of_trials": int(
                    row["number_of_trials"]
                ),
                "mean_total_cost": float(
                    row["mean_total_cost"]
                ),
                "std_total_cost": float(
                    row["std_total_cost"]
                ),
                "success_rate": float(
                    row["success_rate"]
                ),
                "mean_compute_time_ms": float(
                    row["mean_compute_time_ms"]
                ),
                "std_compute_time_ms": float(
                    row["std_compute_time_ms"]
                ),
                "max_compute_time_ms": float(
                    row["max_compute_time_ms"]
                ),
            }

            results.append(result)

    return results


def filter_target_results(
    results: list,
) -> list:
    """指定した予測周期の結果だけを取り出す。"""

    target_results = [
        result
        for result in results
        if result["environment"] == "pendulum"
        and np.isclose(
            result["prediction_delta_t"],
            TARGET_PREDICTION_DELTA_T,
        )
    ]

    if not target_results:
        raise ValueError(
            "指定した予測周期のデータがありません。\n"
            f"prediction_delta_t="
            f"{TARGET_PREDICTION_DELTA_T}"
        )

    return target_results


def create_result_grid(
    results: list,
    value_name: str,
) -> tuple:
    """Kを縦軸、Nを横軸とする二次元配列を作成する。"""

    # KとNを数値の昇順に並べる
    K_values = sorted(
        {
            result["K"]
            for result in results
        }
    )

    N_values = sorted(
        {
            result["N"]
            for result in results
        }
    )

    # 未計測のセルはNaNとする
    value_grid = np.full(
        shape=(len(K_values), len(N_values)),
        fill_value=np.nan,
        dtype=float,
    )

    # KとNから配列番号を取得するための辞書
    K_to_index = {
        K: index
        for index, K in enumerate(K_values)
    }

    N_to_index = {
        N: index
        for index, N in enumerate(N_values)
    }

    for result in results:
        row_index = K_to_index[result["K"]]
        column_index = N_to_index[result["N"]]

        value_grid[row_index, column_index] = result[
            value_name
        ]

    return K_values, N_values, value_grid


def add_real_time_contour(
    axis,
    time_grid: np.ndarray,
) -> None:
    """20 msの実時間制約を表す白線を描画する。"""

    finite_values = time_grid[
        np.isfinite(time_grid)
    ]

    if finite_values.size == 0:
        return

    minimum_time = np.min(finite_values)
    maximum_time = np.max(finite_values)

    # 20 msがデータ範囲外の場合、等高線は描けない
    if not (
        minimum_time
        <= REAL_TIME_LIMIT_MS
        <= maximum_time
    ):
        print(
            "[INFO] 20 ms contour was not drawn because "
            "all measured times are on the same side "
            "of the 20 ms limit."
        )
        return

    # imshowのセル中心に合わせた格子座標
    x_grid, y_grid = np.meshgrid(
        np.arange(time_grid.shape[1]),
        np.arange(time_grid.shape[0]),
    )

    contour = axis.contour(
        x_grid,
        y_grid,
        time_grid,
        levels=[REAL_TIME_LIMIT_MS],
        colors="white",
        linewidths=2.0,
    )

    axis.clabel(
        contour,
        inline=True,
        fontsize=9,
        fmt={
            REAL_TIME_LIMIT_MS: "20 ms"
        },
    )


def plot_heatmap(
    value_grid: np.ndarray,
    K_values: list,
    N_values: list,
    title: str,
    colorbar_label: str,
    output_path: Path,
    value_format: str,
    color_map: str,
    minimum_value: float | None = None,
    maximum_value: float | None = None,
    time_grid: np.ndarray | None = None,
) -> None:
    """1枚のヒートマップを作成して保存する。"""

    figure, axis = plt.subplots(
        figsize=(7, 7)
    )

    image = axis.imshow(
        value_grid,
        origin="lower",
        aspect="equal",
        cmap=color_map,
        vmin=minimum_value,
        vmax=maximum_value,
    )

    # 横軸：予測ステップ数N
    x_tick_positions = [
        index
        for index, N in enumerate(N_values)
        if N % 10 == 0
    ]

    x_tick_labels = [
        N_values[index]
        for index in x_tick_positions
    ]

    axis.set_xticks(x_tick_positions)
    axis.set_xticklabels(x_tick_labels)
    axis.set_xlabel("Prediction steps (N)")


    # 縦軸：サンプル数K
    y_tick_positions = np.arange(
        1,
        len(K_values),
        2,
    )

    y_tick_labels = [
        K_values[index]
        for index in y_tick_positions
    ]

    axis.set_yticks(y_tick_positions)
    axis.set_yticklabels(y_tick_labels)
    axis.set_ylabel("Number of samples (K)")

    axis.set_title(title)

    colorbar = figure.colorbar(
        image,
        ax=axis,
        shrink = 0.75,
        pad = 0.04,
    )
    colorbar.set_label(colorbar_label)

    # 計算時間の格子が渡された場合は20 ms線を描画する
    if time_grid is not None:
        add_real_time_contour(
            axis=axis,
            time_grid=time_grid,
        )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        f"[INFO] Saved figure: {output_path}"
    )

    plt.close(figure)


def main() -> None:
    """Pendulumの3種類のヒートマップを作成する。"""

    # 画像保存フォルダーを作成する
    FIGURE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    # summary CSVを読み込む
    all_results = load_summary_results()

    # 予測周期20 msのデータだけを取り出す
    results = filter_target_results(
        all_results
    )

    # 総コストの格子を作成する
    K_values, N_values, cost_grid = (
        create_result_grid(
            results=results,
            value_name="mean_total_cost",
        )
    )

    # 成功率の格子を作成する
    _, _, success_grid = create_result_grid(
        results=results,
        value_name="success_rate",
    )

    # 平均計算時間の格子を作成する
    _, _, time_grid = create_result_grid(
        results=results,
        value_name="mean_compute_time_ms",
    )

    dt_ms = int(
        round(
            TARGET_PREDICTION_DELTA_T
            * 1000
        )
    )

    # --------------------------------------------------------
    # 平均総コスト
    # --------------------------------------------------------
    plot_heatmap(
        value_grid=cost_grid,
        K_values=K_values,
        N_values=N_values,
        title=(
            "Pendulum Mean Total Cost\n"
            f"Prediction dt = {dt_ms} ms"
        ),
        colorbar_label=(
            "total cost "
        ),
        output_path=(
            FIGURE_DIRECTORY
            / f"total_cost_dt{dt_ms:03d}.png"
        ),
        value_format=".1f",
        color_map="turbo",
        # 総コスト図だけ800～3000に固定する
        minimum_value=800.0,
        maximum_value=3000.0,
        time_grid=time_grid,
    )

    # --------------------------------------------------------
    # タスク成功率
    # --------------------------------------------------------
    plot_heatmap(
        value_grid=success_grid,
        K_values=K_values,
        N_values=N_values,
        title=(
            "Pendulum Task Success Rate\n"
            f"Prediction dt = {dt_ms} ms"
        ),
        colorbar_label=(
            "Success rate "
        ),
        output_path=(
            FIGURE_DIRECTORY
            / f"success_rate_dt{dt_ms:03d}.png"
        ),
        value_format=".2f",
        color_map="turbo_r",
        minimum_value=0.0,
        maximum_value=1.0,
        time_grid=time_grid,
    )

    # --------------------------------------------------------
    # 平均計算時間
    # --------------------------------------------------------
    # 計算時間ヒートマップは予測周期ごとには生成しない
    # 計算量は主にKとNで決まり、卒論でも共通の図として扱われている
    # plot_heatmap(
    #     value_grid=time_grid,
    #     K_values=K_values,
    #     N_values=N_values,
    #     title=(
    #         "Pendulum Mean Computation Time\n"
    #         f"Prediction dt = {dt_ms} ms"
    #     ),
    #     colorbar_label=(
    #         "Mean computation time [ms]"
    #     ),
    #     output_path=(
    #         FIGURE_DIRECTORY
    #         / f"compute_time_dt{dt_ms:03d}.png"
    #     ),
    #     value_format=".3f",
    #     color_map="turbo",
    #     time_grid=time_grid,
    # )

    print()
    print(
        "[INFO] All Pendulum heatmaps "
        "were generated."
    )


if __name__ == "__main__":
    main()