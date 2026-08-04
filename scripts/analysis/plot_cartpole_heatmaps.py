import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 入力CSVと画像保存先
# ============================================================

# このファイルは、
# リポジトリ/scripts/analysis/plot_cartpole_heatmaps.py
# にあるため、parents[2]がリポジトリのルートになる
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


# CartPoleの集約済み実験結果
SUMMARY_RESULTS_PATH = (
    REPOSITORY_ROOT
    / "data"
    / "experiment_results"
    / "cartpole"
    / "summary_results.csv"
)


# CartPoleヒートマップの保存先
FIGURE_DIRECTORY = (
    REPOSITORY_ROOT
    / "media"
    / "experiment_figures"
    / "cartpole"
)


# ============================================================
# 今回描画する予測周期
# ============================================================

# 実行コマンド：
# python -m scripts.analysis.plot_cartpole_heatmaps

# 20 msの場合は0.02
# 40 msの場合は0.04
# 200 msの場合は0.20
TARGET_PREDICTION_DELTA_T = 0.12


# 実時間制約を表す平均計算時間
REAL_TIME_LIMIT_MS = 20.0


# 卒論条件では同一条件を5seedで実行する
EXPECTED_NUMBER_OF_TRIALS = 5


# CartPole総コスト図の表示範囲
#
# この範囲を超える高コスト条件は、
# カラーバー上限の色で表示する
COST_MINIMUM_VALUE = 7000.0
COST_MAXIMUM_VALUE = 18000.0


def load_summary_results() -> list:
    """CartPoleのsummary_results.csvを読み込む。"""

    if not SUMMARY_RESULTS_PATH.exists():
        raise FileNotFoundError(
            "CartPoleのsummary_results.csvが見つかりません。\n"
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
    """指定した予測周期のCartPole結果だけを取り出す。"""

    target_results = [
        result
        for result in results
        if result["environment"] == "cartpole"
        and np.isclose(
            result["prediction_delta_t"],
            TARGET_PREDICTION_DELTA_T,
        )
    ]

    if not target_results:
        raise ValueError(
            "指定した予測周期のCartPoleデータがありません。\n"
            "prediction_delta_t="
            f"{TARGET_PREDICTION_DELTA_T}"
        )

    # 5試行が揃っていない条件を確認する
    incomplete_results = [
        result
        for result in target_results
        if result["number_of_trials"]
        != EXPECTED_NUMBER_OF_TRIALS
    ]

    if incomplete_results:
        print(
            "[WARNING] 5 trials are not complete "
            "for the following conditions:"
        )

        for result in incomplete_results:
            print(
                "          "
                f"K={result['K']}, "
                f"N={result['N']}, "
                f"trials={result['number_of_trials']}"
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

    # 未計測セルはNaNとする
    value_grid = np.full(
        shape=(
            len(K_values),
            len(N_values),
        ),
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
        row_index = K_to_index[
            result["K"]
        ]

        column_index = N_to_index[
            result["N"]
        ]

        value = result[value_name]

        # NaNや無限大は未計測セルとして残す
        if np.isfinite(value):
            value_grid[
                row_index,
                column_index,
            ] = value

    return (
        K_values,
        N_values,
        value_grid,
    )


def add_real_time_contour(
    axis,
    time_grid: np.ndarray,
) -> None:
    """20 ms以下と20 ms超のセル境界をマス目に沿って描画する。"""

    time_grid = np.asarray(
        time_grid,
        dtype=float,
    )

    finite_mask = np.isfinite(
        time_grid
    )

    finite_values = time_grid[
        finite_mask
    ]

    if finite_values.size == 0:
        print(
            "[INFO] Computation-time grid "
            "has no finite values."
        )
        return

    minimum_time = float(
        np.min(finite_values)
    )

    maximum_time = float(
        np.max(finite_values)
    )

    # 全条件が20 ms以下、または全条件が20 ms超の場合は、
    # ヒートマップ内部に境界線が存在しない
    if not (
        minimum_time
        <= REAL_TIME_LIMIT_MS
        < maximum_time
    ):
        print(
            "[INFO] 20 ms boundary was not drawn. "
            "Measured time range: "
            f"{minimum_time:.3f} to "
            f"{maximum_time:.3f} ms"
        )
        return

    # True：20 ms以内の実時間実行可能領域
    # False：20 msを超える領域
    feasible_grid = (
        time_grid
        <= REAL_TIME_LIMIT_MS
    )

    number_of_rows = (
        time_grid.shape[0]
    )

    number_of_columns = (
        time_grid.shape[1]
    )

    boundary_was_drawn = False

    # --------------------------------------------------------
    # 左右に隣接するセルの境界
    # マス目に沿った縦線を描画する
    # --------------------------------------------------------

    for row_index in range(
        number_of_rows
    ):
        for column_index in range(
            1,
            number_of_columns,
        ):
            left_is_finite = finite_mask[
                row_index,
                column_index - 1,
            ]

            right_is_finite = finite_mask[
                row_index,
                column_index,
            ]

            # NaNを含むセル間には線を描かない
            if not (
                left_is_finite
                and right_is_finite
            ):
                continue

            left_is_feasible = feasible_grid[
                row_index,
                column_index - 1,
            ]

            right_is_feasible = feasible_grid[
                row_index,
                column_index,
            ]

            # 20 ms以下と20 ms超が隣接する場合に、
            # セル境界へ縦線を描画する
            if (
                left_is_feasible
                != right_is_feasible
            ):
                x_position = (
                    column_index
                    - 0.5
                )

                axis.plot(
                    [
                        x_position,
                        x_position,
                    ],
                    [
                        row_index - 0.5,
                        row_index + 0.5,
                    ],
                    color="white",
                    linewidth=2.5,
                    solid_capstyle="butt",
                    zorder=5,
                )

                boundary_was_drawn = True

    # --------------------------------------------------------
    # 上下に隣接するセルの境界
    # マス目に沿った横線を描画する
    # --------------------------------------------------------

    for row_index in range(
        1,
        number_of_rows,
    ):
        for column_index in range(
            number_of_columns
        ):
            lower_is_finite = finite_mask[
                row_index - 1,
                column_index,
            ]

            upper_is_finite = finite_mask[
                row_index,
                column_index,
            ]

            # NaNを含むセル間には線を描かない
            if not (
                lower_is_finite
                and upper_is_finite
            ):
                continue

            lower_is_feasible = feasible_grid[
                row_index - 1,
                column_index,
            ]

            upper_is_feasible = feasible_grid[
                row_index,
                column_index,
            ]

            # 20 ms以下と20 ms超が隣接する場合に、
            # セル境界へ横線を描画する
            if (
                lower_is_feasible
                != upper_is_feasible
            ):
                y_position = (
                    row_index
                    - 0.5
                )

                axis.plot(
                    [
                        column_index - 0.5,
                        column_index + 0.5,
                    ],
                    [
                        y_position,
                        y_position,
                    ],
                    color="white",
                    linewidth=2.5,
                    solid_capstyle="butt",
                    zorder=5,
                )

                boundary_was_drawn = True

def plot_heatmap(
    value_grid: np.ndarray,
    K_values: list,
    N_values: list,
    title: str,
    colorbar_label: str,
    output_path: Path,
    color_map: str,
    minimum_value: float | None = None,
    maximum_value: float | None = None,
    time_grid: np.ndarray | None = None,
) -> None:
    """1枚のCartPoleヒートマップを作成して保存する。"""

    figure, axis = plt.subplots(
        figsize=(8, 7)
    )

    # NaNセルを描画対象から除外する
    masked_grid = np.ma.masked_invalid(
        value_grid
    )

    image = axis.imshow(
        masked_grid,
        origin="lower",
        aspect="equal",
        cmap=color_map,
        vmin=minimum_value,
        vmax=maximum_value,
    )

    # 横軸：予測ステップ数N
    # N=10、20、...、100を表示する
    x_tick_positions = [
        index
        for index, N in enumerate(N_values)
        if N % 10 == 0
    ]

    x_tick_labels = [
        N_values[index]
        for index in x_tick_positions
    ]

    axis.set_xticks(
        x_tick_positions
    )

    axis.set_xticklabels(
        x_tick_labels
    )

    axis.set_xlabel(
        "Prediction steps (N)"
    )

    # 縦軸：サンプル数K
    # 20条件のうち2条件ごとに表示する
    y_tick_positions = np.arange(
        1,
        len(K_values),
        2,
    )

    y_tick_labels = [
        K_values[index]
        for index in y_tick_positions
    ]

    axis.set_yticks(
        y_tick_positions
    )

    axis.set_yticklabels(
        y_tick_labels
    )

    axis.set_ylabel(
        "Number of samples (K)"
    )

    axis.set_title(title)

    colorbar = figure.colorbar(
        image,
        ax=axis,
        shrink=0.80,
        pad=0.04,
    )

    colorbar.set_label(
        colorbar_label
    )

    # 計算時間格子が渡された場合は、
    # 20 msの実時間境界を重ねる
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
        f"[INFO] Saved figure: "
        f"{output_path}"
    )

    plt.close(figure)


def main() -> None:
    """CartPoleの3種類のヒートマップを作成する。"""

    # 画像保存フォルダーを作成する
    FIGURE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    # CartPoleのsummary CSVを読み込む
    all_results = load_summary_results()

    # 指定した予測周期の結果だけを取り出す
    results = filter_target_results(
        all_results
    )

    # 平均総コストの格子を作成する
    (
        K_values,
        N_values,
        cost_grid,
    ) = create_result_grid(
        results=results,
        value_name="mean_total_cost",
    )

    # 成功率の格子を作成する
    (
        _,
        _,
        success_grid,
    ) = create_result_grid(
        results=results,
        value_name="success_rate",
    )

    # 平均計算時間の格子を作成する
    (
        _,
        _,
        time_grid,
    ) = create_result_grid(
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
            "CartPole Mean Total Cost\n"
            f"Prediction dt = {dt_ms} ms"
        ),
        colorbar_label=(
            "Mean total cost"
        ),
        output_path=(
            FIGURE_DIRECTORY
            / f"total_cost_dt{dt_ms:03d}.png"
        ),
        color_map="turbo",
        minimum_value=COST_MINIMUM_VALUE,
        maximum_value=COST_MAXIMUM_VALUE,
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
            "CartPole Task Success Rate\n"
            f"Prediction dt = {dt_ms} ms"
        ),
        colorbar_label=(
            "Success rate"
        ),
        output_path=(
            FIGURE_DIRECTORY
            / f"success_rate_dt{dt_ms:03d}.png"
        ),
        color_map="turbo_r",
        minimum_value=0.0,
        maximum_value=1.0,
        time_grid=time_grid,
    )

    # --------------------------------------------------------
    # 平均計算時間
    # --------------------------------------------------------

    plot_heatmap(
        value_grid=time_grid,
        K_values=K_values,
        N_values=N_values,
        title=(
            "CartPole Mean Computation Time\n"
            f"Prediction dt = {dt_ms} ms"
        ),
        colorbar_label=(
            "Mean computation time [ms]"
        ),
        output_path=(
            FIGURE_DIRECTORY
            / f"compute_time_dt{dt_ms:03d}.png"
        ),
        color_map="turbo",
        minimum_value=0,
        maximum_value=20,

        # 計算時間図にも20 ms境界を重ねる
        # 白線が短すぎて不要な場合はNoneへ変更する
        time_grid=time_grid,
    )

    print()
    print(
        "[INFO] All CartPole heatmaps "
        "were generated."
    )


if __name__ == "__main__":
    main()