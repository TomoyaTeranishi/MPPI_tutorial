import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ============================================================
# 日本語フォントの設定
# ============================================================

available_fonts = {
    font.name
    for font in font_manager.fontManager.ttflist
}

japanese_font_candidates = [
    "Yu Gothic",
    "Meiryo",
    "Noto Sans CJK JP",
    "Noto Sans JP",
    "IPAexGothic",
]

for font_name in japanese_font_candidates:
    if font_name in available_fonts:
        plt.rcParams["font.family"] = font_name
        print(f"[INFO] Japanese font: {font_name}")
        break
else:
    print("[WARNING] Japanese font was not found.")

# マイナス記号の文字化けを防ぐ
plt.rcParams["axes.unicode_minus"] = False


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
    / "pendulum_fixed"
    / "summary_results.csv"
)

# ヒートマップの保存先
FIGURE_DIRECTORY = (
    REPOSITORY_ROOT
    / "media"
    / "experiment_figures"
    / "pendulum_fixed"
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
    """20 ms以下と20 ms超のセル境界を黒縁付き白線で描画する。"""

    time_grid = np.asarray(
        time_grid,
        dtype=float,
    )

    finite_mask = np.isfinite(time_grid)

    if not np.any(finite_mask):
        print(
            "[INFO] Computation-time grid "
            "has no finite values."
        )
        return

    # 20 ms超を1、20 ms以下を0として二値化する
    binary_mask = np.zeros(
        time_grid.shape,
        dtype=int,
    )

    binary_mask[finite_mask] = (
        time_grid[finite_mask]
        > REAL_TIME_LIMIT_MS
    ).astype(int)

    boundary_segments = []
    rows, columns = binary_mask.shape

    # 上下に隣接するセルの境界へ横線を追加する
    difference_y = np.diff(
        binary_mask,
        axis=0,
    )

    for row in range(rows - 1):
        for column in range(columns):
            adjacent_cells_are_finite = (
                finite_mask[row, column]
                and finite_mask[row + 1, column]
            )

            if (
                adjacent_cells_are_finite
                and difference_y[row, column] != 0
            ):
                boundary_segments.append([
                    (
                        column - 0.5,
                        row + 0.5,
                    ),
                    (
                        column + 0.5,
                        row + 0.5,
                    ),
                ])

    # 左右に隣接するセルの境界へ縦線を追加する
    difference_x = np.diff(
        binary_mask,
        axis=1,
    )

    for row in range(rows):
        for column in range(columns - 1):
            adjacent_cells_are_finite = (
                finite_mask[row, column]
                and finite_mask[row, column + 1]
            )

            if (
                adjacent_cells_are_finite
                and difference_x[row, column] != 0
            ):
                boundary_segments.append([
                    (
                        column + 0.5,
                        row - 0.5,
                    ),
                    (
                        column + 0.5,
                        row + 0.5,
                    ),
                ])

    if not boundary_segments:
        finite_values = time_grid[finite_mask]

        print(
            "[INFO] 20 ms boundary was not drawn. "
            "Measured time range: "
            f"{np.min(finite_values):.3f} to "
            f"{np.max(finite_values):.3f} ms"
        )
        return

    # 下地として黒い太線を描画する
    background_line = LineCollection(
        boundary_segments,
        colors="black",
        linewidths=5.0,
        zorder=9,
    )

    axis.add_collection(background_line)

    # 黒線の上へ白い細線を重ねる
    foreground_line = LineCollection(
        boundary_segments,
        colors="white",
        linewidths=2.5,
        zorder=10,
    )

    axis.add_collection(foreground_line)


def plot_heatmap(
    value_grid: np.ndarray,
    K_values: list,
    N_values: list,
    colorbar_label: str,
    output_path: Path,
    color_map: str,
    minimum_value: float | None = None,
    maximum_value: float | None = None,
    time_grid: np.ndarray | None = None,
) -> None:
    """1枚のPendulumヒートマップを作成して保存する。"""

    figure, axis = plt.subplots(
        figsize=(5, 5)
    )

    # NaNセルを描画対象から除外する
    masked_grid = np.ma.masked_invalid(
        value_grid
    )

    image = axis.imshow(
        masked_grid,
        origin="lower",
        aspect="equal",
        interpolation="nearest",
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

    axis.set_xticks(
        x_tick_positions
    )

    axis.set_xticklabels(
        x_tick_labels
    )

    axis.set_xlabel(
        "予測ステップ数：N",
        fontsize=16,
        fontweight="bold",
    )

    # 縦軸：状態軌道数K
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
        "状態軌道数：K",
        fontsize=16,
        fontweight="bold",
    )

    # 軸の目盛り線と目盛り数字
    axis.tick_params(
        axis="both",
        which="major",
        labelsize=11,
        width=1.0,
        length=6,
    )

    # 目盛り数字を太字にする
    for tick_label in axis.get_xticklabels():
        tick_label.set_fontweight("bold")

    for tick_label in axis.get_yticklabels():
        tick_label.set_fontweight("bold")

    # ヒートマップと同じ高さのカラーバーを作る
    divider = make_axes_locatable(axis)

    colorbar_axis = divider.append_axes(
        "right",
        size="5%",
        pad=0.20,
    )

    colorbar = figure.colorbar(
        image,
        cax=colorbar_axis,
    )

    colorbar.set_label(
        colorbar_label,
        fontsize=16,
        fontweight="bold",
    )

    colorbar.ax.tick_params(
        labelsize=11,
        width=1.0,
        length=6,
    )

    # カラーバーの目盛り数字を太字にする
    for tick_label in colorbar.ax.get_yticklabels():
        tick_label.set_fontweight("bold")

    # 平均計算時間が20 msとなる境界を描画する
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
        pad_inches=0.05,
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
        colorbar_label="総コスト",
        output_path=(
            FIGURE_DIRECTORY
            / f"total_cost_dt{dt_ms:03d}.png"
        ),
        color_map="jet",
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
        colorbar_label="タスク成功率",
        output_path=(
            FIGURE_DIRECTORY
            / f"success_rate_dt{dt_ms:03d}.png"
        ),
        color_map="jet_r",
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
    #     colorbar_label=(
    #         "平均計算時間 [ms]"
    #     ),
    #     output_path=(
    #         FIGURE_DIRECTORY
    #         / f"compute_time_dt{dt_ms:03d}.png"
    #     ),
    #     color_map="jet",
    #     time_grid=time_grid,
    # )

    print()
    print(
        "[INFO] All Pendulum heatmaps "
        "were generated."
    )


if __name__ == "__main__":
    main()