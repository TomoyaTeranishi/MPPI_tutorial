# CartPoleの平均計算時間について、
# KとNを説明変数とする回帰式を最小二乗法で求める
#
# 回帰式：
# t_calc_ms = a * (K * N) + b * K + c * N + d

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 入力CSVと保存先
# ============================================================

# このファイルはscripts/analysisにあるため、
# parents[2]がリポジトリのルートになる
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


# CartPoleの集約済み実験結果
SUMMARY_RESULTS_PATH = (
    REPOSITORY_ROOT
    / "data"
    / "experiment_results"
    / "cartpole"
    / "summary_results.csv"
)


# CartPoleの回帰結果の保存先
REGRESSION_DIRECTORY = (
    REPOSITORY_ROOT
    / "data"
    / "experiment_results"
    / "cartpole"
    / "regression"
)


# 回帰式による推定計算時間図の保存先
FIGURE_DIRECTORY = (
    REPOSITORY_ROOT
    / "media"
    / "experiment_figures"
    / "cartpole"
)


# ============================================================
# 今回解析する予測周期
# ============================================================

# 実行コマンド：
# python -m scripts.analysis.fit_cartpole_computation_time

# 20 msの場合は0.02
# 40 msの場合は0.04
# 200 msの場合は0.20
TARGET_PREDICTION_DELTA_T = 0.02


# 卒論条件では同一条件を5seedで実行する
EXPECTED_NUMBER_OF_TRIALS = 5


# 実環境の制御周期に対応する実時間制約
REAL_TIME_LIMIT_MS = 20.0


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
                "mean_compute_time_ms": float(
                    row["mean_compute_time_ms"]
                ),
            }

            results.append(result)

    return results


def filter_results(
    results: list[dict],
) -> list:
    """指定した予測周期のCartPole結果だけを抽出する。"""

    filtered_results = [
        result
        for result in results
        if result["environment"] == "cartpole"
        and np.isclose(
            result["prediction_delta_t"],
            TARGET_PREDICTION_DELTA_T,
        )
        and result["number_of_trials"]
        == EXPECTED_NUMBER_OF_TRIALS
        and np.isfinite(
            result["mean_compute_time_ms"]
        )
    ]

    if not filtered_results:
        raise ValueError(
            "指定した予測周期について、"
            "5試行が揃ったCartPoleデータがありません。\n"
            "prediction_delta_t="
            f"{TARGET_PREDICTION_DELTA_T}"
        )

    # CSV出力と回帰結果の順序を一定にする
    filtered_results.sort(
        key=lambda result: (
            result["K"],
            result["N"],
        )
    )

    return filtered_results


def fit_regression(
    results: list[dict],
) -> dict:
    """計算時間の回帰式を最小二乗法で求める。"""

    K = np.array(
        [
            result["K"]
            for result in results
        ],
        dtype=float,
    )

    N = np.array(
        [
            result["N"]
            for result in results
        ],
        dtype=float,
    )

    measured_time_ms = np.array(
        [
            result["mean_compute_time_ms"]
            for result in results
        ],
        dtype=float,
    )

    # 回帰式：
    # t_calc_ms = a * (K * N) + b * K + c * N + d
    design_matrix = np.column_stack(
        (
            K * N,
            K,
            N,
            np.ones_like(K),
        )
    )

    # 最小二乗法で係数a、b、c、dを求める
    coefficients, _, rank, singular_values = (
        np.linalg.lstsq(
            design_matrix,
            measured_time_ms,
            rcond=None,
        )
    )

    a, b, c, d = coefficients

    # 回帰式から各条件の計算時間を推定する
    estimated_time_ms = (
        design_matrix
        @ coefficients
    )

    # 実測値と推定値の差
    errors_ms = (
        measured_time_ms
        - estimated_time_ms
    )

    # 残差平方和
    residual_sum_of_squares = float(
        np.sum(
            errors_ms**2
        )
    )

    # 全平方和
    total_sum_of_squares = float(
        np.sum(
            (
                measured_time_ms
                - np.mean(measured_time_ms)
            ) ** 2
        )
    )

    # 決定係数R^2
    if np.isclose(
        total_sum_of_squares,
        0.0,
    ):
        r_squared = 1.0
    else:
        r_squared = (
            1.0
            - residual_sum_of_squares
            / total_sum_of_squares
        )

    # 平均絶対誤差
    mean_absolute_error_ms = float(
        np.mean(
            np.abs(errors_ms)
        )
    )

    # 二乗平均平方根誤差
    root_mean_squared_error_ms = float(
        np.sqrt(
            np.mean(
                errors_ms**2
            )
        )
    )

    # 最大絶対誤差
    maximum_absolute_error_ms = float(
        np.max(
            np.abs(errors_ms)
        )
    )

    return {
        "a": float(a),
        "b": float(b),
        "c": float(c),
        "d": float(d),
        "rank": int(rank),
        "singular_values": singular_values,
        "number_of_conditions": len(results),
        "residual_sum_of_squares": (
            residual_sum_of_squares
        ),
        "r_squared": float(r_squared),
        "mean_absolute_error_ms": (
            mean_absolute_error_ms
        ),
        "root_mean_squared_error_ms": (
            root_mean_squared_error_ms
        ),
        "maximum_absolute_error_ms": (
            maximum_absolute_error_ms
        ),
        "K": K,
        "N": N,
        "measured_time_ms": measured_time_ms,
        "estimated_time_ms": estimated_time_ms,
        "errors_ms": errors_ms,
    }


def save_regression_summary(
    regression_result: dict,
) -> Path:
    """回帰係数と評価指標をJSONへ保存する。"""

    REGRESSION_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    dt_ms = int(
        round(
            TARGET_PREDICTION_DELTA_T
            * 1000
        )
    )

    output_path = (
        REGRESSION_DIRECTORY
        / (
            "computation_time_regression_"
            f"dt{dt_ms:03d}.json"
        )
    )

    output_data = {
        "environment": "cartpole",
        "prediction_delta_t": (
            TARGET_PREDICTION_DELTA_T
        ),
        "expected_number_of_trials": (
            EXPECTED_NUMBER_OF_TRIALS
        ),
        "number_of_conditions": (
            regression_result[
                "number_of_conditions"
            ]
        ),
        "equation": (
            "t_calc_ms = "
            "a * (K * N) + b * K + c * N + d"
        ),
        "a": regression_result["a"],
        "b": regression_result["b"],
        "c": regression_result["c"],
        "d": regression_result["d"],
        "r_squared": (
            regression_result[
                "r_squared"
            ]
        ),
        "mean_absolute_error_ms": (
            regression_result[
                "mean_absolute_error_ms"
            ]
        ),
        "root_mean_squared_error_ms": (
            regression_result[
                "root_mean_squared_error_ms"
            ]
        ),
        "maximum_absolute_error_ms": (
            regression_result[
                "maximum_absolute_error_ms"
            ]
        ),
        "residual_sum_of_squares": (
            regression_result[
                "residual_sum_of_squares"
            ]
        ),
    }

    with output_path.open(
        mode="w",
        encoding="utf-8",
    ) as json_file:
        json.dump(
            output_data,
            json_file,
            ensure_ascii=False,
            indent=4,
        )

    return output_path


def save_prediction_results(
    results: list[dict],
    regression_result: dict,
) -> Path:
    """各条件の実測時間、推定時間、誤差をCSVへ保存する。"""

    REGRESSION_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    dt_ms = int(
        round(
            TARGET_PREDICTION_DELTA_T
            * 1000
        )
    )

    output_path = (
        REGRESSION_DIRECTORY
        / (
            "computation_time_predictions_"
            f"dt{dt_ms:03d}.csv"
        )
    )

    estimated_times = regression_result[
        "estimated_time_ms"
    ]

    with output_path.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        fieldnames = [
            "environment",
            "K",
            "N",
            "control_delta_t",
            "prediction_delta_t",
            "number_of_trials",
            "measured_time_ms",
            "estimated_time_ms",
            "error_ms",
            "absolute_error_ms",
        ]

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result, estimated_time in zip(
            results,
            estimated_times,
        ):
            measured_time = result[
                "mean_compute_time_ms"
            ]

            error = (
                measured_time
                - estimated_time
            )

            writer.writerow(
                {
                    "environment": "cartpole",
                    "K": result["K"],
                    "N": result["N"],
                    "control_delta_t": (
                        result[
                            "control_delta_t"
                        ]
                    ),
                    "prediction_delta_t": (
                        TARGET_PREDICTION_DELTA_T
                    ),
                    "number_of_trials": (
                        result[
                            "number_of_trials"
                        ]
                    ),
                    "measured_time_ms": (
                        measured_time
                    ),
                    "estimated_time_ms": float(
                        estimated_time
                    ),
                    "error_ms": float(
                        error
                    ),
                    "absolute_error_ms": float(
                        abs(error)
                    ),
                }
            )

    return output_path


def create_estimated_time_grid(
    results: list[dict],
    regression_result: dict,
) -> tuple:
    """回帰式による推定計算時間の二次元格子を作る。"""

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

    K_grid, N_grid = np.meshgrid(
        K_values,
        N_values,
        indexing="ij",
    )

    a = regression_result["a"]
    b = regression_result["b"]
    c = regression_result["c"]
    d = regression_result["d"]

    estimated_time_grid = (
        a * K_grid * N_grid
        + b * K_grid
        + c * N_grid
        + d
    )

    return (
        K_values,
        N_values,
        estimated_time_grid,
    )


def plot_estimated_time_heatmap(
    K_values: list,
    N_values: list,
    estimated_time_grid: np.ndarray,
) -> Path:
    """回帰式による推定計算時間のヒートマップを作る。"""

    FIGURE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    dt_ms = int(
        round(
            TARGET_PREDICTION_DELTA_T
            * 1000
        )
    )

    figure, axis = plt.subplots(
        figsize=(8, 7)
    )

    # 推定時間の最大値を取得する
    maximum_estimated_time = float(
        np.max(
            estimated_time_grid
        )
    )

    # 20 msまたは推定最大値のうち、
    # 大きい方をカラーバー上限とする
    colorbar_maximum = max(
        REAL_TIME_LIMIT_MS,
        maximum_estimated_time,
    )

    image = axis.imshow(
        estimated_time_grid,
        origin="lower",
        aspect="equal",
        cmap="turbo",
        vmin=0.0,
        vmax=colorbar_maximum,
    )

    # 横軸はN=10、20、...、100を表示する
    x_tick_positions = [
        index
        for index, N in enumerate(N_values)
        if N % 10 == 0
    ]

    axis.set_xticks(
        x_tick_positions
    )

    axis.set_xticklabels(
        [
            N_values[index]
            for index in x_tick_positions
        ]
    )

    # 縦軸は2条件ごとに表示する
    y_tick_positions = np.arange(
        1,
        len(K_values),
        2,
    )

    axis.set_yticks(
        y_tick_positions
    )

    axis.set_yticklabels(
        [
            K_values[index]
            for index in y_tick_positions
        ]
    )

    axis.set_xlabel(
        "Prediction steps (N)"
    )

    axis.set_ylabel(
        "Number of samples (K)"
    )

    axis.set_title(
        "CartPole Estimated Computation Time\n"
        f"Prediction dt = {dt_ms} ms"
    )

    colorbar = figure.colorbar(
        image,
        ax=axis,
        shrink=0.80,
        pad=0.04,
    )

    colorbar.set_label(
        "Estimated computation time [ms]"
    )

    # 推定時間が20 msをまたぐ場合のみ白線を描く
    minimum_time = float(
        np.min(
            estimated_time_grid
        )
    )

    maximum_time = float(
        np.max(
            estimated_time_grid
        )
    )

    if (
        minimum_time
        <= REAL_TIME_LIMIT_MS
        <= maximum_time
    ):
        x_grid, y_grid = np.meshgrid(
            np.arange(
                estimated_time_grid.shape[1]
            ),
            np.arange(
                estimated_time_grid.shape[0]
            ),
        )

        contour = axis.contour(
            x_grid,
            y_grid,
            estimated_time_grid,
            levels=[
                REAL_TIME_LIMIT_MS
            ],
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
    else:
        print(
            "[INFO] Estimated times do not cross "
            "the 20 ms limit. "
            "Estimated range: "
            f"{minimum_time:.3f} to "
            f"{maximum_time:.3f} ms"
        )

    figure.tight_layout()

    output_path = (
        FIGURE_DIRECTORY
        / (
            "estimated_compute_time_"
            f"dt{dt_ms:03d}.png"
        )
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def main() -> None:
    """CartPoleの計算時間回帰を実行する。"""

    all_results = load_summary_results()

    results = filter_results(
        results=all_results,
    )

    regression_result = fit_regression(
        results=results,
    )

    json_path = save_regression_summary(
        regression_result=regression_result,
    )

    prediction_csv_path = (
        save_prediction_results(
            results=results,
            regression_result=regression_result,
        )
    )

    (
        K_values,
        N_values,
        estimated_time_grid,
    ) = create_estimated_time_grid(
        results=results,
        regression_result=regression_result,
    )

    figure_path = (
        plot_estimated_time_heatmap(
            K_values=K_values,
            N_values=N_values,
            estimated_time_grid=(
                estimated_time_grid
            ),
        )
    )

    print()
    print(
        "[RESULT] CartPole computation "
        "time regression"
    )

    print(
        "Number of conditions : "
        f"{regression_result['number_of_conditions']}"
    )

    print(
        "Prediction delta t   : "
        f"{TARGET_PREDICTION_DELTA_T:.3f} s"
    )

    print()

    print("Equation:")

    print(
        "t_calc_ms = "
        f"{regression_result['a']:.12e} * (K * N)"
        f" + {regression_result['b']:.12e} * K"
        f" + {regression_result['c']:.12e} * N"
        f" + {regression_result['d']:.12e}"
    )

    print()

    print(
        "a                  : "
        f"{regression_result['a']:.12e}"
    )

    print(
        "b                  : "
        f"{regression_result['b']:.12e}"
    )

    print(
        "c                  : "
        f"{regression_result['c']:.12e}"
    )

    print(
        "d                  : "
        f"{regression_result['d']:.12e}"
    )

    print(
        "R squared          : "
        f"{regression_result['r_squared']:.6f}"
    )

    print(
        "Mean absolute error: "
        f"{regression_result['mean_absolute_error_ms']:.6f} ms"
    )

    print(
        "RMSE               : "
        f"{regression_result['root_mean_squared_error_ms']:.6f} ms"
    )

    print(
        "Maximum abs. error : "
        f"{regression_result['maximum_absolute_error_ms']:.6f} ms"
    )

    print()

    print(
        f"[INFO] Saved JSON  : {json_path}"
    )

    print(
        f"[INFO] Saved CSV   : {prediction_csv_path}"
    )

    print(
        f"[INFO] Saved figure: {figure_path}"
    )


if __name__ == "__main__":
    main()