import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# このファイルは scripts/analysis にあるため、
# parents[2]がリポジトリのルートになる
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

# 集約済み実験結果の保存場所
SUMMARY_RESULTS_PATH = (
    REPOSITORY_ROOT
    / "data"
    / "experiment_results"
    / "pendulum"
    / "summary_results.csv"
)

# 回帰結果の保存先
REGRESSION_DIRECTORY = (
    REPOSITORY_ROOT
    / "data"
    / "experiment_results"
    / "pendulum_fixed"
    / "regression"
)

# 回帰式による推定時間図の保存先
FIGURE_DIRECTORY = (
    REPOSITORY_ROOT
    / "media"
    / "experiment_figures"
    / "pendulum"
)


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
            results.append(
                {
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
            )

    return results


def filter_results(
    results: list[dict],
    prediction_delta_t: float,
) -> list:
    """指定された予測周期の結果だけを抽出する。"""

    filtered_results = [
        result
        for result in results
        if result["environment"] == "pendulum"
        and np.isclose(
            result["prediction_delta_t"],
            prediction_delta_t,
        )
    ]

    if not filtered_results:
        raise ValueError(
            "指定された予測周期のデータがありません。\n"
            f"prediction_delta_t={prediction_delta_t}"
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
    # t_calc = a * (K * N) + b * K + c * N + d
    design_matrix = np.column_stack(
        (
            K * N,
            K,
            N,
            np.ones_like(K),
        )
    )

    # 最小二乗法で係数a、b、c、dを求める
    coefficients, _, _, _ = np.linalg.lstsq(
        design_matrix,
        measured_time_ms,
        rcond=None,
    )

    a, b, c, d = coefficients

    # 回帰式から各条件の計算時間を推定する
    estimated_time_ms = (
        design_matrix @ coefficients
    )

    # 残差平方和
    residual_sum_of_squares = np.sum(
        (
            measured_time_ms
            - estimated_time_ms
        ) ** 2
    )

    # 全平方和
    total_sum_of_squares = np.sum(
        (
            measured_time_ms
            - np.mean(measured_time_ms)
        ) ** 2
    )

    # 決定係数R^2
    if np.isclose(total_sum_of_squares, 0.0):
        r_squared = 1.0
    else:
        r_squared = (
            1.0
            - residual_sum_of_squares
            / total_sum_of_squares
        )

    # 平均絶対誤差
    mean_absolute_error_ms = np.mean(
        np.abs(
            measured_time_ms
            - estimated_time_ms
        )
    )

    # 二乗平均平方根誤差
    root_mean_squared_error_ms = np.sqrt(
        np.mean(
            (
                measured_time_ms
                - estimated_time_ms
            ) ** 2
        )
    )

    return {
        "a": float(a),
        "b": float(b),
        "c": float(c),
        "d": float(d),
        "r_squared": float(r_squared),
        "mean_absolute_error_ms": float(
            mean_absolute_error_ms
        ),
        "root_mean_squared_error_ms": float(
            root_mean_squared_error_ms
        ),
        "K": K,
        "N": N,
        "measured_time_ms": measured_time_ms,
        "estimated_time_ms": estimated_time_ms,
    }


def save_regression_summary(
    regression_result: dict,
    prediction_delta_t: float,
) -> Path:
    """回帰係数と評価指標をJSONへ保存する。"""

    REGRESSION_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    dt_ms = int(
        round(
            prediction_delta_t * 1000
        )
    )

    output_path = (
        REGRESSION_DIRECTORY
        / f"computation_time_regression_dt{dt_ms:03d}.json"
    )

    output_data = {
        "environment": "pendulum",
        "prediction_delta_t": prediction_delta_t,
        "equation": (
            "t_calc_ms = a * (K * N) + b * K + c * N + d"
        ),
        "a": regression_result["a"],
        "b": regression_result["b"],
        "c": regression_result["c"],
        "d": regression_result["d"],
        "r_squared": regression_result["r_squared"],
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
    prediction_delta_t: float,
) -> Path:
    """各条件の実測時間・推定時間・誤差をCSVへ保存する。"""

    REGRESSION_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    dt_ms = int(
        round(
            prediction_delta_t * 1000
        )
    )

    output_path = (
        REGRESSION_DIRECTORY
        / f"computation_time_predictions_dt{dt_ms:03d}.csv"
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
            "prediction_delta_t",
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
                    "environment": "pendulum",
                    "K": result["K"],
                    "N": result["N"],
                    "prediction_delta_t": (
                        prediction_delta_t
                    ),
                    "measured_time_ms": (
                        measured_time
                    ),
                    "estimated_time_ms": float(
                        estimated_time
                    ),
                    "error_ms": float(error),
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
    prediction_delta_t: float,
) -> Path:
    """回帰式による推定計算時間のヒートマップを作る。"""

    FIGURE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    dt_ms = int(
        round(
            prediction_delta_t * 1000
        )
    )

    figure, axis = plt.subplots(
        figsize=(8, 7)
    )

    image = axis.imshow(
        estimated_time_grid,
        origin="lower",
        aspect="auto",
        cmap="turbo",
    )

    # 横軸は10、20、...、100を表示する
    x_tick_positions = [
        index
        for index, N in enumerate(N_values)
        if N % 10 == 0
    ]

    axis.set_xticks(x_tick_positions)
    axis.set_xticklabels(
        [
            N_values[index]
            for index in x_tick_positions
        ]
    )

    # 縦軸は卒論図に近い間隔で表示する
    y_tick_positions = np.arange(
        1,
        len(K_values),
        2,
    )

    axis.set_yticks(y_tick_positions)
    axis.set_yticklabels(
        [
            K_values[index]
            for index in y_tick_positions
        ]
    )

    axis.set_xlabel(
        "Prediction steps N"
    )
    axis.set_ylabel(
        "Number of samples K"
    )

    axis.set_title(
        "Pendulum Estimated Computation Time\n"
        f"Prediction dt = {dt_ms} ms"
    )

    colorbar = figure.colorbar(
        image,
        ax=axis,
    )

    colorbar.set_label(
        "Estimated computation time [ms]"
    )

    # 推定時間に20 msをまたぐ領域がある場合だけ白線を引く
    minimum_time = np.min(
        estimated_time_grid
    )
    maximum_time = np.max(
        estimated_time_grid
    )

    if minimum_time <= 20.0 <= maximum_time:
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
            levels=[20.0],
            colors="white",
            linewidths=2.0,
        )

        axis.clabel(
            contour,
            inline=True,
            fontsize=9,
            fmt={20.0: "20 ms"},
        )
    else:
        print(
            "[INFO] Estimated times do not cross "
            "the 20 ms limit."
        )

    figure.tight_layout()

    output_path = (
        FIGURE_DIRECTORY
        / f"estimated_compute_time_dt{dt_ms:03d}.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def parse_arguments() -> argparse.Namespace:
    """コマンドライン引数を読み込む。"""

    parser = argparse.ArgumentParser(
        description=(
            "Fit Pendulum computation time regression."
        )
    )

    parser.add_argument(
        "--prediction-dt",
        type=float,
        default=0.02,
        help=(
            "Prediction time step in seconds. "
            "Default: 0.02"
        ),
    )

    return parser.parse_args()


def main() -> None:
    """回帰式を算出し、結果を保存する。"""

    arguments = parse_arguments()

    prediction_delta_t = (
        arguments.prediction_dt
    )

    all_results = load_summary_results()

    results = filter_results(
        results=all_results,
        prediction_delta_t=prediction_delta_t,
    )

    regression_result = fit_regression(
        results=results,
    )

    json_path = save_regression_summary(
        regression_result=regression_result,
        prediction_delta_t=prediction_delta_t,
    )

    prediction_csv_path = save_prediction_results(
        results=results,
        regression_result=regression_result,
        prediction_delta_t=prediction_delta_t,
    )

    (
        K_values,
        N_values,
        estimated_time_grid,
    ) = create_estimated_time_grid(
        results=results,
        regression_result=regression_result,
    )

    figure_path = plot_estimated_time_heatmap(
        K_values=K_values,
        N_values=N_values,
        estimated_time_grid=estimated_time_grid,
        prediction_delta_t=prediction_delta_t,
    )

    print()
    print("[RESULT] Computation time regression")
    print(
        "Equation:"
    )
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
    print()
    print(
        f"[INFO] Saved JSON: {json_path}"
    )
    print(
        f"[INFO] Saved CSV : {prediction_csv_path}"
    )
    print(
        f"[INFO] Saved figure: {figure_path}"
    )


if __name__ == "__main__":
    main()