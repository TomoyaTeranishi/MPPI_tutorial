#　条件を変えながら繰り替えし、結果を保存する
import csv
from pathlib import Path

import numpy as np

from scripts.experiments.run_pendulum_trial import run_pendulum_trial

# 卒論の基準実験で使用するサンプル数K
K_VALUES = [
    5,
    110,
    215,
    320,
    425,
    530,
    635,
    740,
    845,
    950,
    1055,
    1160,
    1265,
    1370,
    1475,
    1580,
    1685,
    1790,
    1895,
    2000,
]

# 卒論の基準実験で使用する予測ステップ数N　5,10,....,95,100
N_VALUES = list(range(5, 101, 5))

# 同じK,N,Δtについて、5種類の探索noiseで評価する　成功率の割合はこのseedの成功数に相当する
SEEDS = [0, 1, 2, 3, 4]

# 実環境の制御周期Tsを20 msに固定する
CONTROL_DELTA_T = 0.02

#python -m scripts.experiments.run_pendulum_sweep 　実行コマンド
# MPPI内部モデルの予測周期Δt
PREDICTION_DELTA_T = 0.20

# 1試行当たりの実行時間
SIMULATION_TIME = 7.5


# ============================================================
# CSVファイルの保存先
# ============================================================

# このファイルは、
# リポジトリ/scripts/experiments/run_pendulum_sweep.py
# にあるため、parents[2]がリポジトリのルートになる
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

# 実験結果を保存するフォルダー
RESULT_DIRECTORY = (
    REPOSITORY_ROOT
    / "data"
    / "experiment_results"
    / "pendulum_fixed"
)

# 各試行の結果を保存するCSV
RAW_RESULTS_PATH = (
    RESULT_DIRECTORY
    / "raw_results.csv"
)

# 5試行を集約した結果を保存するCSV
SUMMARY_RESULTS_PATH = (
    RESULT_DIRECTORY
    / "summary_results.csv"
)


# ============================================================
# CSVへ保存する列名
# ============================================================

# 単一試行結果の列
RAW_FIELDNAMES = [
    "environment",
    "K",
    "N",
    "control_delta_t",
    "prediction_delta_t",
    "seed",
    "simulation_steps",
    "total_cost",
    "success",
    "mean_compute_time_ms",
    "max_compute_time_ms",
]

# 条件ごとの集約結果の列
SUMMARY_FIELDNAMES = [
    "environment",
    "K",
    "N",
    "control_delta_t",
    "prediction_delta_t",
    "number_of_trials",
    "mean_total_cost",
    "std_total_cost",
    "success_rate",
    "mean_compute_time_ms",
    "std_compute_time_ms",
    "max_compute_time_ms",
]


def create_result_directory() -> None:
    """実験結果を保存するフォルダーを作成する。"""

    RESULT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )


def make_trial_key(
    K: int,
    N: int,
    control_delta_t: float,
    prediction_delta_t: float,
    seed: int,
) -> tuple:
    """試行済みかを判定するためのキーを作成する。"""

    # 浮動小数点数をそのまま比較すると誤差が生じる可能性があるため、
    # 時間刻みは小数点以下10桁へ丸める
    return (
        K,
        N,
        round(control_delta_t, 10),
        round(prediction_delta_t, 10),
        seed,
    )


def append_raw_result(
    result: dict,
) -> None:
    """単一試行の結果をraw_results.csvへ追記する。"""

    # CSVがまだ存在していない場合は、最初にヘッダーを記録する
    file_exists = RAW_RESULTS_PATH.exists()

    with RAW_RESULTS_PATH.open(
        mode="a",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=RAW_FIELDNAMES,
        )

        if not file_exists:
            writer.writeheader()

        # run_pendulum_trial()の戻り値から、
        # raw CSVに必要な項目だけを取り出す
        row = {
            field_name: result[field_name]
            for field_name in RAW_FIELDNAMES
        }

        writer.writerow(row)


def load_completed_trials() -> set:
    """raw_results.csvから、既に完了している試行を読み込む。"""

    # CSVがまだない場合は、完了済み試行なしとして空集合を返す
    if not RAW_RESULTS_PATH.exists():
        return set()

    completed_trials = set()

    with RAW_RESULTS_PATH.open(
        mode="r",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            trial_key = make_trial_key(
                K=int(row["K"]),
                N=int(row["N"]),
                control_delta_t=float(
                    row["control_delta_t"]
                ),
                prediction_delta_t=float(
                    row["prediction_delta_t"]
                ),
                seed=int(row["seed"]),
            )

            completed_trials.add(trial_key)

    return completed_trials


def load_raw_results() -> list:
    """raw_results.csvを読み込み、辞書のリストとして返す。"""

    if not RAW_RESULTS_PATH.exists():
        return []

    results = []

    with RAW_RESULTS_PATH.open(
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
                "seed": int(row["seed"]),
                "simulation_steps": int(
                    row["simulation_steps"]
                ),
                "total_cost": float(
                    row["total_cost"]
                ),
                "success": (
                    row["success"].strip().lower()
                    == "true"
                ),
                "mean_compute_time_ms": float(
                    row["mean_compute_time_ms"]
                ),
                "max_compute_time_ms": float(
                    row["max_compute_time_ms"]
                ),
            }

            results.append(result)

    return results


def aggregate_results(
    raw_results: list[dict],
) -> list[dict]:
    """予測周期、K、Nごとに試行結果を集約する。"""

    summary_results = []

    # raw CSVに保存されている予測周期を昇順で取得する
    prediction_delta_t_values = sorted(
        {
            result["prediction_delta_t"]
            for result in raw_results
            if result["environment"] == "pendulum"
        }
    )

    for prediction_delta_t in prediction_delta_t_values:
        for K in K_VALUES:
            for N in N_VALUES:

                # 現在の予測周期、K、Nに一致する結果を抽出する
                matching_results = [
                    result
                    for result in raw_results
                    if result["environment"] == "pendulum"
                    and result["K"] == K
                    and result["N"] == N
                    and np.isclose(
                        result["control_delta_t"],
                        CONTROL_DELTA_T,
                    )
                    and np.isclose(
                        result["prediction_delta_t"],
                        prediction_delta_t,
                    )
                    and result["seed"] in SEEDS
                ]

                if not matching_results:
                    continue

                # seed順に並べる
                matching_results.sort(
                    key=lambda result: result["seed"]
                )

                total_costs = np.array(
                    [
                        result["total_cost"]
                        for result in matching_results
                    ],
                    dtype=float,
                )

                successes = np.array(
                    [
                        result["success"]
                        for result in matching_results
                    ],
                    dtype=float,
                )

                mean_compute_times = np.array(
                    [
                        result["mean_compute_time_ms"]
                        for result in matching_results
                    ],
                    dtype=float,
                )

                max_compute_times = np.array(
                    [
                        result["max_compute_time_ms"]
                        for result in matching_results
                    ],
                    dtype=float,
                )

                summary_results.append(
                    {
                        "environment": "pendulum",
                        "K": K,
                        "N": N,
                        "control_delta_t": CONTROL_DELTA_T,
                        "prediction_delta_t": prediction_delta_t,
                        "number_of_trials": len(matching_results),
                        "mean_total_cost": float(
                            np.mean(total_costs)
                        ),
                        "std_total_cost": float(
                            np.std(total_costs)
                        ),
                        "success_rate": float(
                            np.mean(successes)
                        ),
                        "mean_compute_time_ms": float(
                            np.mean(mean_compute_times)
                        ),
                        "std_compute_time_ms": float(
                            np.std(mean_compute_times)
                        ),
                        "max_compute_time_ms": float(
                            np.max(max_compute_times)
                        ),
                    }
                )

    return summary_results


def save_summary_results(
    summary_results: list[dict],
) -> None:
    """集約結果をsummary_results.csvへ保存する。"""

    with SUMMARY_RESULTS_PATH.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=SUMMARY_FIELDNAMES,
        )

        writer.writeheader()
        writer.writerows(summary_results)


def run_sweep() -> None:
    """小規模なK・Nパラメータ走査を実行する。"""

    # CSV保存用フォルダーを用意する
    create_result_directory()

    # 既に完了している試行を読み込む
    completed_trials = load_completed_trials()

    total_trial_count = (
        len(K_VALUES)
        * len(N_VALUES)
        * len(SEEDS)
    )

    trial_counter = 0

    print("[INFO] Start Pendulum parameter sweep")
    print(
        f"[INFO] Total trials: "
        f"{total_trial_count}"
    )
    print(
        f"[INFO] Raw CSV: "
        f"{RAW_RESULTS_PATH}"
    )
    print(
        f"[INFO] Summary CSV: "
        f"{SUMMARY_RESULTS_PATH}"
    )
    print()

    for K in K_VALUES:
        for N in N_VALUES:
            for seed in SEEDS:

                trial_counter += 1

                trial_key = make_trial_key(
                    K=K,
                    N=N,
                    control_delta_t=CONTROL_DELTA_T,
                    prediction_delta_t=PREDICTION_DELTA_T,
                    seed=seed,
                )

                # 同じ条件が既にCSVへ保存されている場合はスキップする
                if trial_key in completed_trials:
                    print(
                        f"[SKIP] "
                        f"{trial_counter}/{total_trial_count} "
                        f"prediction_dt={PREDICTION_DELTA_T:.2f}, "
                        f"K={K}, N={N}, seed={seed}"
                    )
                    continue

                print(
                    f"[RUN ] "
                    f"{trial_counter}/{total_trial_count} "
                    f"prediction_dt={PREDICTION_DELTA_T:.2f}, "
                    f"K={K}, N={N}, seed={seed}"
                )

                # 単一試行を実行する
                result = run_pendulum_trial(
                    K=K,
                    N=N,
                    prediction_delta_t=PREDICTION_DELTA_T,
                    seed=seed,
                    control_delta_t=CONTROL_DELTA_T,
                    simulation_time=SIMULATION_TIME,
                )

                # 1試行が終了するたびに即座に保存する
                append_raw_result(result)

                # この実行中にも完了済みとして登録する
                completed_trials.add(trial_key)

                print(
                    f"       "
                    f"cost={result['total_cost']:.6f}, "
                    f"success={result['success']}, "
                    f"mean_time="
                    f"{result['mean_compute_time_ms']:.6f} ms"
                )

    # 全試行の終了後、raw CSVを読み直す
    raw_results = load_raw_results()

    # K、Nごとに5試行を集約する
    summary_results = aggregate_results(
        raw_results=raw_results,
    )

    # 集約結果をsummary CSVへ保存する
    save_summary_results(
        summary_results=summary_results,
    )

    print()
    print("[SUMMARY] Pendulum parameter sweep")

    for summary in summary_results:
        print(
            f"K={summary['K']}, "
            f"N={summary['N']}, "
            f"trials={summary['number_of_trials']}, "
            f"mean_cost="
            f"{summary['mean_total_cost']:.6f}, "
            f"std_cost="
            f"{summary['std_total_cost']:.6f}, "
            f"success_rate="
            f"{summary['success_rate']:.2f}, "
            f"mean_time="
            f"{summary['mean_compute_time_ms']:.6f} ms"
        )

    print()
    print("[INFO] Parameter sweep completed.")


if __name__ == "__main__":
    run_sweep()