# 1つの条件・1つのseedについて、一回の実験を最初から最後まで実行するファイル
# 1条件だけを確認したいときにも使える　7.5秒間の制御を実行し、総コスト、成功・失敗、計算時間を返すファイル
import time

import numpy as np

from scripts.mppi_pendulum import MPPIControllerForPendulum
from scripts.pendulum import Pendulum

def normalize_angle(
    theta: np.ndarray | float,
) -> np.ndarray | float:
    """角度を [-pi, pi] の範囲に正規化する。"""

    return (
        (theta + np.pi) % (2.0 * np.pi)
    ) - np.pi

def calculate_evaluation_cost(
    state: np.ndarray,
    torque: float,
) -> float:
    """卒論の定義に基づき、1ステップ分の評価コストを計算する。"""

    theta = normalize_angle(state[0])
    theta_dot = state[1]

    # Pendulumの評価用コスト (3.37) のちに制御ループで375回足す
    #最終的な制御性能を比較する評価コスト（状態+入力）
    # theta^2 + 0.1 * theta_dot^2 + 0.1 * torque^2
    cost = (
        theta**2
        + 0.1 * theta_dot**2
        + 0.1 * torque**2
    )

    return float(cost)

def is_pendulum_success(
    state_history: np.ndarray,
    window_size: int = 10,
) -> bool:
    """最後の10ステップすべてが成功条件を満たすか判定する。"""

    # 履歴が10ステップ未満の場合は成功判定できない
    if len(state_history) < window_size:
        return False

    # 最後の10ステップを取り出す
    recent_states = state_history[-window_size:]

    # 0列目：角度、1列目：角速度
    theta = normalize_angle(recent_states[:, 0])
    theta_dot = recent_states[:, 1]

    # 最後の10ステップすべてで角度条件を満たすか確認する　（3.46）
    angle_condition = np.all(
        np.abs(theta) <= 0.1
    )

    # 最後の10ステップすべてで角速度条件を満たすか確認する　（3.46）
    speed_condition = np.all(
        np.abs(theta_dot) <= 0.1
    )

    return bool(
        angle_condition and speed_condition
    )

def run_pendulum_trial(
    K: int,
    N: int,
    prediction_delta_t: float,
    seed: int,
    control_delta_t: float = 0.02,
    simulation_time: float = 7.5,
) -> dict:
    """卒論条件でPendulumの単一試行を実行する。"""

    # 制御周期と実行時間からステップ数を計算する
    simulation_steps = int(
        simulation_time / control_delta_t
    )

    # 描画なしのPendulum環境を作成する
    pendulum = Pendulum(
        mass_of_pole=1.0,
        length_of_pole=1.0,
        max_torque_abs=2.0,
        max_speed_abs=8.0,
        delta_t=control_delta_t,
        visualize=False,
    )

    # 振り子が下向きで静止した状態から開始する
    pendulum.reset(
        init_state=np.array(
            [np.pi, 0.0],
            dtype=float,
        )
    )

    # 指定したK、N、予測周期、seedでMPPIを作成する
    controller = MPPIControllerForPendulum(
        prediction_delta_t=prediction_delta_t,
        mass_of_pole=1.0,
        length_of_pole=1.0,
        max_torque_abs=2.0,
        max_speed_abs=8.0,
        horizon_step_T=N,
        number_of_samples_K=K,
        param_exploration=0.05, #探索noise
        param_lambda=0.5, #コストの差を重みにどれくらい強く反映するかを決める値
        param_alpha=0.8, #探索用のサンプルの割合を決める
        sigma=1.0,
        #（2.11）
        stage_cost_weight=np.array(
            [1.0, 0.1],
            dtype=float,
        ),
        # (2.8)
        terminal_cost_weight=5.0 * np.array(
            [1.0, 0.1],
            dtype=float,
        ),
        seed=seed,
    )

    # 評価値と履歴を保存する変数
    total_cost = 0.0
    state_history = []
    compute_times_ms = []

    for _ in range(simulation_steps):

        # 制御入力を計算する前の状態を取得する
        current_state = pendulum.get_state()

        # MPPIの制御入力計算だけを測定する
        start_time = time.perf_counter()

        #MPPIによる制御入力の計算
        input_torque, _ = controller.calc_control_input(
            observed_x=current_state,
        )
        #各ステップの計算時間を保存
        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000.0

        compute_times_ms.append(elapsed_ms)

        # 実際に適用するトルクを許容範囲内に制限する　（3.2）
        applied_torque = float(
            np.clip(
                input_torque,
                -2.0,
                2.0,
            )
        )

        # 実環境へ入力u_tを適用し、
        # 制御周期Tsだけ状態を進める
        pendulum.update(
            u=np.array(
                [applied_torque],
                dtype=float,
            ),
            delta_t=control_delta_t,
            append_frame=False,
        )

        # 入力適用後の状態x_{t+1}を取得する
        updated_state = pendulum.get_state()

        # 卒論の式（3.37）に合わせ、
        # 更新後状態と、その遷移に使用した入力から
        # 1ステップ分の評価コストを計算する
        total_cost += calculate_evaluation_cost(
            state=updated_state,
            torque=applied_torque,
        )

        # 更新後状態を成功判定用に記録する
        state_history.append(
            updated_state
        )

    # リストをNumPy配列へ変換する
    state_history = np.asarray(
        state_history,
        dtype=float,
    )

    compute_times_ms = np.asarray(
        compute_times_ms,
        dtype=float,
    )

    # 最後の10ステップから成功・失敗を判定する
    success = is_pendulum_success(
        state_history=state_history,
        window_size=10,
    )

    # 1試行分の結果を辞書として返す
    return {
        "environment": "pendulum",
        "K": K,
        "N": N,
        "control_delta_t": control_delta_t,
        "prediction_delta_t": prediction_delta_t,
        "seed": seed,
        "simulation_steps": simulation_steps,
        "total_cost": float(total_cost),
        "success": success,
        "mean_compute_time_ms": float(
            np.mean(compute_times_ms)
        ),
        "max_compute_time_ms": float(
            np.max(compute_times_ms)
        ),
        "final_theta": float(
            state_history[-1, 0]
        ),
        "final_theta_dot": float(
            state_history[-1, 1]
        ),
    }

# run_pendulum_trial.pyを単体で実行した場合のみ単体確認で動作する
if __name__ == "__main__":

    # 同じK、N、予測周期について5つのseedで試行する
    seeds = [0, ]
    trial_results = []

    for seed in seeds:
        result = run_pendulum_trial(
            K=500,
            N=40,
            prediction_delta_t=0.02,
            seed=seed,
        )

        trial_results.append(result)

        print(
            f"seed={seed}, "
            f"total_cost={result['total_cost']:.6f}, "
            f"success={result['success']}, "
            f"mean_time={result['mean_compute_time_ms']:.6f} ms"
        )

    # 各試行の評価値をNumPy配列へ変換する
    total_costs = np.array(
        [
            result["total_cost"]
            for result in trial_results
        ],
        dtype=float,
    )

    successes = np.array(
        [
            result["success"]
            for result in trial_results
        ],
        dtype=float,
    )

    mean_compute_times = np.array(
        [
            result["mean_compute_time_ms"]
            for result in trial_results
        ],
        dtype=float,
    )

    max_compute_times = np.array(
        [
            result["max_compute_time_ms"]
            for result in trial_results
        ],
        dtype=float,
    )

    # 5試行分を集約する
    summary = {
        "environment": "pendulum",
        "K": 500,
        "N": 40,
        "prediction_delta_t": 0.02,
        "number_of_trials": len(seeds),
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

    print()
    print("[SUMMARY] Pendulum 5 trials")
    print(f"K                    : {summary['K']}")
    print(f"N                    : {summary['N']}")
    print(
        "prediction_delta_t   : "
        f"{summary['prediction_delta_t']:.3f} s"
    )
    print(
        "number_of_trials     : "
        f"{summary['number_of_trials']}"
    )
    print(
        "mean_total_cost      : "
        f"{summary['mean_total_cost']:.6f}"
    )
    print(
        "std_total_cost       : "
        f"{summary['std_total_cost']:.6f}"
    )
    print(
        "success_rate         : "
        f"{summary['success_rate']:.2f}"
    )
    print(
        "mean_compute_time_ms : "
        f"{summary['mean_compute_time_ms']:.6f}"
    )
    print(
        "std_compute_time_ms  : "
        f"{summary['std_compute_time_ms']:.6f}"
    )
    print(
        "max_compute_time_ms  : "
        f"{summary['max_compute_time_ms']:.6f}"
    )