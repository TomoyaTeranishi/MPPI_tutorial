# 1つの条件・1つのseedについて、
# CartPoleの実験を最初から最後まで実行する
#
# 5.0秒間の制御を行い、
# 総コスト、成功判定、計算時間、最終状態を返す

import time

import numpy as np

from scripts.cartpole import CartPole
from scripts.mppi_cartpole import MPPIControllerForCartPole


def normalize_angle(
    theta: np.ndarray | float,
) -> np.ndarray | float:
    """角度を[-pi, pi)の範囲に正規化する。"""

    return (
        (theta + np.pi)
        % (2.0 * np.pi)
    ) - np.pi


def calculate_evaluation_cost(
    state: np.ndarray,
) -> float:
    """卒論の式（3.38）に基づき、1ステップ分の評価コストを計算する。"""

    # 状態の順序：
    # [台車位置, ポール角度, 台車速度, ポール角速度]
    cart_position = state[0]
    theta = normalize_angle(state[1])
    cart_velocity = state[2]
    theta_dot = state[3]

    # CartPoleの評価用コスト 式（3.38）
    #
    # 5 * x^2
    # + 10 * theta^2
    # + 0.1 * x_dot^2
    # + 0.1 * theta_dot^2
    #
    # 卒論のCartPole評価コストには、
    # 入力forceの二乗項を含めない
    cost = (
        5.0 * cart_position**2
        + 10.0 * theta**2
        + 0.1 * cart_velocity**2
        + 0.1 * theta_dot**2
    )

    return float(cost)


def is_cartpole_success(
    state_history: np.ndarray,
    window_size: int = 10,
) -> bool:
    """最後の10ステップすべてが式（3.47）を満たすか判定する。"""

    # 履歴が必要なステップ数より短い場合は、
    # 成功判定を行えない
    if len(state_history) < window_size:
        return False

    # 最後の10ステップを取得する
    recent_states = state_history[-window_size:]

    cart_position = recent_states[:, 0]

    theta = normalize_angle(
        recent_states[:, 1]
    )

    cart_velocity = recent_states[:, 2]
    theta_dot = recent_states[:, 3]

    # 式（3.47）
    # 最後の10ステップすべてで、
    # 以下の4条件を満たす必要がある

    position_condition = np.all(
        np.abs(cart_position) <= 1.0
    )

    cart_velocity_condition = np.all(
        np.abs(cart_velocity) <= 0.6
    )

    angle_condition = np.all(
        np.abs(theta) <= 0.1
    )

    angular_velocity_condition = np.all(
        np.abs(theta_dot) <= 0.2
    )

    return bool(
        position_condition
        and cart_velocity_condition
        and angle_condition
        and angular_velocity_condition
    )


def run_cartpole_trial(
    K: int,
    N: int,
    prediction_delta_t: float,
    seed: int,
    control_delta_t: float = 0.02,
    simulation_time: float = 5.0,
) -> dict:
    """卒論条件でCartPoleの単一試行を実行する。"""

    # 5.0秒 / 0.02秒 = 250ステップ
    simulation_steps = int(
        simulation_time
        / control_delta_t
    )

    # 描画なしのCartPole実環境を作成する
    cartpole = CartPole(
        mass_of_cart=2.0,
        mass_of_pole=1.0,
        length_of_pole=1.0,
        max_force_abs=100.0,
        delta_t=control_delta_t,
        visualize=False,
    )

    # 台車中央、ポール下向き、静止状態から開始する
    cartpole.reset(
        init_state=np.array(
            [
                0.0,
                np.pi,
                0.0,
                0.0,
            ],
            dtype=float,
        )
    )

    # 指定されたK、N、予測周期、seedで、
    # CartPole用MPPIコントローラを作成する
    controller = MPPIControllerForCartPole(
        prediction_delta_t=prediction_delta_t,
        mass_of_cart=2.0,
        mass_of_pole=1.0,
        length_of_pole=1.0,
        max_force_abs=100.0,
        horizon_step_T=N,
        number_of_samples_K=K,
        param_exploration=0.0,
        param_lambda=50.0,
        param_alpha=1.0,
        sigma=10.0,
        stage_cost_weight=np.array(
            [
                5.0,
                10.0,
                0.1,
                0.1,
            ],
            dtype=float,
        ),
        terminal_cost_weight=np.array(
            [
                5.0,
                10.0,
                0.1,
                0.1,
            ],
            dtype=float,
        ),
        seed=seed,
    )

    # 1試行分の評価値と履歴
    total_cost = 0.0
    state_history = []
    compute_times_ms = []

    for _ in range(simulation_steps):

        # MPPI計算前の現在状態を取得する
        current_state = cartpole.get_state()

        # calc_control_input()だけの処理時間を測定する
        start_time = time.perf_counter()

        input_force, _ = (
            controller.calc_control_input(
                observed_x=current_state,
            )
        )

        elapsed_ms = (
            time.perf_counter()
            - start_time
        ) * 1000.0

        compute_times_ms.append(
            elapsed_ms
        )

        # 実環境へ適用する力を入力制約内へ収める
        applied_force = float(
            np.clip(
                input_force,
                -100.0,
                100.0,
            )
        )

        # 実環境へ入力u_tを適用し、
        # 制御周期Tsだけ状態を進める
        cartpole.update(
            u=np.array(
                [applied_force],
                dtype=float,
            ),
            delta_t=control_delta_t,
            append_frame=False,
        )

        # 入力適用後の状態x_{t+1}を取得する
        updated_state = cartpole.get_state()

        # 卒論の式（3.38）に合わせ、
        # 更新後状態について1ステップ分の評価コストを計算する
        total_cost += calculate_evaluation_cost(
            state=updated_state,
        )

        # 更新後状態を成功判定用に保存する
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

    # 最後の10ステップで成功判定を行う
    success = is_cartpole_success(
        state_history=state_history,
        window_size=10,
    )

    # 1試行分の結果を返す
    return {
        "environment": "cartpole",
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
        "final_cart_position": float(
            state_history[-1, 0]
        ),
        "final_theta": float(
            state_history[-1, 1]
        ),
        "final_cart_velocity": float(
            state_history[-1, 2]
        ),
        "final_theta_dot": float(
            state_history[-1, 3]
        ),
    }


# このファイルを直接実行した場合のみ、
# 代表条件について5つのseedで単体確認する
if __name__ == "__main__":

    K = 5
    N = 5
    prediction_delta_t = 0.02

    seeds = [
        0,
        
    ]

    trial_results = []

    print("[INFO] Start CartPole trial test")
    print(f"[INFO] K: {K}")
    print(f"[INFO] N: {N}")
    print(
        "[INFO] prediction_delta_t: "
        f"{prediction_delta_t:.3f} s"
    )
    print()

    for seed in seeds:

        result = run_cartpole_trial(
            K=K,
            N=N,
            prediction_delta_t=prediction_delta_t,
            seed=seed,
            control_delta_t=0.02,
            simulation_time=5.0,
        )

        trial_results.append(
            result
        )

        print(
            f"seed={seed}, "
            f"total_cost="
            f"{result['total_cost']:.6f}, "
            f"success="
            f"{result['success']}, "
            f"mean_time="
            f"{result['mean_compute_time_ms']:.6f} ms, "
            f"max_time="
            f"{result['max_compute_time_ms']:.6f} ms"
        )

        print(
            "    final_state: "
            f"x="
            f"{result['final_cart_position']:+.6f} m, "
            f"theta="
            f"{result['final_theta']:+.6f} rad, "
            f"x_dot="
            f"{result['final_cart_velocity']:+.6f} m/s, "
            f"theta_dot="
            f"{result['final_theta_dot']:+.6f} rad/s"
        )

    # 5試行分の結果をNumPy配列へ変換する
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

    print()
    print("[SUMMARY] CartPole 5 trials")

    print(
        f"K                    : {K}"
    )

    print(
        f"N                    : {N}"
    )

    print(
        "prediction_delta_t   : "
        f"{prediction_delta_t:.3f} s"
    )

    print(
        "number_of_trials     : "
        f"{len(seeds)}"
    )

    print(
        "mean_total_cost      : "
        f"{np.mean(total_costs):.6f}"
    )

    print(
        "std_total_cost       : "
        f"{np.std(total_costs):.6f}"
    )

    print(
        "success_rate         : "
        f"{np.mean(successes):.2f}"
    )

    print(
        "mean_compute_time_ms : "
        f"{np.mean(mean_compute_times):.6f}"
    )

    print(
        "std_compute_time_ms  : "
        f"{np.std(mean_compute_times):.6f}"
    )

    print(
        "max_compute_time_ms  : "
        f"{np.max(max_compute_times):.6f}"
    )