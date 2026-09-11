#　環境モデルとMPPI内部モデルの1ステップ結果が一致するか確認するテストコード
import numpy as np

from scripts.pendulum import Pendulum
from scripts.mppi_pendulum import MPPIControllerForPendulum


def test_one_step_model_match() -> None:
    """環境側とMPPI内部モデルの1ステップ結果が一致するか確認する。"""

    # 両方のモデルで同じ条件を使用する
    dt = 0.05
    initial_state = np.array([2.5, 1.0])
    torque = 1.5

    # --------------------------------------------------
    # 実際のシミュレーション環境で次状態を計算
    # --------------------------------------------------
    pendulum = Pendulum(
        mass_of_pole=1.0,
        length_of_pole=1.0,
        max_torque_abs=2.0,
        max_speed_abs=8.0,
        delta_t=dt,
        visualize=False,
    )

    pendulum.reset(
        init_state=initial_state.copy(),
    )

    pendulum.update(
        u=np.array([torque]),
        delta_t=dt,
        append_frame=False,
    )

    environment_next_state = pendulum.get_state()

    # --------------------------------------------------
    # MPPI内部モデルで同じ次状態を計算
    # --------------------------------------------------
    controller = MPPIControllerForPendulum(
        prediction_delta_t = dt,
        mass_of_pole=1.0,
        length_of_pole=1.0,
        max_torque_abs=2.0,
        max_speed_abs=8.0,
        horizon_step_T=10,
        number_of_samples_K=1,
    )

    # _F()は形状(K, 2)の状態配列を受け取る
    # 今回はK=1なので、形状を(1, 2)にする
    batch_state = initial_state[None, :]

    # K=1に対応する要素数1の入力配列
    batch_torque = np.array([torque])

    controller_next_state = controller._F(
        x_t=batch_state,
        v_t=batch_torque,
    )[0]

    # --------------------------------------------------
    # 両方の計算結果を比較
    # --------------------------------------------------
    difference = (
        environment_next_state
        - controller_next_state
    )

    print("Environment:", environment_next_state)
    print("Controller :", controller_next_state)
    print("Difference :", difference)

    # 誤差がほぼゼロなら、同じモデルとして判定する
    assert np.allclose(
        environment_next_state,
        controller_next_state,
        rtol=1.0e-12,
        atol=1.0e-12,
    ), "Environment and internal model do not match."

    print("[PASS] Environment and internal model match.")


if __name__ == "__main__":
  test_one_step_model_match()