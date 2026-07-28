import math
import numpy as np
try:
    # scriptsパッケージとして読み込まれた場合
    from .pendulum import Pendulum
except ImportError:
    from pendulum import Pendulum    
from typing import Tuple

class MPPIControllerForPendulum():
    def __init__(
            self,
            prediction_delta_t: float = 0.05,
            mass_of_pole: float = 1.0,
            length_of_pole: float = 1.0,
            max_torque_abs: float = 2.0,
            max_speed_abs: float = 8.0,
            horizon_step_T: int = 30,
            number_of_samples_K: int = 1000,
            param_exploration: float = 0.01,
            param_lambda: float = 1.0,
            param_alpha: float = 0.1,
            sigma: float = 1.0,
            stage_cost_weight: np.ndarray = np.array([1.0, 0.1]),
            terminal_cost_weight: np.ndarray = np.array([1.0, 0.1]),
            seed: int = 0,
    ) -> None:
        """initialize mppi controller for pendulum"""
        # Pendulum用MPPIコントローラの初期化
        # mppi parameters
        self.dim_u = 1 # dimension of control input vector
        self.T = horizon_step_T # prediction horizon
        self.K = number_of_samples_K # number of sample trajectories
        self.param_exploration = param_exploration  # constant parameter of mppi
        self.param_lambda = param_lambda  # constant parameter of mppi
        self.param_alpha = param_alpha # constant parameter of mppi
        self.param_gamma = self.param_lambda * (1.0 - (self.param_alpha))  # constant parameter of mppi
        self.Sigma = sigma # 探索ノイズの標準偏差
        self.rng = np.random.default_rng(seed=seed) # 同じseedを指定すると、同じノイズ列を再現できる
        self.stage_cost_weight = stage_cost_weight
        self.terminal_cost_weight = terminal_cost_weight

        # pendulum parameters
        self.g = 9.81
        self.prediction_delta_t = prediction_delta_t
        self.mass_of_pole = mass_of_pole
        self.length_of_pole = length_of_pole
        self.max_torque = max_torque_abs
        self.max_speed = max_speed_abs

        # mppi variables
        self.u_prev = np.zeros((self.T))

    def calc_control_input(self, observed_x: np.ndarray) -> Tuple[float, np.ndarray]:
        """calculate optimal control input with vectorized K samples"""

        # load previous control input sequence
        u = self.u_prev.copy()

        # set initial x value from observation
        x0 = np.asarray(observed_x, dtype=float).copy()

        # sample noise: shape is (K, T)
        epsilon = self._calc_epsilon(self.Sigma, self.K, self.T)

        # prepare cost buffer: shape is (K,)
        S = np.zeros(self.K)

        # prepare noisy control input sequence: shape is (K, T)
        v = np.zeros((self.K, self.T))

        num_exploitation = int((1.0 - self.param_exploration) * self.K)

        # exploitation: v = u + epsilon
        if num_exploitation > 0:
            v[:num_exploitation, :] = (
                u[None, :] + epsilon[:num_exploitation, :]
            )

        # exploration: v = epsilon
        if num_exploitation < self.K:
            v[num_exploitation:, :] = epsilon[num_exploitation:, :]

        # K本分の状態をまとめて持つ: shape is (K, 2)
        x = np.tile(x0, (self.K, 1)).astype(float)

        # rollout all K samples at once
        for t in range(self.T):

            # raw sampled input: shape is (K,)
            v_t_raw = v[:, t]

            # clamp input: shape is (K,)
            v_t = self._g_batch(v_t_raw)

            # update K states at once
            x = self._F_batch(x, v_t)

            # add stage cost for all K samples
            S += self._c_batch(x)

            # add MPPI control cost term
            S += self.param_gamma * u[t] * (1.0 / self.Sigma) * v_t_raw

        # add terminal cost
        S += self._phi_batch(x)

        # compute information theoretic weights
        w = self._compute_weights(S)

        # calculate weighted noise: shape is (T,)
        w_epsilon = np.sum(w[:, None] * epsilon, axis=0)

        # apply moving average filter for smoothing input sequence
        w_epsilon = self._moving_average_filter(xx=w_epsilon, window_size=5)

        # update control input sequence
        u = u + w_epsilon

        # clamp updated input sequence
        u = np.clip(u, -self.max_torque, self.max_torque)

        # update previous control input sequence
        self.u_prev[:-1] = u[1:]
        self.u_prev[-1] = u[-1]

        # return optimal control input and input sequence
        return u[0], u

    def _calc_epsilon(self, sigma: float, size_sample: int, size_time_step: int) -> np.ndarray:
        """sample epsilon"""
        epsilon = self.rng.normal(0.0, sigma, (self.K, self.T)) # size is self.K x self.T
        return epsilon

    def _g(self, v: np.ndarray) -> float:
        """clamp input"""
        v = np.clip(v, -self.max_torque, self.max_torque)
        return v

    def _c(self, x_t: np.ndarray) -> float:
        """calculate stage cost"""
        # parse x_t
        theta, theta_dot = x_t[0], x_t[1]
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi # normalize theta to [-pi, pi]

        # calculate stage cost
        stage_cost = self.stage_cost_weight[0]*theta**2 + self.stage_cost_weight[1]*theta_dot**2
        return stage_cost

    def _phi(self, x_T: np.ndarray) -> float:
        """calculate terminal cost"""
        # parse x_T
        theta, theta_dot = x_T[0], x_T[1]
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi # normalize theta to [-pi, pi]

        # calculate terminal cost
        terminal_cost = self.terminal_cost_weight[0]*theta**2 + self.terminal_cost_weight[1]*theta_dot**2
        return terminal_cost

    def _F(self, x_t: np.ndarray, v_t: np.ndarray) -> np.ndarray:
        # k本のサンプル軌道について、振り子の次の状態を計算する
        """calculate next state of the pendulum"""
        # get previous state variables
        theta, theta_dot = x_t[:, 0], x_t[:, 1]

        # 振り子の物理パラメータ
        g = self.g
        m = self.mass_of_pole
        l = self.length_of_pole
        # MPPI内部モデルの予測周期 Δt        
        dt = self.prediction_delta_t

        # k個のトルクを、[-max_torque, max_torque]の範囲に制限する
        torque = np.clip(v_t, -self.max_torque, self.max_torque,)
        #　角速度を更新する
        new_theta_dot = theta_dot + (3 * g / (2 * l) * np.sin(theta) + 3.0 / (m * l**2) * torque) * dt
        #　角速度を[-max_speed, max_speed]の範囲に制限する
        new_theta_dot = np.clip(new_theta_dot, -self.max_speed, self.max_speed)
        #　角度を更新する
        new_theta = theta + new_theta_dot * dt
        # 角度を[-pi, pi]の範囲に正規化
        new_theta = ((new_theta + np.pi) % (2 * np.pi)) - np.pi

        # thetaとtheta_dotを列方向に並べて、(K, 2)の形状の配列として返す
        return np.column_stack(
            (new_theta, new_theta_dot)
        )

    def _g_batch(self, v: np.ndarray) -> np.ndarray:
        """clamp input for all samples

        v: shape is (K,)
        """
        return np.clip(v, -self.max_torque, self.max_torque)

    def _F_batch(self, x_t: np.ndarray, v_t: np.ndarray) -> np.ndarray:
        """calculate next states of the pendulum for all samples

        x_t: shape is (K, 2)
        v_t: shape is (K,)
        """

        x_t = np.asarray(x_t, dtype=float)
        v_t = np.asarray(v_t, dtype=float)

        theta = x_t[:, 0]
        theta_dot = x_t[:, 1]

        g = self.g
        m = self.mass_of_pole
        l = self.length_of_pole
        dt = self.prediction_delta_t

        torque = v_t

        new_theta_dot = theta_dot + (
            3.0 * g / (2.0 * l) * np.sin(theta)
            + 3.0 / (m * l**2) * torque
        ) * dt

        new_theta_dot = np.clip(
            new_theta_dot,
            -self.max_speed,
            self.max_speed
        )

        new_theta = theta + new_theta_dot * dt

        x_next = np.empty((x_t.shape[0], 2), dtype=float)
        x_next[:, 0] = new_theta
        x_next[:, 1] = new_theta_dot

        return x_next

    def _c_batch(self, x_t: np.ndarray) -> np.ndarray:
        """calculate stage cost for all samples

        x_t: shape is (K, 2)
        """

        theta = x_t[:, 0]
        theta_dot = x_t[:, 1]

        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi

        stage_cost = (
            self.stage_cost_weight[0] * theta**2
            + self.stage_cost_weight[1] * theta_dot**2
        )

        return np.asarray(stage_cost, dtype=float)

    def _phi_batch(self, x_T: np.ndarray) -> np.ndarray:
        """calculate terminal cost for all samples

        x_T: shape is (K, 2)
        """

        theta = x_T[:, 0]
        theta_dot = x_T[:, 1]

        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi

        terminal_cost = (
            self.terminal_cost_weight[0] * theta**2
            + self.terminal_cost_weight[1] * theta_dot**2
        )

        return np.asarray(terminal_cost, dtype=float)

    def _compute_weights(self, S: np.ndarray) -> np.ndarray:
        """compute weights for each sample"""

        rho = np.min(S)

        w = np.exp((-1.0 / self.param_lambda) * (S - rho))

        w = w / np.sum(w)

        return w

    def _moving_average_filter(self, xx: np.ndarray, window_size: int) -> np.ndarray:
        """apply moving average filter for smoothing input sequence
        Ref. https://zenn.dev/bluepost/articles/1b7b580ab54e95
        Note: The original MPPI paper uses the Savitzky-Golay Filter for smoothing control inputs.
        """
        b = np.ones(window_size)/window_size
        xx_mean = np.convolve(xx, b, mode="same")
        n_conv = math.ceil(window_size/2)
        xx_mean[0] *= window_size/n_conv
        for i in range(1, n_conv):
            xx_mean[i] *= window_size/(i+n_conv)
            xx_mean[-i] *= window_size/(i + n_conv - (window_size % 2)) 
        return xx_mean


def run_simulation_mppi_pendulum() -> None:
    """run simulation of swinging up pendulum with MPPI controller"""
    # MPPIを用いてPendulumの振り上げシミュレーションを実行する
    print("[INFO] Start simulation of swinging up a pendulum with MPPI controller")

    # simulation settings
    # 実環境へ制御入力を適用する周期 Ts
    control_delta_t = 0.05 # [sec]

    # MPPI内部モデルの予測周期 Δt
    prediction_delta_t = 0.05 # [sec]

    # シミュレーション全体の実行時間
    simulation_time = 7.5

    # 制御周期から実行ステップ数を計算する
    simulation_steps = int(
        simulation_time / control_delta_t
    )

    print(
        f"[INFO] control_delta_t : "
        f"{control_delta_t:.2f}[s]"
    )
    print(
        f"[INFO] prediction_delta_t : "
        f"{prediction_delta_t:.2f}[s]"
    )
    print(
        f"[INFO] simulation_steps : "
        f"{simulation_steps}[steps]"
    )
    print(
        f"[INFO] total_simulation_time : "
        f"{control_delta_t * simulation_steps:.2f}[s]"
    )

    # initialize a pendulum as a control target
    pendulum = Pendulum(
        mass_of_pole = 1.0,
        length_of_pole = 1.0,
        max_torque_abs = 2.0,
        max_speed_abs = 8.0,
        # 実環境は制御周期Tsで状態を更新する
        delta_t = control_delta_t,
        visualize = True,
    )
    pendulum.reset(
        init_state = np.array([np.pi, 0.0]), # [theta(rad), theta_dot(rad/s)]
    )

    # initialize a mppi controller for the pendulum
    mppi = MPPIControllerForPendulum(
        # MPPI内部モデルの予測周期Δt
        prediction_delta_t = prediction_delta_t,
        mass_of_pole = 1.0,
        length_of_pole = 1.0,
        max_torque_abs = 2.0,
        max_speed_abs = 8.0,
        horizon_step_T = 20,
        number_of_samples_K = 2000,
        param_exploration = 0.05,
        param_lambda = 0.5,
        param_alpha = 0.8,
        sigma = 1.0,
        stage_cost_weight    = np.array([1.0, 0.1]), # weight for [theta, theta_dot]
        terminal_cost_weight = 5.0 * np.array([1.0, 0.1]), # weight for [theta, theta_dot]
        seed = 1,
    )

    # simulation loop
    # シミュレーションループ
    for step in range(simulation_steps):

        # 現在の振り子状態を取得する
        # current_state[0]：角度
        # current_state[1]：角速度
        current_state = pendulum.get_state()

        # MPPIで制御入力を計算する
        input_torque, input_torque_sequence = mppi.calc_control_input(
            observed_x=current_state,
        )

        # 現在時刻は実環境の制御周期から計算する
        current_time = step * control_delta_t

        # 現在の状態と制御入力を表示する
        print(
            f"Time: {current_time:>2.2f}[s], "
            f"theta={current_state[0]:>+3.3f}[rad], "
            f"theta_dot={current_state[1]:>+3.3f}[rad/s], "
            f"input torque={input_torque:>+3.2f}[Nm]"
        )

        # 実環境を制御周期Tsだけ進める
        pendulum.update(
            u=np.array([input_torque]),
            delta_t=control_delta_t,
        )

    # save animation
    pendulum.save_animation("mppi_pendulum.mp4", interval=int(control_delta_t * 1000), movie_writer="ffmpeg") # ffmpeg is required to write mp4 file


if __name__ == "__main__":
    run_simulation_mppi_pendulum()
