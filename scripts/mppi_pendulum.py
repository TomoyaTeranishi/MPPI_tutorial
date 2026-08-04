import math
import numpy as np
try:
    # scriptsパッケージとして読み込まれた場合
    from .pendulum import Pendulum
except ImportError:
    from pendulum import Pendulum    
from typing import Tuple

class MPPIControllerForPendulum():
    # MPPIコントローラのパラメータと振り子の物理パラメータを保存する
    def __init__(
            self,
            prediction_delta_t: float = 0.05,
            mass_of_pole: float = 1.0,
            length_of_pole: float = 1.0,
            max_torque_abs: float = 2.0,
            max_speed_abs: float = 8.0,
            horizon_step_T: int = 30,
            number_of_samples_K: int = 1000,
            param_exploration: float = 0.01, # 純粋な探索軌道の割合
            param_lambda: float = 1.0, # 逆温度 λ
            param_alpha: float = 0.1, # 制御入力コストに関する係数
            sigma: float = 1.0, #探索ノイズの標準偏差
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

    # MPPIの最適制御入力を計算する
    def calc_control_input(self, observed_x: np.ndarray) -> Tuple[float, np.ndarray]:
        #　卒論のAlgorithm1でいう、状態取得からSendToAcutuators(u_0)までの処理を行う
        """calculate optimal control input with vectorized K samples"""

        # load previous control input sequence
        u = self.u_prev.copy() #.copy()によって、計算途中にu_prevが書き換わることを防ぐ

        # set initial x value from observation
        x0 = np.asarray(observed_x, dtype=float).copy() #現在の状態を予測の初期状態にする　式(2.13) Algorithm(l.2)
                                                        #obseved_xは実環境から取得した現在の状態
        # sample noise: shape is (K, T)
        epsilon = self._calc_epsilon(self.Sigma, self.K, self.T)    #探索のためのノイズを生成する（ex.ε(1,0)=ε(K,N)）
        
        # prepare cost buffer: shape is (K,)
        S = np.zeros(self.K) # サンプル軌道コストを保存する配列を用意する（ex.S(1)=S(K)）

        # prepare noisy control input sequence: shape is (K, T)
        v = np.zeros((self.K, self.T)) #k本の候補入力系列を保存

        num_exploitation = int((1.0 - self.param_exploration) * self.K) #利用サンプル数の計算uに依存せず広い範囲を探索する軌道の数など Algorithm(l.4)

        # exploitation: v = u + epsilon　現在の入力系列周辺を探索する軌道の数 Algorithm(l.4-5)
        if num_exploitation > 0:
            v[:num_exploitation, :] = (
                u[None, :] + epsilon[:num_exploitation, :]
            )

        # exploration: v = epsilon　基準入力に依存せず広い範囲を探索する軌道の数 Algorithm(l.6-7)
        if num_exploitation < self.K:
            v[num_exploitation:, :] = epsilon[num_exploitation:, :]

        # K本分の状態をまとめて持つ: shape is (K, 2) Algorithm(l.3)
        x = np.tile(x0, (self.K, 1)).astype(float)

        # rollout all K samples at once
        for t in range(self.T):

            # raw sampled input: shape is (K,)
            v_t_raw = v[:, t]

            # clamp input: shape is (K,)
            v_t = self._g_batch(v_t_raw)

            # update K states at once
            x = self._F_batch(x, v_t)

            # add ステージコスト for all K samples (2.11)
            S += self._c_batch(x)

            # add MPPI control cost term　Algorithm(l.8) ガンマはutとvtの関係をどの程度コストへ反映するかを決める係数
            S += self.param_gamma * u[t] * (1.0 / self.Sigma) * v_t_raw

        # add terminal cost(2.8)
        S += self._phi_batch(x)

        # compute information theoretic weights 重み
        w = self._compute_weights(S)

        # calculate weighted noise: shape is (T,)　重み付きノイズ
        w_epsilon = np.sum(w[:, None] * epsilon, axis=0)

        # apply moving average filter for smoothing input sequence　重み付きノイズを平滑化　Algorithm(l.11)
        w_epsilon = self._moving_average_filter(xx=w_epsilon, window_size=5)

        # update control input sequence Algorithm(l.11)
        u = u + w_epsilon

        # clamp updated input sequence
        u = np.clip(u, -self.max_torque, self.max_torque)

        # update previous control input sequence　入力系列を1ステップ左へシフト
        self.u_prev[:-1] = u[1:]
        self.u_prev[-1] = u[-1]

        # return optimal control input and input sequence　最初の入力だけを返す
        return u[0], u

    # 探索ノイズを生成　（2.7）Algorithm(l.3)
    def _calc_epsilon(self, sigma: float, size_sample: int, size_time_step: int) -> np.ndarray:
        """sample epsilon"""
        epsilon = self.rng.normal(0.0, sigma, (self.K, self.T)) # size is self.K x self.T
        return epsilon

    # 入力トルクを制限
    def _g(self, v: np.ndarray) -> float:
        """clamp input"""
        v = np.clip(v, -self.max_torque, self.max_torque)
        return v

    # ステージコストを計算
    def _c(self, x_t: np.ndarray) -> float:
        """calculate stage cost"""
        # parse x_t
        theta, theta_dot = x_t[0], x_t[1]
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi # normalize theta to [-pi, pi]

        # calculate stage cost
        stage_cost = self.stage_cost_weight[0]*theta**2 + self.stage_cost_weight[1]*theta_dot**2
        return stage_cost

    # 終端コストを計算
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

    # 入力トルクを制限 (3.2)
    def _g_batch(self, v: np.ndarray) -> np.ndarray:
        """clamp input for all samples

        v: shape is (K,)
        """
        return np.clip(v, -self.max_torque, self.max_torque)

    def _F_batch(self, x_t: np.ndarray, v_t: np.ndarray) -> np.ndarray:
        """calculate next states of the pendulum for all samples
        # 状態：k本のサンプル軌道について、振り子の次の状態を計算する (2.6)
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

        #(3.6)しかし卒論の運動方程式とは不一致 坂本コードとは一緒
        new_theta_dot = theta_dot + (
            3.0 * g / (2.0 * l) * np.sin(theta)
            + 3.0 / (m * l**2) * torque
        ) * dt
        # 角速度制限　表3.6
        new_theta_dot = np.clip(
            new_theta_dot,
            -self.max_speed,
            self.max_speed
        )

        new_theta = theta + new_theta_dot * dt #(3.5)　卒論と不一致　坂本コードとは一致

        new_theta = (
            (new_theta + np.pi) % (2.0 * np.pi)
        ) -np.pi

        #(2.6) k本の次の状態を作成して返す
        x_next = np.empty((x_t.shape[0], 2), dtype=float)
        x_next[:, 0] = new_theta
        x_next[:, 1] = new_theta_dot

        return x_next

    # ステージコスト
    def _c_batch(self, x_t: np.ndarray) -> np.ndarray:
        """calculate stage cost for all samples

        x_t: shape is (K, 2)
        """

        theta = x_t[:, 0] #(3.1)
        theta_dot = x_t[:, 1]

        #　正規化
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        #状態に依存するコスト　Algorithm(l.8) (2.11)のc(xt)
        stage_cost = (
            self.stage_cost_weight[0] * theta**2
            + self.stage_cost_weight[1] * theta_dot**2
        )

        return np.asarray(stage_cost, dtype=float)

    # 終端コスト（状態）
    def _phi_batch(self, x_T: np.ndarray) -> np.ndarray:
        """calculate terminal cost for all samples

        x_T: shape is (K, 2)
        """

        theta = x_T[:, 0]
        theta_dot = x_T[:, 1]

        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        # (2.8)のΦ(xN) Algorithm1(l.9) self.terminal_cost_weightに関してはl.381で定義済み
        terminal_cost = (
            self.terminal_cost_weight[0] * theta**2
            + self.terminal_cost_weight[1] * theta_dot**2
        )

        return np.asarray(terminal_cost, dtype=float)

    #　軌道ごとの重みを計算する Algorithm2
    def _compute_weights(self, S: np.ndarray) -> np.ndarray:
        """compute weights for each sample"""

        rho = np.min(S)

        w = np.exp((-1.0 / self.param_lambda) * (S - rho)) #未正規化重み

        w = w / np.sum(w) #正規化　

        return w

    # 入力平滑化 Algorithm1(l.11)このコードは移動平均　坂本コードも移動平均
    def _moving_average_filter(self, xx: np.ndarray, window_size: int) -> np.ndarray:
        """apply moving average filter for smoothing input sequence
        Ref. https://zenn.dev/bluepost/articles/1b7b580ab54e95
        Note: The original MPPI paper uses the Savitzky-Golay Filter for smoothing control inputs.
        """
        b = np.ones(window_size)/window_size #平滑化係数
        xx_mean = np.convolve(xx, b, mode="same") #畳み込み
        n_conv = math.ceil(window_size/2) #端点補正
        xx_mean[0] *= window_size/n_conv
        for i in range(1, n_conv):
            xx_mean[i] *= window_size/(i+n_conv)
            xx_mean[-i] *= window_size/(i + n_conv - (window_size % 2)) 
        return xx_mean

# 単発デモ　本実験とは結果は変わらない？
def run_simulation_mppi_pendulum() -> None:
    """run simulation of swinging up pendulum with MPPI controller"""
    # Pendulum環境とMPPIControllerForPendulumを接続し、MPPIによる振り上げ制御を1回実行して、最後に動画を保存するデモ関数
    print("[INFO] Start simulation of swinging up a pendulum with MPPI controller")

    # simulation settings
    # 実環境へ制御入力を適用する周期 Ts
    control_delta_t = 0.02 # [sec]

    # MPPI内部モデルの予測周期 Δt
    prediction_delta_t = 0.05 # [sec]

    # シミュレーション全体の実行時間
    simulation_time = 7.5

    # 制御周期から実行ステップ数を計算する ステップ数375回
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
    # mppiが制御する実環境側の振り子を作成する　表3.3
    pendulum = Pendulum(
        mass_of_pole = 1.0,
        length_of_pole = 1.0,
        max_torque_abs = 2.0,
        max_speed_abs = 8.0,
        # 実環境は制御周期Tsで状態を更新する
        delta_t = control_delta_t,
        visualize = False,
    )
    # 初期状態の設定　（3.1）
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
        param_exploration = 0.05, #探索率
        param_lambda = 0.5, #Algorithm2 重み計算　逆温度
        param_alpha = 0.8, #Algorithm1（l.8）制御コストの計算
        sigma = 1.0, #(2.7),Algorithm1(l.3) 標準偏差
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

        # MPPIで制御入力を計算する（コントローラーがトルクを決める）
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

        # 実環境を制御周期Tsだけ進める（決められたトルクで振り子を動かす）
        pendulum.update(
            u=np.array([input_torque]),
            delta_t=control_delta_t,
        )

    # save animation
    pendulum.save_animation("mppi_pendulum.mp4", interval=int(control_delta_t * 1000), movie_writer="ffmpeg") # ffmpeg is required to write mp4 file

# ファイル末尾の単体確認　Sweepから読み込まれた場合は実行されない
if __name__ == "__main__":
    run_simulation_mppi_pendulum()
