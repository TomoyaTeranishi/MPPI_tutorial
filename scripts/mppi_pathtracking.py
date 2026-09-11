import math
import numpy as np
from typing import Tuple
from pathtracking_kbm import Vehicle


class MPPIControllerForPathTracking():
    def __init__(
            self,
            delta_t: float = 0.05,
            wheel_base: float = 2.5, # [m]
            max_steer_abs: float = 0.523, # [rad]
            max_accel_abs: float = 2.000, # [m/s^2]
            ref_path: np.ndarray = np.array([[0.0, 0.0, 0.0, 1.0], [10.0, 0.0, 0.0, 1.0]]),
            horizon_step_T: int = 30,
            number_of_samples_K: int = 1000,
            param_exploration: float = 0.0,
            param_lambda: float = 50.0,
            param_alpha: float = 1.0,
            sigma: np.ndarray = np.array([[0.5, 0.0], [0.0, 0.1]]), 
            stage_cost_weight: np.ndarray = np.array([50.0, 50.0, 1.0, 20.0]), # weight for [x, y, yaw, v]
            terminal_cost_weight: np.ndarray = np.array([50.0, 50.0, 1.0, 20.0]), # weight for [x, y, yaw, v]
            visualize_optimal_traj = True,  # if True, optimal trajectory is visualized
            visualze_sampled_trajs = False, # if True, sampled trajectories are visualized
    ) -> None:
        """initialize mppi controller for path-tracking"""
        # mppi parameters
        self.dim_x = 4 # dimension of system state vector
        self.dim_u = 2 # dimension of control input vector
        self.T = horizon_step_T # prediction horizon
        self.K = number_of_samples_K # number of sample trajectories
        self.param_exploration = param_exploration  # constant parameter of mppi
        self.param_lambda = param_lambda  # constant parameter of mppi
        self.param_alpha = param_alpha # constant parameter of mppi
        self.param_gamma = self.param_lambda * (1.0 - (self.param_alpha))  # constant parameter of mppi
        self.Sigma = sigma # deviation of noise
        self.stage_cost_weight = stage_cost_weight
        self.terminal_cost_weight = terminal_cost_weight
        self.visualize_optimal_traj = visualize_optimal_traj
        self.visualze_sampled_trajs = visualze_sampled_trajs

        # vehicle parameters
        self.delta_t = delta_t #[s]
        self.wheel_base = wheel_base#[m]
        self.max_steer_abs = max_steer_abs # [rad]
        self.max_accel_abs = max_accel_abs # [m/s^2]
        self.ref_path = ref_path

        # mppi variables
        self.u_prev = np.zeros((self.T, self.dim_u))

        # ref_path info
        self.prev_waypoints_idx = 0

    def calc_control_input(self, observed_x: np.ndarray) -> Tuple[float, np.ndarray]:
        """calculate optimal control input with vectorized K samples"""
        # load previous control input sequence
        u = self.u_prev.copy()

        # set initial x value from observation
        x0 = np.asarray(observed_x, dtype=float).copy()

        # get the waypoint closest to current vehicle position
        self._get_nearest_waypoint(x0[0], x0[1], update_prev_idx=True)
        if self.prev_waypoints_idx >= self.ref_path.shape[0] - 1:
            print("[INFO] Reached the end of the reference path.")
            raise IndexError

        # sample noise: shape is (K, T, dim_u)
        epsilon = self._calc_epsilon(self.Sigma, self.K, self.T, self.dim_u)

        # sampled control input sequence: shape is (K, T, dim_u)
        v = np.zeros((self.K, self.T, self.dim_u))

        num_exploitation = int((1.0 - self.param_exploration) * self.K)

        # exploitation: v = u + epsilon
        if num_exploitation > 0:
            v[:num_exploitation, :, :] = (
                u[None, :, :] + epsilon[:num_exploitation, :, :]
            )

        # exploration: v = epsilon
        if num_exploitation < self.K:
            v[num_exploitation:, :, :] = epsilon[num_exploitation:, :, :]

        # cost buffer: shape is (K,)
        S = np.zeros(self.K, dtype=float)

        # inverse of Sigma
        Sigma_inv = np.linalg.inv(self.Sigma)

        # K本分の状態をまとめて持つ: shape is (K, dim_x)
        x = np.tile(x0, (self.K, 1)).astype(float)

        # rollout all K samples at once
        for t in range(self.T):

            # raw sampled inputs: shape is (K, dim_u)
            v_t_raw = v[:, t, :]

            # clamp inputs: shape is (K, dim_u)
            v_t = self._g_batch(v_t_raw)

            # update K states at once
            x = self._F_batch(x, v_t)

            # add stage cost for all K samples
            S += self._c_batch(x)

            # add MPPI control cost term
            S += self.param_gamma * np.einsum(
                "i,ij,kj->k",
                u[t],
                Sigma_inv,
                v_t_raw
            )

        # add terminal cost
        S += self._phi_batch(x)

        # compute information theoretic weights
        w = self._compute_weights(S)

        # calculate weighted noise: shape is (T, dim_u)
        w_epsilon = np.sum(w[:, None, None] * epsilon, axis=0)

        # apply moving average filter for smoothing input sequence
        w_epsilon = self._moving_average_filter(xx=w_epsilon, window_size=10)

        # update control input sequence
        u = u + w_epsilon

        # clamp updated control input sequence
        u[:, 0] = np.clip(u[:, 0], -self.max_steer_abs, self.max_steer_abs)
        u[:, 1] = np.clip(u[:, 1], -self.max_accel_abs, self.max_accel_abs)

        # calculate optimal trajectory
        optimal_traj = np.zeros((self.T, self.dim_x))
        if self.visualize_optimal_traj:
            x_opt = x0.copy()
            for t in range(self.T):
                x_opt = self._F(x_opt, self._g(u[t].copy()))
                optimal_traj[t] = x_opt

        # calculate sampled trajectories
        sampled_traj_list = np.zeros((self.K, self.T, self.dim_x))
        if self.visualze_sampled_trajs:
            x_samples = np.tile(x0, (self.K, 1)).astype(float)
            for t in range(self.T):
                v_t = self._g_batch(v[:, t, :])
                x_samples = self._F_batch(x_samples, v_t)
                sampled_traj_list[:, t, :] = x_samples

        # update previous control input sequence
        self.u_prev[:-1] = u[1:]
        self.u_prev[-1] = u[-1]

        return u[0], u, optimal_traj, sampled_traj_list

    def _calc_epsilon(self, sigma: np.ndarray, size_sample: int, size_time_step: int, size_dim_u: int) -> np.ndarray:
        """sample epsilon"""
        # check if sigma row size == sigma col size == size_dim_u and size_dim_u > 0
        if sigma.shape[0] != sigma.shape[1] or sigma.shape[0] != size_dim_u or size_dim_u < 1:
            print("[ERROR] sigma must be a square matrix with the size of size_dim_u.")
            raise ValueError

        # sample epsilon
        mu = np.zeros((size_dim_u)) # set average as a zero vector
        epsilon = np.random.multivariate_normal(mu, sigma, (size_sample, size_time_step))
        return epsilon

    def _g(self, v: np.ndarray) -> float:
        """clamp input"""
        # limit control inputs
        v[0] = np.clip(v[0], -self.max_steer_abs, self.max_steer_abs) # limit steering input
        v[1] = np.clip(v[1], -self.max_accel_abs, self.max_accel_abs) # limit acceleraiton input
        return v
    
    
    def _g_batch(self, v: np.ndarray) -> np.ndarray:
        """clamp inputs for all samples

        v: shape is (K, dim_u)
        """

        v_clipped = np.asarray(v, dtype=float).copy()

        # limit steering input
        v_clipped[:, 0] = np.clip(
            v_clipped[:, 0],
            -self.max_steer_abs,
            self.max_steer_abs
        )

        # limit acceleration input
        v_clipped[:, 1] = np.clip(
            v_clipped[:, 1],
            -self.max_accel_abs,
            self.max_accel_abs
        )

        return v_clipped


    def _c(self, x_t: np.ndarray) -> float:
        """calculate stage cost"""
        # parse x_t
        x, y, yaw, v = x_t
        yaw = ((yaw + 2.0*np.pi) % (2.0*np.pi)) # normalize theta to [0, 2*pi]

        # calculate stage cost
        _, ref_x, ref_y, ref_yaw, ref_v = self._get_nearest_waypoint(x, y)
        yaw_diff = np.arctan2(np.sin(yaw-ref_yaw), np.cos(yaw-ref_yaw)) # normalize yaw error to [-pi, pi]
        stage_cost = self.stage_cost_weight[0]*(x-ref_x)**2 + self.stage_cost_weight[1]*(y-ref_y)**2 + \
                     self.stage_cost_weight[2]*(yaw_diff)**2 + self.stage_cost_weight[3]*(v-ref_v)**2
        return stage_cost
    
    def _c_batch(self, x_t: np.ndarray) -> np.ndarray:
        """calculate stage cost for all samples

        x_t: shape is (K, dim_x)
        """
        x_t = np.asarray(x_t, dtype=float)

        x = x_t[:, 0]
        y = x_t[:, 1]
        yaw = x_t[:, 2]
        vel = x_t[:, 3]

        yaw = ((yaw + 2.0 * np.pi) % (2.0 * np.pi))

        _, ref_x, ref_y, ref_yaw, ref_v = self._get_nearest_waypoint_batch(x, y)

        yaw_diff = np.arctan2(
            np.sin(yaw - ref_yaw),
            np.cos(yaw - ref_yaw)
        )

        stage_cost = (
            self.stage_cost_weight[0] * (x - ref_x)**2
            + self.stage_cost_weight[1] * (y - ref_y)**2
            + self.stage_cost_weight[2] * yaw_diff**2
            + self.stage_cost_weight[3] * (vel - ref_v)**2
        )

        return np.asarray(stage_cost, dtype=float)

    def _phi(self, x_T: np.ndarray) -> float:
        """calculate terminal cost"""
        # parse x_T
        x, y, yaw, v = x_T
        yaw = ((yaw + 2.0*np.pi) % (2.0*np.pi)) # normalize theta to [0, 2*pi]

        # calculate terminal cost
        _, ref_x, ref_y, ref_yaw, ref_v = self._get_nearest_waypoint(x, y)
        yaw_diff = np.arctan2(np.sin(yaw-ref_yaw), np.cos(yaw-ref_yaw)) # normalize yaw error to [-pi, pi]
        terminal_cost = self.terminal_cost_weight[0]*(x-ref_x)**2 + self.terminal_cost_weight[1]*(y-ref_y)**2 + \
                        self.terminal_cost_weight[2]*(yaw_diff)**2 + self.terminal_cost_weight[3]*(v-ref_v)**2
        return terminal_cost
    
    def _phi_batch(self, x_T: np.ndarray) -> np.ndarray:
        """calculate terminal cost for all samples

        x_T: shape is (K, dim_x)
        """
        x_T = np.asarray(x_T, dtype=float)

        x = x_T[:, 0]
        y = x_T[:, 1]
        yaw = x_T[:, 2]
        vel = x_T[:, 3]

        yaw = ((yaw + 2.0 * np.pi) % (2.0 * np.pi))

        _, ref_x, ref_y, ref_yaw, ref_v = self._get_nearest_waypoint_batch(x, y)

        yaw_diff = np.arctan2(
            np.sin(yaw - ref_yaw),
            np.cos(yaw - ref_yaw)
        )

        terminal_cost = (
            self.terminal_cost_weight[0] * (x - ref_x)**2
            + self.terminal_cost_weight[1] * (y - ref_y)**2
            + self.terminal_cost_weight[2] * yaw_diff**2
            + self.terminal_cost_weight[3] * (vel - ref_v)**2
        )

        return np.asarray(terminal_cost, dtype=float)
    
    def _get_nearest_waypoint(self, x: float, y: float, update_prev_idx: bool = False):
        """search the closest waypoint to the vehicle on the reference path"""

        SEARCH_IDX_LEN = 200 # [points] forward search range
        prev_idx = self.prev_waypoints_idx
        dx = [x - ref_x for ref_x in self.ref_path[prev_idx:(prev_idx + SEARCH_IDX_LEN), 0]]
        dy = [y - ref_y for ref_y in self.ref_path[prev_idx:(prev_idx + SEARCH_IDX_LEN), 1]]
        d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]
        min_d = min(d)
        nearest_idx = d.index(min_d) + prev_idx

        # get reference values of the nearest waypoint
        ref_x = self.ref_path[nearest_idx,0]
        ref_y = self.ref_path[nearest_idx,1]
        ref_yaw = self.ref_path[nearest_idx,2]
        ref_v = self.ref_path[nearest_idx,3]

        # update nearest waypoint index if necessary
        if update_prev_idx:
            self.prev_waypoints_idx = nearest_idx 

        return nearest_idx, ref_x, ref_y, ref_yaw, ref_v
    
    def _get_nearest_waypoint_batch(self, x: np.ndarray, y: np.ndarray):
        """search closest waypoints for all samples

        x: shape is (K,)
        y: shape is (K,)
        """
        SEARCH_IDX_LEN = 200
        prev_idx = self.prev_waypoints_idx
        ref_segment = self.ref_path[prev_idx:(prev_idx + SEARCH_IDX_LEN)]

        ref_x = ref_segment[:, 0]
        ref_y = ref_segment[:, 1]

        dx = x[:, None] - ref_x[None, :]
        dy = y[:, None] - ref_y[None, :]
        d = dx**2 + dy**2

        nearest_local_idx = np.argmin(d, axis=1)
        nearest_idx = nearest_local_idx + prev_idx

        ref_x_nearest = self.ref_path[nearest_idx, 0]
        ref_y_nearest = self.ref_path[nearest_idx, 1]
        ref_yaw_nearest = self.ref_path[nearest_idx, 2]
        ref_v_nearest = self.ref_path[nearest_idx, 3]

        return nearest_idx, ref_x_nearest, ref_y_nearest, ref_yaw_nearest, ref_v_nearest

    def _F(self, x_t: np.ndarray, v_t: np.ndarray) -> np.ndarray:
        """calculate next state of the vehicle"""
        # get previous state variables
        x, y, yaw, v = x_t
        steer, accel = v_t

        # prepare params
        l = self.wheel_base
        dt = self.delta_t

        # update state variables
        new_x = x + v * np.cos(yaw) * dt
        new_y = y + v * np.sin(yaw) * dt
        new_yaw = yaw + v / l * np.tan(steer) * dt
        new_v = v + accel * dt

        # return updated state
        x_t_plus_1 = np.array([new_x, new_y, new_yaw, new_v])
        return x_t_plus_1
    
    def _F_batch(self, x_t: np.ndarray, v_t: np.ndarray) -> np.ndarray:
        """calculate next states of the vehicle for all samples

        x_t: shape is (K, dim_x)
        v_t: shape is (K, dim_u)
        """

        x_t = np.asarray(x_t, dtype=float)
        v_t = np.asarray(v_t, dtype=float)

        x = x_t[:, 0]
        y = x_t[:, 1]
        yaw = x_t[:, 2]
        vel = x_t[:, 3]

        steer = v_t[:, 0]
        accel = v_t[:, 1]

        l = self.wheel_base
        dt = self.delta_t

        new_x = x + vel * np.cos(yaw) * dt
        new_y = y + vel * np.sin(yaw) * dt
        new_yaw = yaw + vel / l * np.tan(steer) * dt
        new_v = vel + accel * dt

        x_next = np.empty_like(x_t)
        x_next[:, 0] = new_x
        x_next[:, 1] = new_y
        x_next[:, 2] = new_yaw
        x_next[:, 3] = new_v

        return x_next

    
    
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
        dim = xx.shape[1]
        xx_mean = np.zeros(xx.shape)

        for d in range(dim):
            xx_mean[:,d] = np.convolve(xx[:,d], b, mode="same")
            n_conv = math.ceil(window_size/2)
            xx_mean[0,d] *= window_size/n_conv
            for i in range(1, n_conv):
                xx_mean[i,d] *= window_size/(i+n_conv)
                xx_mean[-i,d] *= window_size/(i + n_conv - (window_size % 2)) 
        return xx_mean


def run_simulation_mppi_pathtracking() -> None:
    """run simulation of pathtracking with MPPI controller"""
    print("[INFO] Start simulation of pathtracking with MPPI controller")

    # simulation settings
    delta_t = 0.05 # [sec]
    sim_steps = 1000 # [steps]
    print(f"[INFO] delta_t : {delta_t:.2f}[s] , sim_steps : {sim_steps}[steps], total_sim_time : {delta_t*sim_steps:.2f}[s]")

    # load the reference path
    ref_path = np.genfromtxt('./data/ovalpath.csv', delimiter=',', skip_header=1)

    # initialize a vehicle as a control target
    vehicle = Vehicle(
        wheel_base=2.5,
        max_steer_abs=0.523, # [rad]
        max_accel_abs=2.000, # [m/s^2]
        ref_path = ref_path[:, 0:2], # ndarray, size is <num_of_waypoints x 2>
    )
    vehicle.reset(
        init_state = np.array([0.0, 1.0, 0.0, 0.0]), # [x[m], y[m], yaw[rad], v[m/s]]
    )

    # initialize a mppi controller for the vehicle
    mppi = MPPIControllerForPathTracking(
        delta_t = delta_t*2.0, # [s]
        wheel_base = 2.5, # [m]
        max_steer_abs = 0.523, # [rad]
        max_accel_abs = 2.000, # [m/s^2]
        ref_path = ref_path, # ndarray, size is <num_of_waypoints x 2>
        horizon_step_T = 20, # [steps]
        number_of_samples_K = 500, # [samples]
        param_exploration = 0.0,
        param_lambda = 100.0,
        param_alpha = 0.98,
        sigma = np.array([[0.075, 0.0], [0.0, 2.0]]),
        stage_cost_weight = np.array([50.0, 50.0, 1.0, 20.0]), # weight for [x, y, yaw, v]
        terminal_cost_weight = np.array([50.0, 50.0, 1.0, 20.0]), # weight for [x, y, yaw, v]
        visualze_sampled_trajs = True
    )

    # simulation loop
    for i in range(sim_steps):

        # get current state of vehicle
        current_state = vehicle.get_state()

        try:
            # calculate input force with MPPI
            optimal_input, optimal_input_sequence, optimal_traj, sampled_traj_list = mppi.calc_control_input(
                observed_x = current_state
            )
        except IndexError as e:
            # the vehicle has reached the end of the reference path
            print("[ERROR] IndexError detected. Terminate simulation.")
            break

        # print current state and input force
        print(f"Time: {i*delta_t:>2.2f}[s], x={current_state[0]:>+3.3f}[m], y={current_state[1]:>+3.3f}[m], yaw={current_state[2]:>+3.3f}[rad], v={current_state[3]:>+3.3f}[m/s], steer={optimal_input[0]:>+6.2f}[rad], accel={optimal_input[1]:>+6.2f}[m/s]")

        # update states of vehicle
        vehicle.update(
            u=optimal_input, 
            delta_t=delta_t, 
            optimal_traj=optimal_traj[:, 0:2], 
            sampled_traj_list=sampled_traj_list[:, :, 0:2]  # ← 元の完璧な状態に戻す！
        )


    # save animation
    
    
    
    vehicle.save_animation(
        "mppi_pathtracking_demo.mp4",
        interval=int(delta_t * 1000),
        movie_writer="ffmpeg"
    )

        





if __name__ == "__main__":
    run_simulation_mppi_pathtracking()
