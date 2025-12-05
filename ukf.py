import warnings
import numpy as np
from scipy.linalg import sqrtm
from abc import abstractmethod

from gnss_lib_py.navdata.navdata import NavData
from gnss_lib_py.navdata.operations import loop_time
from gnss_lib_py.algorithms.snapshot import solve_wls
from gnss_lib_py.utils.coordinates import ecef_to_geodetic
from gnss_lib_py.utils.filters import BaseFilter

def solve_gnss_ukf(measurements, init_dict=None, params_dict=None, delta_t_decimals=-2):
    """GNSS UKF across timesteps. Interface identical to solve_gnss_ekf."""
    measurements.in_rows(["gps_millis","corr_pr_m","x_sv_m","y_sv_m","z_sv_m"])

    if init_dict is None:
        init_dict = {}

    if "state_0" not in init_dict:
        pos_0 = None
        for _, _, measurement_subset in loop_time(measurements,"gps_millis", delta_t_decimals=delta_t_decimals):
            pos_0 = solve_wls(measurement_subset)
            if pos_0 is not None:
                break
        state_0 = np.zeros((7,1))
        if pos_0 is not None:
            state_0[:3,0] = pos_0[["x_rx_wls_m","y_rx_wls_m","z_rx_wls_m"]]
            state_0[6,0] = pos_0[["b_rx_wls_m"]]
        init_dict["state_0"] = state_0

    init_dict.setdefault("sigma_0", np.eye(init_dict["state_0"].size))
    init_dict.setdefault("Q", np.eye(init_dict["state_0"].size))  # TUNE: process noise
    init_dict.setdefault("R", np.eye(1))  # TUNE: measurement noise

    if params_dict is None:
        params_dict = {}
    params_dict.setdefault("motion_type", "constant_velocity")
    params_dict.setdefault("measure_type", "pseudorange")

    gnss_ukf = GNSSUKF(init_dict, params_dict)
    states = []

    for timestamp, delta_t, measurement_subset in loop_time(measurements,"gps_millis"):
        pos_sv_m = np.atleast_2d(measurement_subset[["x_sv_m","y_sv_m","z_sv_m"]].T)
        corr_pr_m = measurement_subset["corr_pr_m"].reshape(-1,1)
        not_nan_idx = ~np.isnan(pos_sv_m).any(axis=1) & ~np.isnan(corr_pr_m).any(axis=1)
        pos_sv_m = pos_sv_m[not_nan_idx]
        corr_pr_m = corr_pr_m[not_nan_idx]

        gnss_ukf.predict(u=None, predict_dict={"delta_t": delta_t})
        gnss_ukf.update(corr_pr_m, update_dict={"pos_sv_m": pos_sv_m.T, "measurement_noise": np.eye(pos_sv_m.shape[0])})
        states.append([timestamp] + np.squeeze(gnss_ukf.state).tolist())

    states = np.array(states)
    if states.size == 0:
        warnings.warn("No valid state estimate in solve_gnss_ukf.", RuntimeWarning)
        return None

    state_estimate = NavData()
    state_estimate["gps_millis"] = states[:,0]
    state_estimate["x_rx_ukf_m"] = states[:,1]
    state_estimate["y_rx_ukf_m"] = states[:,2]
    state_estimate["z_rx_ukf_m"] = states[:,3]
    state_estimate["vx_rx_ukf_mps"] = states[:,4]
    state_estimate["vy_rx_ukf_mps"] = states[:,5]
    state_estimate["vz_rx_ukf_mps"] = states[:,6]
    state_estimate["b_rx_ukf_m"] = states[:,7]

    lat, lon, alt = ecef_to_geodetic(state_estimate[["x_rx_ukf_m","y_rx_ukf_m","z_rx_ukf_m"]].reshape(3,-1))
    state_estimate["lat_rx_ukf_deg"] = lat
    state_estimate["lon_rx_ukf_deg"] = lon
    state_estimate["alt_rx_ukf_m"] = alt

    return state_estimate

class BaseUnscentedKalmanFilter(BaseFilter):
    def __init__(self, init_dict, params_dict):
        super().__init__(init_dict['state_0'], init_dict['sigma_0'])
        assert _check_square_mat(init_dict['Q'], self.state_dim)
        self.Q = init_dict['Q']
        self.R = init_dict['R']
        self.lam = init_dict.get('lam', 2)  # TUNE: UKF spread parameter
        self.N_sig = init_dict.get('N_sig', 2 * self.state_dim + 1)  # number of sigma points
        self.params_dict = params_dict

    def predict(self, u=None, predict_dict=None):
        """Predict step: propagate sigma points through dynamics."""
        if predict_dict is None:
            predict_dict = {}
        if u is None:
            u = np.zeros((self.state_dim,1))

        x_tm_tm, W = self.U_transform()
        x_t_tm = np.zeros((self.state_dim, self.N_sig))
        for ind in range(self.N_sig):
            x_t_tm[:, [ind]] = self.dyn_model(np.expand_dims(x_tm_tm[:, ind], axis=1), u, predict_dict)

        self.state, self.sigma = self.inv_U_transform(W, x_t_tm)
        self.sigma += self.Q

    def update(self, z, update_dict=None):
        """Update step: correct state estimate with measurements."""
        if update_dict is None:
            update_dict = {}

        m = z.shape[0]
        x_t_tm, W = self.U_transform()
        y_t_tm = np.zeros((m, self.N_sig))
        for ind in range(self.N_sig):
            y_t_tm[:, [ind]] = self.measure_model(x_t_tm[:, [ind]], update_dict)

        y_hat, S_y = self.inv_U_transform(W, y_t_tm)
        S_y += self.R + 1e-8*np.eye(m)

        S_xy = np.zeros((self.state_dim, m))
        for ind in range(self.N_sig):
            S_xy += W[ind] * (x_t_tm[:, [ind]] - self.state) @ (y_t_tm[:, [ind]] - y_hat).T

        K = S_xy @ np.linalg.pinv(S_y)
        self.state += K @ (z - y_hat)
        self.sigma -= K @ S_y @ K.T

    def U_transform(self):
        """Generate sigma points and weights using unscented transform."""
        N = self.state_dim
        X = np.zeros((N, self.N_sig))
        W = np.zeros((self.N_sig, 1))
        delta = sqrtm((self.lam + N) * (self.sigma + 1e-12*np.eye(N))).real
        X[:, 0] = self.state.squeeze()
        for i in range(N):
            X[:, i+1] = self.state.squeeze() + delta[:, i]
            X[:, i+1+N] = self.state.squeeze() - delta[:, i]
        W[0] = self.lam / (self.lam + N)
        W[1:] = 1 / (2*(self.lam+N))
        return X, W

    def inv_U_transform(self, W, x):
        """Recover mean and covariance from sigma points."""
        mu = np.sum(W.T * x, axis=1, keepdims=True)
        x_hat = x - mu
        S = sum(W[i] * np.outer(x_hat[:, i], x_hat[:, i]) for i in range(self.N_sig))
        return mu, S

    @abstractmethod
    def measure_model(self, x, update_dict=None):
        raise NotImplementedError

    @abstractmethod
    def dyn_model(self, x, u, predict_dict=None):
        raise NotImplementedError

class GNSSUKF(BaseUnscentedKalmanFilter):
    """GNSS UKF: 7-state [x, y, z, vx, vy, vz, b]"""
    def __init__(self, init_dict, params_dict):
        super().__init__(init_dict, params_dict)
        self.delta_t = params_dict.get("dt", 1.0)
        self.motion_type = params_dict.get("motion_type","stationary")
        self.measure_type = params_dict.get("measure_type","pseudorange")

    def dyn_model(self, x, u, predict_dict=None):
        return self._get_A(predict_dict) @ x

    def measure_model(self, x, update_dict):
        if self.measure_type == "pseudorange":
            pos_sv_m = update_dict["pos_sv_m"]
            return np.sqrt((x[0]-pos_sv_m[0,:])**2 + (x[1]-pos_sv_m[1,:])**2 + (x[2]-pos_sv_m[2,:])**2).reshape(-1,1) + x[6]
        raise NotImplementedError

    def _get_A(self, predict_dict=None):
        dt = predict_dict.get("delta_t", self.delta_t) if predict_dict else self.delta_t
        A = np.eye(7)
        if self.motion_type == "constant_velocity":
            A[:3, 3:6] = dt*np.eye(3)
        return A

def _check_col_vect(vect, dim):
    return np.shape(vect) == (dim, 1)

def _check_square_mat(mat, dim):
    return np.shape(mat) == (dim, dim)