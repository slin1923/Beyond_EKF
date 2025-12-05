import warnings
import numpy as np

from gnss_lib_py.navdata.navdata import NavData
from gnss_lib_py.navdata.operations import loop_time
from gnss_lib_py.algorithms.snapshot import solve_wls
from gnss_lib_py.utils.coordinates import ecef_to_geodetic
from gnss_lib_py.utils.filters import BaseExtendedKalmanFilter

def solve_gnss_hinf(measurements, init_dict=None, params_dict=None, delta_t_decimals=-2):
    """GNSS H-infinity filter across timesteps. Interface identical to solve_gnss_ekf."""
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
    init_dict.setdefault("gamma", 2.0)  # TUNE: robustness param (must be > 1, larger = more robust)

    if params_dict is None:
        params_dict = {}
    params_dict.setdefault("motion_type", "constant_velocity")
    params_dict.setdefault("measure_type", "pseudorange")

    gnss_hinf = GNSSHInfinity(init_dict, params_dict)
    states = []

    for timestamp, delta_t, measurement_subset in loop_time(measurements,"gps_millis"):
        pos_sv_m = np.atleast_2d(measurement_subset[["x_sv_m","y_sv_m","z_sv_m"]].T)
        corr_pr_m = measurement_subset["corr_pr_m"].reshape(-1,1)
        not_nan_idx = ~np.isnan(pos_sv_m).any(axis=1) & ~np.isnan(corr_pr_m).any(axis=1)
        pos_sv_m = pos_sv_m[not_nan_idx]
        corr_pr_m = corr_pr_m[not_nan_idx]

        gnss_hinf.predict(predict_dict={"delta_t": delta_t})
        gnss_hinf.update(corr_pr_m, update_dict={"pos_sv_m": pos_sv_m.T, "measurement_noise": np.eye(pos_sv_m.shape[0])})
        states.append([timestamp] + np.squeeze(gnss_hinf.state).tolist())

    states = np.array(states)
    if states.size == 0:
        warnings.warn("No valid state estimate in solve_gnss_hinf.", RuntimeWarning)
        return None

    state_estimate = NavData()
    state_estimate["gps_millis"] = states[:,0]
    state_estimate["x_rx_hinf_m"] = states[:,1]
    state_estimate["y_rx_hinf_m"] = states[:,2]
    state_estimate["z_rx_hinf_m"] = states[:,3]
    state_estimate["vx_rx_hinf_mps"] = states[:,4]
    state_estimate["vy_rx_hinf_mps"] = states[:,5]
    state_estimate["vz_rx_hinf_mps"] = states[:,6]
    state_estimate["b_rx_hinf_m"] = states[:,7]

    lat, lon, alt = ecef_to_geodetic(state_estimate[["x_rx_hinf_m","y_rx_hinf_m","z_rx_hinf_m"]].reshape(3,-1))
    state_estimate["lat_rx_hinf_deg"] = lat
    state_estimate["lon_rx_hinf_deg"] = lon
    state_estimate["alt_rx_hinf_m"] = alt

    return state_estimate

class GNSSHInfinity(BaseExtendedKalmanFilter):
    """GNSS H-infinity filter: 7-state [x, y, z, vx, vy, vz, b] with robustness param gamma."""
    def __init__(self, init_dict, params_dict):
        super().__init__(init_dict, params_dict)
        self.gamma = init_dict.get('gamma', 2.0)
        self.delta_t = params_dict.get('dt', 1.0)
        self.motion_type = params_dict.get('motion_type', 'stationary')
        self.measure_type = params_dict.get('measure_type', 'pseudorange')

    def predict(self, u=None, predict_dict=None):
        """Predict step: propagate state and covariance."""
        if u is None:
            u = np.zeros((self.state_dim, 1))
        if predict_dict is None:
            predict_dict = {}
        assert _check_col_vect(u, np.size(u)), "Control input not column vector"
        
        self.state = self.dyn_model(u, predict_dict)
        A = self.linearize_dynamics(predict_dict)
        self.sigma = A @ self.sigma @ A.T + self.Q
        
        assert _check_col_vect(self.state, self.state_dim), "Bad state shape after predict"
        assert _check_square_mat(self.sigma, self.state_dim), "Bad covariance shape after predict"

    def update(self, z, update_dict=None):
        """H-infinity update: S = R + H*P*H' + gamma^(-2)*I, K = P*H'*S^(-1)"""
        if update_dict is None:
            update_dict = {}
        measurement_noise = update_dict.get('measurement_noise', self.R)
        assert _check_col_vect(z, np.size(z)), "Measurements not column vector"
        
        H = self.linearize_measurements(update_dict)
        z_expect = self.measure_model(update_dict)
        assert _check_col_vect(z_expect, np.size(z)), "Expected measurements not column vector"
        
        # H-infinity innovation covariance
        m = H.shape[0]
        S_hinf = (self.gamma**(-2)) * np.eye(m) + H @ self.sigma @ H.T + measurement_noise
        
        try:
            S_hinf_inv = np.linalg.inv(S_hinf)
        except np.linalg.LinAlgError:
            warnings.warn(f"S_hinf singular with gamma={self.gamma}.", RuntimeWarning)
            S_hinf_inv = np.linalg.pinv(S_hinf)
        
        K_hinf = self.sigma @ H.T @ S_hinf_inv
        self.state += K_hinf @ (z - z_expect)
        self.sigma -= K_hinf @ H @ self.sigma
        
        assert _check_col_vect(self.state, self.state_dim), "Bad state shape after update"
        assert _check_square_mat(self.sigma, self.state_dim), "Bad covariance shape after update"

    def dyn_model(self, u, predict_dict=None):
        """Dynamics: x_new = A @ x"""
        return self._get_A(predict_dict) @ self.state

    def measure_model(self, update_dict):
        """Pseudorange: rho = ||x_rx - x_sv|| + b"""
        if self.measure_type == 'pseudorange':
            pos_sv_m = update_dict['pos_sv_m']
            return (np.sqrt((self.state[0] - pos_sv_m[0,:])**2 + 
                           (self.state[1] - pos_sv_m[1,:])**2 + 
                           (self.state[2] - pos_sv_m[2,:])**2) + self.state[6]).reshape(-1,1)
        raise NotImplementedError

    def linearize_dynamics(self, predict_dict=None):
        """State transition matrix A."""
        return self._get_A(predict_dict)

    def linearize_measurements(self, update_dict):
        """Measurement Jacobian H."""
        if self.measure_type == 'pseudorange':
            pos_sv_m = update_dict['pos_sv_m']
            m = pos_sv_m.shape[1]
            H = np.zeros([m, self.state_dim])
            pseudo_expect = self.measure_model(update_dict)
            rx_pos = self.state[:3].reshape(-1, 1)
            H[:, :3] = (rx_pos - pos_sv_m).T / pseudo_expect
            H[:, 6] = 1
            return H
        raise NotImplementedError

    def _get_A(self, predict_dict=None):
        """Build state transition matrix."""
        dt = predict_dict.get("delta_t", self.delta_t) if predict_dict else self.delta_t
        A = np.eye(7)
        if self.motion_type == "constant_velocity":
            A[:3, 3:6] = dt * np.eye(3)
        return A

def _check_col_vect(vect, dim):
    return np.shape(vect) == (dim, 1)

def _check_square_mat(mat, dim):
    return np.shape(mat) == (dim, dim)