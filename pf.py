import warnings
import numpy as np
from gnss_lib_py.navdata.navdata import NavData
from gnss_lib_py.navdata.operations import loop_time
from gnss_lib_py.algorithms.snapshot import solve_wls
from gnss_lib_py.utils.coordinates import ecef_to_geodetic
from gnss_lib_py.utils.filters import BaseFilter

R_FIX = 1 # measurement noise variance

class GNSSPF(BaseFilter):
    """GNSS-only Particle Filter (3D pos, 3D vel, clock bias) - Vectorized"""
    def __init__(self, init_dict, params_dict):
        super().__init__(init_dict['state_0'], init_dict['sigma_0'])
        self.Q = init_dict['Q']
        self.R = init_dict['R']
        self.delta_t = params_dict.get("dt", 1)
        self.motion_type = params_dict.get("motion_type","stationary")
        self.measure_type = params_dict.get("measure_type","pseudorange")
        self.Np = params_dict.get("Np", 10)

        # Vectorized: Initialize all particles at once
        self.particles = np.repeat(self.state, self.Np, axis=1)
        self.particles += np.random.multivariate_normal(np.zeros(7), self.sigma, self.Np).T # add initial noise
        self.weights = np.ones((self.Np,1)) / self.Np

    def predict(self, predict_dict=None):
        """Vectorized predict: propagate all particles at once."""
        if predict_dict is None:
            predict_dict = {}
        delta_t = predict_dict.get("delta_t", self.delta_t)

        A = self._get_A(delta_t)
        Q_scaled = self.Q * delta_t
        
        # Vectorized: propagate all particles + add noise (7 x Np)
        self.particles = A @ self.particles + np.random.multivariate_normal(np.zeros(7), Q_scaled, self.Np).T
        
        # Vectorized weighted mean
        self.state = (self.particles * self.weights.T).sum(axis=1, keepdims=True)
        self.sigma = self._compute_covariance()

    def update(self, z, update_dict=None):
        """Vectorized update: compute likelihoods for all particles at once."""
        if update_dict is None:
            update_dict = {}

        pos_sv_m = update_dict['pos_sv_m']  # (3 x m)
        m = pos_sv_m.shape[1]
        R = update_dict.get("measurement_noise", self.R)
        z_flat = z.flatten()  # (m,)

        # Vectorized: compute pseudorange for all particles (m x Np)
        if self.measure_type == "pseudorange":
            # particles: (7 x Np), pos_sv_m: (3 x m)
            # Broadcast: particles[:3,:,None] - pos_sv_m[:,None,:] -> (3 x Np x m)
            dx = self.particles[0,:,None] - pos_sv_m[0,None,:]  # (Np x m)
            dy = self.particles[1,:,None] - pos_sv_m[1,None,:]
            dz = self.particles[2,:,None] - pos_sv_m[2,None,:]
            ranges = np.sqrt(dx**2 + dy**2 + dz**2)  # (Np x m)
            pred = ranges + self.particles[6,:,None]  # add clock bias (Np x m)
        else:
            raise NotImplementedError

        # Vectorized: compute log-likelihoods for all particles
        # residuals: (Np x m), z_flat: (m,)
        residuals = pred - z_flat[None,:]  # (Np x m)
        R_inv = np.linalg.inv(R)
        
        # Compute quadratic form: sum over measurements
        # For each particle: residual[i] @ R_inv @ residual[i].T
        log_likelihoods = -0.5 * np.sum((residuals @ R_inv) * residuals, axis=1)  # (Np,)
        
        # Numerical stability
        log_likelihoods -= np.max(log_likelihoods)
        likelihoods = np.exp(log_likelihoods)
        
        # Update weights
        self.weights = (self.weights.flatten() * likelihoods).reshape(-1,1)
        self.weights += 1e-300 # avoid zeros
        self.weights /= np.sum(self.weights) # normalize

        # Resample if needed
        Neff = 1.0 / np.sum(self.weights**2)
        if Neff < self.Np / 2:
            self._resample_particles()

        # Update estimates
        self.state = (self.particles * self.weights.T).sum(axis=1, keepdims=True)
        self.sigma = self._compute_covariance()

    def _resample_particles(self):
        """Systematic resampling."""
        positions = (np.arange(self.Np) + np.random.rand()) / self.Np
        indexes = np.zeros(self.Np, dtype=int)
        cumulative_sum = np.cumsum(self.weights.flatten())
        i, j = 0, 0
        while i < self.Np:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        self.particles = self.particles[:, indexes]
        self.weights.fill(1.0 / self.Np)

    def _compute_covariance(self):
        """Vectorized covariance computation."""
        diff = self.particles - self.state  # (7 x Np)
        # Weighted outer products: sum_i w_i * diff_i * diff_i^T
        return (diff * self.weights.T) @ diff.T

    def _get_A(self, delta_t):
        """State transition matrix."""
        A = np.eye(7)
        if self.motion_type == "constant_velocity":
            A[:3, 3:6] = delta_t * np.eye(3)
        return A

def solve_gnss_pf(measurements, init_dict=None, params_dict=None, delta_t_decimals=-2):
    """GNSS Particle Filter across timesteps. Drop-in replacement."""
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

    # TUNE THESE: Start with smaller process noise for PF!
    init_dict.setdefault("sigma_0", np.eye(init_dict["state_0"].size) * 10)  # Initial uncertainty
    init_dict.setdefault("Q", np.eye(init_dict["state_0"].size))  # TUNE: process noise (small!)
    init_dict.setdefault("R", np.eye(1) * R_FIX)  # TUNE: measurement noise

    if params_dict is None:
        params_dict = {}
    params_dict.setdefault("motion_type", "constant_velocity")
    params_dict.setdefault("measure_type", "pseudorange")
    params_dict.setdefault("Np", 10)  # TUNE: number of particles (try 100-500 for better performance)

    gnss_pf = GNSSPF(init_dict, params_dict)
    states = []

    for timestamp, delta_t, measurement_subset in loop_time(measurements,"gps_millis"):
        pos_sv_m = np.atleast_2d(measurement_subset[["x_sv_m","y_sv_m","z_sv_m"]].T)
        corr_pr_m = measurement_subset["corr_pr_m"].reshape(-1,1)

        not_nan_idx = ~np.isnan(pos_sv_m).any(axis=1) & ~np.isnan(corr_pr_m).any(axis=1)
        pos_sv_m = pos_sv_m[not_nan_idx]
        corr_pr_m = corr_pr_m[not_nan_idx]

        gnss_pf.predict(predict_dict={"delta_t": delta_t})
        gnss_pf.update(corr_pr_m, update_dict={"pos_sv_m": pos_sv_m.T,
                                               "measurement_noise": np.eye(pos_sv_m.shape[0]) * R_FIX})

        states.append([timestamp] + np.squeeze(gnss_pf.state).tolist())

    states = np.array(states)
    if states.size == 0:
        warnings.warn("No valid state estimate in solve_gnss_pf.", RuntimeWarning)
        return None

    state_estimate = NavData()
    state_estimate["gps_millis"] = states[:,0]
    state_estimate["x_rx_pf_m"] = states[:,1]
    state_estimate["y_rx_pf_m"] = states[:,2]
    state_estimate["z_rx_pf_m"] = states[:,3]
    state_estimate["vx_rx_pf_mps"] = states[:,4]
    state_estimate["vy_rx_pf_mps"] = states[:,5]
    state_estimate["vz_rx_pf_mps"] = states[:,6]
    state_estimate["b_rx_pf_m"] = states[:,7]

    lat, lon, alt = ecef_to_geodetic(state_estimate[["x_rx_pf_m","y_rx_pf_m","z_rx_pf_m"]].reshape(3,-1))
    state_estimate["lat_rx_pf_deg"] = lat
    state_estimate["lon_rx_pf_deg"] = lon
    state_estimate["alt_rx_pf_m"] = alt

    return state_estimate