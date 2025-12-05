from gnss_lib_py.navdata.navdata import NavData
import numpy as np

import numpy as np

def compute_position_residual(gt_nav: NavData,
                              est_nav: NavData,
                              est_tag: str = "ekf",
                              gt_tag: str = "gt",
                              time_field: str = "gps_millis"):
    """
    Compute 3D position residuals between ground-truth and estimate NavData,
    with automatic time alignment and interpolation.

    Parameters
    ----------
    gt_nav : NavData
        Ground truth navdata object.
    est_nav : NavData
        Estimate navdata object.
    est_tag : str
        Suffix for estimate fields (e.g., 'ekf', 'pf', 'ukf').
    gt_tag : str
        Suffix for GT fields (e.g., 'gt').
    time_field : str
        Field used to align data (default 'gps_millis').

    Returns
    -------
    residuals : np.ndarray
        Vector of per-epoch 3D position residual magnitudes (meters), aligned to GT timestamps.
    """

    pos_fields = ["x_rx_{}_m", "y_rx_{}_m", "z_rx_{}_m"]
    gt_fields = [f.format(gt_tag) for f in pos_fields]
    est_fields = [f.format(est_tag) for f in pos_fields]

    # ---- Field existence checks ----
    for f in gt_fields + [time_field]:
        if f not in gt_nav.rows:
            raise KeyError(f"GT NavData missing field '{f}'")

    for f in est_fields + [time_field]:
        if f not in est_nav.rows:
            raise KeyError(f"Estimate NavData missing field '{f}'")

    # ---- Extract time vectors ----
    t_gt = np.asarray(gt_nav[time_field], dtype=float)
    t_est = np.asarray(est_nav[time_field], dtype=float)

    # ---- Extract GT positions ----
    gt_xyz = np.vstack([np.asarray(gt_nav[f]) for f in gt_fields]).T  # (N_gt, 3)

    # ---- Extract estimate positions to be interpolated ----
    est_xyz_raw = np.vstack([np.asarray(est_nav[f]) for f in est_fields]).T  # (N_est, 3)

    # ---- Interpolate estimate onto GT timeline ----
    est_xyz_interp = np.vstack([
        np.interp(t_gt, t_est, est_xyz_raw[:, 0]),
        np.interp(t_gt, t_est, est_xyz_raw[:, 1]),
        np.interp(t_gt, t_est, est_xyz_raw[:, 2]),
    ]).T

    # ---- Compute 3D residual magnitude ----
    diff = est_xyz_interp - gt_xyz
    residuals = np.linalg.norm(diff, axis=1)

    return residuals
