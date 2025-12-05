import matplotlib.pyplot as plt
import numpy as np
import time
import gnss_lib_py as glp
from custom_residuals import compute_position_residual
from ukf import solve_gnss_ukf
from pf import solve_gnss_pf
from h_inf import solve_gnss_hinf

# TO DO: set paths to SDC dataset files
ground_truth_path = "SDC_data/ground_truth_2.csv"
raw_data_path = "SDC_data/device_gnss_2.csv"
num_dataset = 2

# Load GNSS data from SDC dataset and take references
ground_truth = glp.AndroidGroundTruth2023(ground_truth_path) 
derived_data = glp.AndroidDerived2023(raw_data_path)
gps_data = derived_data.where("gnss_id","gps") # isolate GPS measurements only

# EKF SOLVER (built into gnss_lib_py)
print("-----Solving reference EKF estimate-----")
t0_ekf = time.time()
ekf_solved_ref = glp.solve_gnss_ekf(gps_data) # reference EKF solution
ekf_runtime = time.time() - t0_ekf
print(f"Reference EKF solved in {ekf_runtime:.2f} seconds.")
ekf_residuals = compute_position_residual(ground_truth, ekf_solved_ref, est_tag="ekf", gt_tag="gt")
ekf_rmse = np.sqrt(np.mean(ekf_residuals**2))
print(f"Reference EKF RMSE: {ekf_rmse:.2f} m")

# WLS SOLVER (built into gnss_lib_py)
print("-----Solving reference WLS estimate-----")
t0_wls = time.time()
wls_solved_ref = glp.solve_wls(gps_data) # reference WLS solution
wls_runtime = time.time() - t0_wls
print(f"Reference WLS solved in {wls_runtime:.2f} seconds.")
wls_residuals = compute_position_residual(ground_truth, wls_solved_ref, est_tag="wls", gt_tag="gt")
wls_rmse = np.sqrt(np.mean(wls_residuals**2))
print(f"Reference WLS RMSE: {wls_rmse:.2f} m")

fig = glp.plot_map(wls_solved_ref, ekf_solved_ref, ground_truth)
fig.write_image(f"ref_figures/reference_solutions_ds{num_dataset}.png", scale=3)
print("-----Reference Estimates plotted and saved-----")

# User selects which filter to run
valid_filters = {"ukf", "pf", "hinf"}
while True:
    filter = input("Select filter to run (ukf, pf, hinf): ").strip().lower()
    if filter in valid_filters:
        break
    print("Invalid choice. Please enter 'ukf', 'pf', or 'h_inf'.")


if filter == "ukf":
    lambdas = input("Enter UKF lambda values separated by commas (or press Enter for default 2): ").strip()
    ukf_runtimes = []
    ukf_solutions = {}
    ukf_rmses = []
    for lambd in (lambdas.split(",") if lambdas else ["2"]):
        print(f"-----Running UKF with lambda={lambd}-----")
        t0_ukf = time.time()
        ukf_solution = solve_gnss_ukf(gps_data, init_dict={"lam": float(lambd)})
        ukf_runtime = time.time() - t0_ukf
        ukf_runtimes.append(ukf_runtime)
        ukf_solutions[lambd] = ukf_solution
        print(f"UKF with lambda={lambd} solved in {ukf_runtime:.2f} seconds.")
        ukf_residuals = compute_position_residual(ground_truth, ukf_solution, est_tag="ukf", gt_tag="gt")
        ukf_rmse = np.sqrt(np.mean(ukf_residuals**2))
        ukf_rmses.append(ukf_rmse)
        print(f"UKF with lambda={lambd} RMSE: {ukf_rmse:.2f} m")

        # Save image
        fig = glp.plot_map(ukf_solution, ground_truth)
        fig.write_image(f"ukf_solution_lambda_{lambd}_ds{num_dataset}.png", scale=3)

elif filter == "hinf":
    gammas = input("Enter H-infinity gamma values separated by commas (or press Enter for default 2.0): ").strip()
    hinf_runtimes = []
    hinf_solutions = {}
    hinf_rmses = []
    for gamma in (gammas.split(",") if gammas else ["2.0"]):
        print(f"-----Running H-infinity filter with gamma={gamma}-----")
        t0_hinf = time.time()
        hinf_solution = solve_gnss_hinf(gps_data, init_dict={"gamma": float(gamma)})
        hinf_runtime = time.time() - t0_hinf
        hinf_runtimes.append(hinf_runtime)
        hinf_solutions[gamma] = hinf_solution
        print(f"H∞ with gamma={gamma} solved in {hinf_runtime:.2f} seconds.")
        hinf_residuals = compute_position_residual(ground_truth, hinf_solution, est_tag="hinf", gt_tag="gt")
        hinf_rmse = np.sqrt(np.mean(hinf_residuals**2))
        hinf_rmses.append(hinf_rmse)
        print(f"H∞ with gamma={gamma} RMSE: {hinf_rmse:.2f} m")

        # Save image
        fig = glp.plot_map(hinf_solution, ground_truth)
        fig.write_image(f"hinf_solution_gamma_{gamma}_ds{num_dataset}.png", scale=3)

