import matplotlib.pyplot as plt
import numpy as np
import time
import gnss_lib_py as glp
from custom_residuals import compute_position_residual
from ukf import solve_gnss_ukf
from pf import solve_gnss_pf
from h_inf import solve_gnss_hinf

# TO DO: set data start index to skip initial convergence period
DATASTART = 200 # where to stop considering results (downsampled for plotting)

# TO DO: set paths to SDC dataset files
ground_truth_path = "SDC_data/ground_truth_3.csv"
raw_data_path = "SDC_data/device_gnss_3.csv"
num_dataset = 3

# Load GNSS data from SDC dataset and take references
ground_truth = glp.AndroidGroundTruth2023(ground_truth_path) 
ground_truth = ground_truth.remove(cols=list(range(DATASTART, ground_truth.shape[1])))
derived_data = glp.AndroidDerived2023(raw_data_path)
gps_data = derived_data.where("gnss_id","gps") # isolate GPS measurements only

# EKF SOLVER (built into gnss_lib_py)
print("-----Solving reference EKF estimate-----")
t0_ekf = time.time()
ekf_solved_ref = glp.solve_gnss_ekf(gps_data) # reference EKF solution
ekf_runtime = time.time() - t0_ekf
ekf_solved_ref = ekf_solved_ref.remove(cols=list(range(DATASTART, ekf_solved_ref.shape[1])))
print(f"Reference EKF solved in {ekf_runtime:.2f} seconds.")
ekf_residuals = compute_position_residual(ground_truth, ekf_solved_ref, est_tag="ekf", gt_tag="gt")
ekf_rmse = np.sqrt(np.mean(ekf_residuals**2))
print(f"Reference EKF RMSE: {ekf_rmse:.2f} m")

# WLS SOLVER (built into gnss_lib_py)
print("-----Solving reference WLS estimate-----")
t0_wls = time.time()
wls_solved_ref = glp.solve_wls(gps_data) # reference WLS solution
wls_runtime = time.time() - t0_wls
wls_solved_ref = wls_solved_ref.remove(cols=list(range(DATASTART, wls_solved_ref.shape[1])))
print(f"Reference WLS solved in {wls_runtime:.2f} seconds.")
wls_residuals = compute_position_residual(ground_truth, wls_solved_ref, est_tag="wls", gt_tag="gt")
wls_rmse = np.sqrt(np.mean(wls_residuals**2))
print(f"Reference WLS RMSE: {wls_rmse:.2f} m")

fig = glp.plot_map(wls_solved_ref, ekf_solved_ref, ground_truth)
fig.write_image(f"ref_figures/reference_solutions_ds{num_dataset}.png", scale=3)
print("-----Reference Estimates plotted and saved-----")

# User selects which filter to run
valid_filters = {"ukf", "pf", "hinf", "all"}
while True:
    filter_choice = input("Select filter to run (ukf, pf, hinf, all): ").strip().lower()
    if filter_choice in valid_filters:
        break
    print("Invalid choice. Please enter 'ukf', 'pf', 'hinf', or 'all'.")


filters_to_run = ["ukf", "hinf", "pf"] if filter_choice == "all" else [filter_choice]

for filt in filters_to_run:
    if filt == "ukf":
        lambdas = input("Enter UKF lambda values separated by commas (or press Enter for default 2): ").strip()
        ukf_runtimes = []
        ukf_solutions = {}
        ukf_rmses = []
        for lambd in (lambdas.split(",") if lambdas else ["2"]):
            print(f"-----Running UKF with lambda={lambd}-----")
            t0_ukf = time.time()
            ukf_solution = solve_gnss_ukf(gps_data, init_dict={"lam": float(lambd)})
            ukf_runtime = time.time() - t0_ukf
            ukf_solution = ukf_solution.remove(cols=list(range(DATASTART, ukf_solution.shape[1])))
            ukf_runtimes.append(ukf_runtime)
            ukf_solutions[lambd] = ukf_solution
            print(f"UKF with lambda={lambd} solved in {ukf_runtime:.2f} seconds.")
            ukf_residuals = compute_position_residual(ground_truth, ukf_solution, est_tag="ukf", gt_tag="gt")
            ukf_rmse = np.sqrt(np.mean(ukf_residuals**2))
            ukf_rmses.append(ukf_rmse)
            print(f"UKF with lambda={lambd} RMSE: {ukf_rmse:.2f} m")

            # Save image
            fig = glp.plot_map(ukf_solution, wls_solved_ref, ground_truth)
            fig.write_image(f"ukf_solution_lambda_{lambd}_ds{num_dataset}.png", scale=3)

    elif filt == "hinf":
        gammas = input("Enter H-infinity gamma values separated by commas (or press Enter for default 2.0): ").strip()
        hinf_runtimes = []
        hinf_solutions = {}
        hinf_rmses = []
        for gamma in (gammas.split(",") if gammas else ["2.0"]):
            print(f"-----Running H-infinity filter with gamma={gamma}-----")
            t0_hinf = time.time()
            hinf_solution = solve_gnss_hinf(gps_data, init_dict={"gamma": float(gamma)})
            hinf_runtime = time.time() - t0_hinf
            hinf_solution = hinf_solution.remove(cols=list(range(DATASTART, hinf_solution.shape[1])))
            hinf_runtimes.append(hinf_runtime)
            hinf_solutions[gamma] = hinf_solution
            print(f"H∞ with gamma={gamma} solved in {hinf_runtime:.2f} seconds.")
            hinf_residuals = compute_position_residual(ground_truth, hinf_solution, est_tag="hinf", gt_tag="gt")
            hinf_rmse = np.sqrt(np.mean(hinf_residuals**2))
            hinf_rmses.append(hinf_rmse)
            print(f"H∞ with gamma={gamma} RMSE: {hinf_rmse:.2f} m")

            # Save image
            fig = glp.plot_map(hinf_solution, wls_solved_ref, ground_truth)
            fig.write_image(f"hinf_solution_gamma_{gamma}_ds{num_dataset}.png", scale=3)

    elif filt == "pf":
        Nps = input("Enter Particle Filter Np values separated by commas (or press Enter for default 100): ").strip()
        pf_runtimes = []
        pf_solutions = {}
        pf_rmses = []
        for Np in (Nps.split(",") if Nps else ["100"]):
            Np_int = int(Np)
            print(f"-----Running Particle Filter with Np={Np_int}-----")
            t0_pf = time.time()
            pf_solution = solve_gnss_pf(gps_data, params_dict={"Np": Np_int})
            pf_runtime = time.time() - t0_pf

            pf_solution = pf_solution.remove(cols=list(range(DATASTART, pf_solution.shape[1])))
            pf_runtimes.append(pf_runtime)
            pf_solutions[Np] = pf_solution
            print(f"Particle Filter with Np={Np_int} simulated in {pf_runtime:.2f} seconds.")
            pf_residuals = compute_position_residual(ground_truth, pf_solution, est_tag="pf", gt_tag="gt")
            pf_rmse = np.sqrt(np.mean(pf_residuals**2))
            pf_rmses.append(pf_rmse)
            print(f"Particle Filter with Np={Np_int} simulated RMSE: {pf_rmse:.2f} m")

            # Save image
            fig = glp.plot_map(pf_solution, ground_truth)
            fig.write_image(f"pf_solution_Np_{Np_int}_ds{num_dataset}.png", scale=3)