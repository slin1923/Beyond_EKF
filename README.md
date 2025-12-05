# 🚀 Project: Beyond EKF

## 📌 Goal
Explore and evaluate the performance and runtime characteristics of:
- Unscented Kalman Filter (UKF)
- Particle Filter (PF)
- H-Infinity Filter (H∞)

…for GNSS localization applications.

---

## 📂 Dataset
This project uses the Google SDC 2023 dataset
(farmed from Kaggle).

---

## 🛠️ Setup

It is recommended to create a Python virtual environment:

python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

Then install dependencies:

pip install -r requirements.txt

### 🔑 Key Library
The core dependency for GNSS processing is:

gnss_lib_py

---

## 📜 Script Descriptions

- custom_residuals.py  
  Custom residual function for comparing two different GNSS estimators  
  (not provided by gnss_lib_py by default).

- ukf/pf/hinf.py  
  Implementations of the UKF, Particle Filter, and H∞ Filter.  
  These can be treated as black-box modules.

---

## ▶️ Usage

pip install -r requirements.txt

Then simply run:

python main.py

Follow the prompts to choose a filter, specify parameters, and generate results.

---

## ✨ Notes
This project explores filtering methods beyond the Extended Kalman Filter by evaluating robustness, accuracy, and computational cost on real-world GNSS data.
