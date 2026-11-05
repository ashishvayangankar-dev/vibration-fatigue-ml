#!/usr/bin/env bash
# Reproduce every result and figure (about 6 minutes on a laptop CPU)
set -e
cd "$(dirname "$0")/src"
python3 calibrate.py            # 1. test-based calibration: direct search vs Bayesian optimization
python3 make_dataset.py         # 2. DoE dataset with the calibrated model
python3 train_surrogates.py     # 3. GPR / random forest / XGBoost / neural-network surrogates
python3 design_optimization.py  # 4. worst-case-temperature design with the GPR surrogate
python3 pinn_modal_id.py        # 5. PINN identification of natural frequency and damping
