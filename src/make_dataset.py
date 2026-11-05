"""
Generate a design-of-experiments dataset with the calibrated physics model.
Latin hypercube sampling over 6 inputs -> natural frequency, RMS stress and fatigue life.
"""
import os
import time
import numpy as np
import pandas as pd
from scipy.stats import qmc
from physics import analyse

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "dataset.csv")

# Connector and modulus values identified in calibrate.py (stage 1/2), held fixed here
CALIBRATED = dict(k_c=6.0e4, c_c=3.0, e_scale=1.15)

BOUNDS = {                       # name: (low, high)
    "L_mm":    (40.0, 70.0),     # bracket length
    "t_mm":    (5.0, 8.0),       # bracket thickness
    "m_mod_g": (50.0, 200.0),    # module mass carried through the connector
    "T_C":     (-40.0, 105.0),   # operating temperature
    "alpha":   (8.0, 35.0),      # Rayleigh mass-proportional damping [1/s]
    "grms":    (0.5, 2.0),       # 10-200 Hz white-noise level [g RMS]
}
INPUTS = list(BOUNDS)


def run_design(row):
    L, t, m, T, a, g = row
    return analyse(L / 1e3, t / 1e3, m / 1e3, T, a, g, **CALIBRATED)


def main(n=2000, seed=0):
    lo = np.array([b[0] for b in BOUNDS.values()])
    hi = np.array([b[1] for b in BOUNDS.values()])
    X = qmc.scale(qmc.LatinHypercube(d=len(BOUNDS), seed=seed).random(n), lo, hi)
    t0 = time.time()
    rows = []
    for x in X:
        r = run_design(x)
        rows.append([*x, r["f1"], r["f2"], r["sigma_rms"], np.log10(r["life_h"]), np.log10(r["life_nb_h"])])
    df = pd.DataFrame(rows, columns=INPUTS + ["f1_Hz", "f2_Hz", "sigma_rms_MPa", "log10_life_h", "log10_life_nb_h"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(df.describe().round(3).to_string())
    print(f"saved {len(df)} samples in {time.time() - t0:.0f} s -> {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main()
