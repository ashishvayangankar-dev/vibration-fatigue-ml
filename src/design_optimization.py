"""
Surrogate-based, uncertainty-aware design:
find the thinnest (lightest) bracket that survives the vibration test spectrum
at EVERY temperature between -40 and 105 degC, using the trained GPR surrogate,
then verify the chosen design with the physics model.
"""
import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_surrogates import features
from make_dataset import INPUTS, run_design

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

# Fixed design case (component, test spectrum, damping)
CASE = dict(L_mm=60.0, m_mod_g=150.0, alpha=17.3, grms=1.0)
TARGET_LIFE_H = 2000.0
T_GRID = np.linspace(-40, 105, 30)
T_MM_GRID = np.linspace(5.0, 8.0, 301)


def predict(gpr, t_mm, T):
    df = pd.DataFrame({"L_mm": CASE["L_mm"], "t_mm": t_mm, "m_mod_g": CASE["m_mod_g"], "T_C": T,
                       "alpha": CASE["alpha"], "grms": CASE["grms"]})[INPUTS]
    return gpr.predict(features(df), return_std=True)


def main():
    gpr = joblib.load(os.path.join(RES, "gpr_life_model.joblib"))
    tt, TT = np.meshgrid(T_MM_GRID, T_GRID, indexing="ij")
    mu, sd = predict(gpr, tt.ravel(), TT.ravel())
    mu, sd = mu.reshape(tt.shape), sd.reshape(tt.shape)

    worst_mean = mu.min(axis=1)                    # worst temperature for each thickness
    worst_cons = (mu - 2 * sd).min(axis=1)         # conservative: mean - 2 sigma
    worst_T = T_GRID[mu.argmin(axis=1)]
    log_target = np.log10(TARGET_LIFE_H)

    i_opt = int(np.argmax(worst_cons >= log_target))
    t_opt = T_MM_GRID[i_opt]

    # Verify with the physics model at all temperatures
    phys_life = lambda t, T: np.log10(run_design([CASE["L_mm"], t, CASE["m_mod_g"], T, CASE["alpha"], CASE["grms"]])["life_h"])
    phys = np.array([phys_life(t_opt, T) for T in T_GRID])
    summary = dict(case=CASE, target_life_h=TARGET_LIFE_H, chosen_thickness_mm=round(float(t_opt), 3),
                   governing_temperature_C=float(worst_T[i_opt]),
                   surrogate_worst_life_h=float(10 ** worst_mean[i_opt]),
                   surrogate_conservative_life_h=float(10 ** worst_cons[i_opt]),
                   physics_worst_life_h=float(10 ** phys.min()),
                   physics_meets_target=bool(phys.min() >= log_target),
                   n_designs_screened=int(tt.size))
    print(json.dumps(summary, indent=2))
    with open(os.path.join(RES, "design_result.json"), "w") as f:
        json.dump(summary, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for t in [5.0, 6.0, 7.0, 8.0]:
        m, s = predict(gpr, np.full_like(T_GRID, t), T_GRID)
        ax.plot(T_GRID, m, label=f"t = {t:.0f} mm (GPR)")
        ax.fill_between(T_GRID, m - 2 * s, m + 2 * s, alpha=0.2)
        ax.plot(T_GRID[::3], [phys_life(t, T) for T in T_GRID[::3]], "k.", ms=5)
    ax.set_xlabel("temperature [°C]")
    ax.set_ylabel("log$_{10}$(fatigue life h)")
    ax.set_title("Thermal effect on vibration-fatigue life\n(lines: GPR ±2σ, dots: physics model)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(T_MM_GRID, worst_mean, label="worst-case over temperature (GPR mean)")
    ax.plot(T_MM_GRID, worst_cons, "--", label="conservative (mean − 2σ)")
    ax.axhline(log_target, color="r", lw=1, label=f"target {TARGET_LIFE_H:.0f} h")
    ax.axvline(t_opt, color="k", lw=1, ls=":")
    ax.plot([t_opt], [phys.min()], "ks", label=f"physics check at t = {t_opt:.2f} mm")
    ax.set_xlabel("bracket thickness t [mm]")
    ax.set_ylabel("log$_{10}$(worst-case life h)")
    ax.set_title("Lightest design meeting the life target")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "design_optimization.png"), dpi=150)


if __name__ == "__main__":
    main()
