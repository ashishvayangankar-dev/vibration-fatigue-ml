"""
Train and compare ML surrogates for vibration-fatigue life:
Gaussian process regression (GPR), random forest, XGBoost and a neural network (MLP).

Target: log10(fatigue life in hours). Working in log space turns the power laws of
random-vibration fatigue (life ~ PSD^(-k/2), Basquin S-N) into near-linear relations.
"""
import os
import json
import time
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor


warnings.filterwarnings("ignore")  # sklearn convergence chatter

GPR_MAX_TRAIN = 1000   # exact GPR scales as O(n^3); it already saturates well below this size
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "dataset.csv")
RES = os.path.join(HERE, "..", "results")
from make_dataset import INPUTS, CALIBRATED, run_design
TARGET = "log10_life_h"


def features(df):
    """Physics-informed features: log of the positive inputs (power laws -> linear), T kept linear."""
    X = df[INPUTS].copy()
    for c in ["L_mm", "t_mm", "m_mod_g", "alpha", "grms"]:
        X[c] = np.log10(X[c])
    return X.values


def make_models():
    kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(len(INPUTS)), nu=2.5) + WhiteKernel(1e-4)
    return {
        "GPR": make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel, normalize_y=True,
                                                                        n_restarts_optimizer=0, random_state=0)),
        "Random forest": RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=0),
        "XGBoost": XGBRegressor(n_estimators=1500, max_depth=5, learning_rate=0.03, subsample=0.8,
                                colsample_bytree=0.9, random_state=0),
        "Neural network (MLP)": make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64, 64),
                                                                            activation="tanh", max_iter=4000,
                                                                            learning_rate_init=1e-3, alpha=1e-5,
                                                                            early_stopping=True, n_iter_no_change=100,
                                                                            random_state=0)),
    }


def metrics(y, p):
    err = p - y
    return dict(R2=float(r2_score(y, p)),
                RMSE_decades=float(np.sqrt(mean_squared_error(y, p))),
                within_factor_2=float(np.mean(np.abs(err) < np.log10(2.0))))


def main():
    os.makedirs(RES, exist_ok=True)
    df = pd.read_csv(DATA)
    X, y = features(df), df[TARGET].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)

    results, preds = {}, {}
    for name, model in make_models().items():
        n_fit = GPR_MAX_TRAIN if name == "GPR" else len(Xtr)
        t0 = time.time()
        model.fit(Xtr[:n_fit], ytr[:n_fit])
        fit_s = time.time() - t0
        t0 = time.time()
        p = model.predict(Xte)
        pred_ms = 1e3 * (time.time() - t0) / len(Xte)
        results[name] = {**metrics(yte, p), "n_train": n_fit, "fit_time_s": round(fit_s, 2), "predict_ms_per_design": round(pred_ms, 4)}
        preds[name] = p
        print(f"{name:22s} R2={results[name]['R2']:.4f}  RMSE={results[name]['RMSE_decades']:.3f} decades  "
              f"within x2: {100 * results[name]['within_factor_2']:.1f}%  fit {fit_s:.1f}s")
        if name == "GPR":
            mu, sd = model.predict(Xte, return_std=True)
            results[name]["coverage_95"] = float(np.mean(np.abs(yte - mu) < 1.96 * sd))
            joblib.dump(model, os.path.join(RES, "gpr_life_model.joblib"))

    # Cost of the physics model for comparison
    t0 = time.time()
    for row in df[INPUTS].values[:100]:
        run_design(row)
    results["physics_model_ms_per_design"] = round(1e3 * (time.time() - t0) / 100, 3)

    # Ablation: raw vs log (physics-informed) features for GPR
    Xraw = df[INPUTS].values
    Xr_tr, Xr_te, _, _ = train_test_split(Xraw, y, test_size=0.2, random_state=0)
    gpr_raw = make_models()["GPR"].fit(Xr_tr[:GPR_MAX_TRAIN], ytr[:GPR_MAX_TRAIN])
    results["ablation_GPR_raw_features"] = metrics(yte, gpr_raw.predict(Xr_te))
    print("GPR raw features:", results["ablation_GPR_raw_features"])

    # Learning curve: how many simulations does each model need?
    sizes = [50, 100, 250, 500, 1000, 2000]
    lc = {"sizes": sizes}
    for name in ["GPR", "XGBoost", "Neural network (MLP)"]:
        scores = []
        for n in sizes:
            if name == "GPR" and n > GPR_MAX_TRAIN:
                scores.append(float("nan"))
                continue
            m = make_models()[name]
            m.fit(Xtr[:n], ytr[:n])
            scores.append(float(np.sqrt(mean_squared_error(yte, m.predict(Xte)))))
        lc[name] = scores
        print(name, "learning curve RMSE:", np.round(scores, 3))
    results["learning_curve_RMSE_decades"] = lc

    with open(os.path.join(RES, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Parity plots
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharex=True, sharey=True)
    lo, hi = y.min() - 0.3, y.max() + 0.3
    for ax, (name, p) in zip(axes, preds.items()):
        ax.scatter(yte, p, s=6, alpha=0.5)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_title(f"{name}\nR$^2$={results[name]['R2']:.3f}, RMSE={results[name]['RMSE_decades']:.2f} dec")
        ax.set_xlabel("physics model log$_{10}$(life h)")
    axes[0].set_ylabel("surrogate log$_{10}$(life h)")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "parity_plots.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for name in ["GPR", "XGBoost", "Neural network (MLP)"]:
        ax.loglog(sizes, lc[name], "o-", label=name)
    ax.set_xlabel("number of training simulations")
    ax.set_ylabel("test RMSE [decades of life]")
    ax.set_title("Data efficiency of the surrogates")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "learning_curve.png"), dpi=150)
    print("saved metrics and figures to", os.path.normpath(RES))


if __name__ == "__main__":
    main()
