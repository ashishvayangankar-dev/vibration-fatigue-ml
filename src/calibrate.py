"""
Test-based model calibration: direct global search vs. Bayesian optimization (GPR surrogate).

A "virtual experiment" is generated from hidden true parameters, and the FE model is then
calibrated to it, as in a shaker test + camera correlation study. The calibration is staged
the way it is done in practice:
  stage 1  stiffness : modulus factor + connector stiffness k_c  -> match natural frequencies f1, f2
  stage 2  damping   : Rayleigh alpha + connector damping c_c    -> match U_rms at 3 measurement points
                       (10-200 Hz white noise at 1 g RMS; camera noise about 2 %)
U_rms errors are weighted by U_exp / max(U_exp), so the points that move most count most.

Method A, direct search: differential evolution evaluates the FE model for every candidate.
Method B, Bayesian optimization: a Gaussian-process surrogate of the error is refitted after
every FE run, and the next run is chosen by expected improvement (EI).
The comparison counts FE evaluations, which dominate the cost for a real 3D model.
"""
import os
import json
import warnings
import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import norm, qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from physics import analyse, K_C_NOM, C_C_NOM

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

GEOM = dict(L=0.060, t=0.006, m_mod=0.150, T=23.0, grms=1.0)
TRUE = dict(e_scale=1.15, alpha=17.3, k_c=6.0e4, c_c=3.0)              # hidden "reality"
URMS_KEYS = ["urms_mid_mm", "urms_tip_mm", "urms_module_mm"]
# Search ranges in a unit square per stage
S1_LO, S1_HI = np.array([0.7, 4.0]), np.array([1.5, 5.5])             # e_scale, log10 k_c
S2_LO, S2_HI = np.array([2.0, 0.5]), np.array([40.0, 15.0])           # alpha, c_c
TARGET = dict(stage1=0.005, stage2=0.025)                              # 0.5 % freq, 2.5 % U_rms


def simulate(prm):
    return analyse(GEOM["L"], GEOM["t"], GEOM["m_mod"], GEOM["T"], prm["alpha"], GEOM["grms"],
                   k_c=prm["k_c"], c_c=prm["c_c"], e_scale=prm["e_scale"], fatigue=False)


def make_experiment(seed=1):
    rng = np.random.default_rng(seed)
    r = simulate(TRUE)
    exp = {k: r[k] * (1 + 0.02 * rng.standard_normal()) for k in URMS_KEYS}       # camera noise
    exp.update({k: r[k] * (1 + 0.002 * rng.standard_normal()) for k in ("f1", "f2")})
    return exp


def freq_error(r, exp):
    return float(np.sqrt(np.mean([((r[k] - exp[k]) / exp[k]) ** 2 for k in ("f1", "f2")])))


def urms_error(r, exp):
    u_exp = np.array([exp[k] for k in URMS_KEYS])
    u_fe = np.array([r[k] for k in URMS_KEYS])
    wt = u_exp / u_exp.max()
    return float(np.sqrt(np.mean(((u_exp - u_fe) / u_exp * wt) ** 2)))


def stage_objectives(exp):
    def s1_params(u):
        p = S1_LO + np.asarray(u) * (S1_HI - S1_LO)
        return dict(e_scale=p[0], k_c=10 ** p[1], alpha=TRUE["alpha"], c_c=TRUE["c_c"])  # damping irrelevant for f

    def s1(u):
        return freq_error(simulate(s1_params(u)), exp)

    def make_s2(stiff):
        def s2_params(u):
            p = S2_LO + np.asarray(u) * (S2_HI - S2_LO)
            return dict(stiff, alpha=p[0], c_c=p[1])

        def s2(u):
            return urms_error(simulate(s2_params(u)), exp)
        return s2, s2_params
    return s1, s1_params, make_s2


def direct_search(f, max_evals=600, seed=0):
    hist, xs = [], []

    def g(u):
        hist.append(f(u)); xs.append(np.array(u))
        return hist[-1]
    pop = 15
    differential_evolution(g, [(0, 1)] * 2, popsize=pop, maxiter=max_evals // (2 * pop) - 1, seed=seed,
                           tol=0, polish=False, init="latinhypercube")
    return np.array(hist), np.array(xs)


def bayes_opt(f, n_init=6, n_iter=24, seed=0):
    rng = np.random.default_rng(seed)
    X = qmc.LatinHypercube(d=2, seed=seed).random(n_init)
    y = np.array([np.log10(f(x)) for x in X])
    kernel = ConstantKernel(1.0) * Matern(length_scale=[0.3, 0.3], nu=2.5) + WhiteKernel(1e-4)
    for _ in range(n_iter):
        gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=3,
                                      random_state=int(rng.integers(1_000_000))).fit(X, y)
        best = X[np.argmin(y)]
        cand = np.vstack([rng.random((20000, 2)),
                          np.clip(best + 0.05 * rng.standard_normal((4000, 2)), 0, 1),
                          np.clip(best + 0.01 * rng.standard_normal((2000, 2)), 0, 1)])
        mu, sd = gp.predict(cand, return_std=True)
        imp = y.min() - mu
        z = imp / np.maximum(sd, 1e-9)
        ei = imp * norm.cdf(z) + sd * norm.pdf(z)           # expected improvement
        x_next = cand[np.argmax(ei)]
        X = np.vstack([X, x_next])
        y = np.append(y, np.log10(f(x_next)))
    return 10 ** y, X


def evals_to(hist, target):
    hit = np.nonzero(np.minimum.accumulate(hist) <= target)[0]
    return int(hit[0] + 1) if len(hit) else None


def main():
    os.makedirs(RES, exist_ok=True)
    exp = make_experiment()
    s1, s1_params, make_s2 = stage_objectives(exp)

    # "Standard" model: nominal modulus and connector, ~4 % modal damping on mode 1
    f1_nom = simulate(dict(e_scale=1.0, alpha=1.0, k_c=K_C_NOM, c_c=C_C_NOM))["f1"]
    standard = dict(e_scale=1.0, alpha=2 * 0.04 * 2 * np.pi * f1_nom, k_c=K_C_NOM, c_c=C_C_NOM)
    r_std = simulate(standard)

    out = dict(true_parameters=TRUE, experiment={k: float(v) for k, v in exp.items()},
               standard_model=dict(freq_error=freq_error(r_std, exp), urms_error=urms_error(r_std, exp)))
    curves = {}
    for name, opt in (("direct_search", direct_search), ("bayesian_optimization", bayes_opt)):
        h1, X1 = opt(s1)
        stiff = s1_params(X1[np.argmin(h1)])
        stiff = dict(e_scale=stiff["e_scale"], k_c=stiff["k_c"])
        s2, s2_params = make_s2(stiff)
        h2, X2 = opt(s2)
        ident = s2_params(X2[np.argmin(h2)])
        curves[name] = (h1, h2)
        out[name] = dict(stage1_evals=len(h1), stage1_best_error=float(h1.min()),
                         stage1_evals_to_target=evals_to(h1, TARGET["stage1"]),
                         stage2_evals=len(h2), stage2_best_error=float(h2.min()),
                         stage2_evals_to_target=evals_to(h2, TARGET["stage2"]),
                         identified={k: float(v) for k, v in ident.items()})
    print(json.dumps(out, indent=2))
    with open(os.path.join(RES, "calibration_result.json"), "w") as f:
        json.dump(out, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    labels = {"direct_search": "direct search (differential evolution)",
              "bayesian_optimization": "Bayesian optimization (GPR + EI)"}
    titles = ["Stage 1: stiffness -> natural frequencies", "Stage 2: damping -> U$_{rms}$ at 3 points"]
    std_err = [out["standard_model"]["freq_error"], out["standard_model"]["urms_error"]]
    for i, ax in enumerate(axes):
        for name, (h1, h2) in curves.items():
            h = (h1, h2)[i]
            ax.semilogy(np.arange(1, len(h) + 1), 100 * np.minimum.accumulate(h), label=labels[name])
        ax.axhline(100 * std_err[i], color="gray", ls=":", label=f"standard model ({100 * std_err[i]:.0f} %)")
        ax.axhline(100 * list(TARGET.values())[i], color="r", lw=1, label="target")
        ax.set_xscale("log")
        ax.set_xlabel("number of FE evaluations")
        ax.set_ylabel("best error [%]")
        ax.set_title(titles[i])
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "calibration_convergence.png"), dpi=150)


if __name__ == "__main__":
    main()
