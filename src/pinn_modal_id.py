"""
Physics-informed neural network (PINN) for modal-parameter identification.

Inverse problem: from a short, noisy free-decay measurement of one vibration mode,
identify its natural frequency f_n and damping ratio zeta. This mirrors test-based
model calibration (e.g. updating FE modal parameters from measured responses).

The network x(tau) is trained to (a) match the measurements and (b) satisfy the
equation of motion  x'' + 2*zeta*w*x' + w^2*x = 0, where w and zeta are trainable.
The result is compared with a classical nonlinear least-squares fit of the closed-form solution.
"""
import os
import json
import numpy as np
import jax
import jax.numpy as jnp
import optax
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

# Ground truth (unknown to the PINN) and measurement setup
FN_TRUE, ZETA_TRUE = 64.0, 0.022         # Hz, -  (first bending mode of a polymer housing-like part)
T_END = 0.05                              # s  (about 3 oscillation cycles)
N_MEAS, NOISE = 60, 0.02                  # samples, noise std relative to initial amplitude
FN_GUESS, ZETA_GUESS = 58.0, 0.04         # starting values: uncalibrated FE model with default 4 % damping


def free_decay(t, fn, zeta):
    w = 2 * np.pi * fn
    wd = w * np.sqrt(1 - zeta**2)
    return np.exp(-zeta * w * t) * (np.cos(wd * t) + zeta * w / wd * np.sin(wd * t))


def init_mlp(key, sizes):
    params = []
    for k, (m, n) in zip(jax.random.split(key, len(sizes) - 1), zip(sizes[:-1], sizes[1:])):
        params.append((jax.random.normal(k, (m, n)) * jnp.sqrt(1.0 / m), jnp.zeros(n)))
    return params


def mlp(params, tau):
    h = jnp.atleast_1d(tau)[None, :]
    for W, b in params[:-1]:
        h = jnp.tanh(h @ W + b)
    W, b = params[-1]
    return (h @ W + b)[0, 0]


def main(steps=10000, seed=0):
    rng = np.random.default_rng(seed)
    t_meas = np.linspace(0, T_END, N_MEAS)
    x_meas = free_decay(t_meas, FN_TRUE, ZETA_TRUE) + NOISE * rng.standard_normal(N_MEAS)
    tau_meas = jnp.array(t_meas / T_END)
    x_meas_j = jnp.array(x_meas)
    tau_col = jnp.linspace(0, 1, 300)                       # collocation points for the physics loss

    W0 = 2 * np.pi * FN_GUESS * T_END                       # dimensionless angular frequency guess
    params = {"net": init_mlp(jax.random.PRNGKey(seed), [1, 64, 64, 64, 1]),
              "log_w_scale": jnp.array(0.0),
              "zeta_raw": jnp.array(np.log(np.expm1(ZETA_GUESS)))}  # softplus^-1

    x = lambda p, tau: mlp(p["net"], tau)
    dx = jax.grad(x, argnums=1)
    ddx = jax.grad(dx, argnums=1)

    def loss_fn(p):
        W = W0 * jnp.exp(p["log_w_scale"])
        zeta = jax.nn.softplus(p["zeta_raw"])
        pred = jax.vmap(lambda t: x(p, t))(tau_meas)
        d1 = jax.vmap(lambda t: dx(p, t))(tau_col)
        d2 = jax.vmap(lambda t: ddx(p, t))(tau_col)
        xc = jax.vmap(lambda t: x(p, t))(tau_col)
        residual = d2 / W**2 + 2 * zeta * d1 / W + xc     # scaled equation of motion
        data = jnp.mean((pred - x_meas_j) ** 2)
        phys = jnp.mean(residual**2)
        ic = dx(p, 0.0) ** 2 / W**2                        # released from rest: x'(0) = 0
        return data + 0.1 * phys + 0.1 * ic, (data, phys)

    sched = optax.exponential_decay(2e-3, transition_steps=5000, decay_rate=0.6)
    opt = optax.adam(sched)
    state = opt.init(params)

    @jax.jit
    def step(p, s):
        (l, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(p)
        upd, s = opt.update(g, s, p)
        return optax.apply_updates(p, upd), s, l, aux

    hist = []
    for i in range(steps):
        params, state, l, (d, ph) = step(params, state)
        if i % 500 == 0 or i == steps - 1:
            fn_i = float(FN_GUESS * jnp.exp(params["log_w_scale"]))
            z_i = float(jax.nn.softplus(params["zeta_raw"]))
            hist.append((i, fn_i, z_i, float(d), float(ph)))
            if i % 5000 == 0:
                print(f"step {i:6d}  loss {float(l):.2e}  f_n {fn_i:6.2f} Hz  zeta {z_i:.4f}")

    fn_pinn, z_pinn = hist[-1][1], hist[-1][2]
    (fn_ls, z_ls), _ = curve_fit(free_decay, t_meas, x_meas, p0=[FN_GUESS, ZETA_GUESS])

    out = dict(true=dict(fn_Hz=FN_TRUE, zeta=ZETA_TRUE),
               initial_guess=dict(fn_Hz=FN_GUESS, zeta=ZETA_GUESS),
               pinn=dict(fn_Hz=fn_pinn, zeta=z_pinn,
                         fn_error_pct=100 * abs(fn_pinn - FN_TRUE) / FN_TRUE,
                         zeta_error_pct=100 * abs(z_pinn - ZETA_TRUE) / ZETA_TRUE),
               least_squares=dict(fn_Hz=float(fn_ls), zeta=float(z_ls),
                                  fn_error_pct=100 * abs(fn_ls - FN_TRUE) / FN_TRUE,
                                  zeta_error_pct=100 * abs(z_ls - ZETA_TRUE) / ZETA_TRUE),
               measurement=dict(n_samples=N_MEAS, noise_rel=NOISE, duration_s=T_END))
    print(json.dumps(out, indent=2))
    with open(os.path.join(RES, "pinn_result.json"), "w") as f:
        json.dump(out, f, indent=2)

    # Figure: fit and parameter convergence
    tt = np.linspace(0, T_END, 600)
    x_pinn = np.array(jax.vmap(lambda t: x(params, t))(jnp.array(tt / T_END)))
    h = np.array(hist)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(t_meas * 1e3, x_meas, "k.", label="noisy measurement")
    axes[0].plot(tt * 1e3, free_decay(tt, FN_TRUE, ZETA_TRUE), "g-", lw=1, label="true response")
    axes[0].plot(tt * 1e3, x_pinn, "r--", label="PINN")
    axes[0].set_xlabel("time [ms]"); axes[0].set_ylabel("displacement [-]"); axes[0].legend(fontsize=8)
    axes[0].set_title("Free-decay response")
    axes[1].plot(h[:, 0], h[:, 1]); axes[1].axhline(FN_TRUE, color="g", ls="--")
    axes[1].set_xlabel("training step"); axes[1].set_ylabel("f$_n$ [Hz]"); axes[1].set_title("Identified natural frequency")
    axes[2].plot(h[:, 0], h[:, 2]); axes[2].axhline(ZETA_TRUE, color="g", ls="--")
    axes[2].set_xlabel("training step"); axes[2].set_ylabel("ζ [-]"); axes[2].set_title("Identified damping ratio")
    for a in axes:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "pinn_modal_id.png"), dpi=150)


if __name__ == "__main__":
    main()
