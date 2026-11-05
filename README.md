# ML-accelerated calibration and vibration-fatigue prediction

A personal follow-up to my master thesis (*Optimization of Modal Parameters for Headlamp Components in Linear FEA Simulations Under Vibration Load Cases*, FAU Erlangen / HELLA, 2025).

In the thesis I calibrated linear FE models of headlamp assemblies to shaker tests: modulus, Rayleigh damping and connector stiffness/damping were tuned in HEEDS until the simulated relative-displacement RMS (U_rms) matched 3D-camera measurements. Two of the future-work items were to **reduce the number of optimizer iterations** and to make connector and damping values **predictable without a new calibration every time**. This project explores how machine learning can help with both, and carries the calibrated model through to **vibration-fatigue life under thermal loads**.

> All models, materials and data here are synthetic and illustrative. No company data, geometry or results are used.

## What is in it

| Step | Script | Idea |
|---|---|---|
| 1 | `src/physics.py` | Compact FE model: polymer bracket (Euler–Bernoulli beam) + housing mass + module on a spring–dashpot connector, Rayleigh damping, temperature-dependent modulus. Direct complex frequency response under 10–200 Hz white noise → U_rms at 3 measurement points, root stress PSD → **Dirlik** rainflow PDF → **Basquin S–N + Miner** → fatigue life. |
| 2 | `src/calibrate.py` | Calibrate the model to a *virtual shaker test* (hidden true parameters + 2 % camera noise), staged like the thesis: stiffness → natural frequencies, then damping → U_rms. **Direct global search** (differential evolution, stand-in for a direct optimizer such as SHERPA) vs. **Bayesian optimization** (GPR surrogate + expected improvement). |
| 3 | `src/make_dataset.py` | 2,000-sample Latin-hypercube DoE with the calibrated model over geometry, module mass, damping, load level and temperature (−40 to 105 °C). |
| 4 | `src/train_surrogates.py` | Fatigue-life surrogates: **Gaussian process regression, random forest, XGBoost, neural network (MLP)**. Parity plots, learning curves, GPR uncertainty coverage. |
| 5 | `src/design_optimization.py` | Uncertainty-aware design: the lightest bracket that reaches the life target at the **worst-case temperature**, using GPR mean − 2σ, then verified with the physics model. |
| 6 | `src/pinn_modal_id.py` | **Physics-informed neural network** (JAX) that identifies natural frequency and damping ratio from a short, noisy free-decay record, compared with classical least squares. |

Run everything with `./run_all.sh` (about 6 minutes on a laptop CPU). Install with `pip install -r requirements.txt`.

## Results

All numbers below come from `./run_all.sh` (seeded, so they reproduce).

### 1. Calibration to a virtual shaker test — `results/calibration_convergence.png`

The "standard" model (nominal modulus and connector, about 4 % damping on mode 1) misses the natural frequencies by 7.7 % and the weighted U_rms by 12.7 %.

| | Direct search (differential evolution) | Bayesian optimization (GPR + EI) | True value |
|---|---|---|---|
| FE runs to reach 0.5 % frequency error | 63 | **21** | |
| FE runs to reach 2.5 % U_rms error | 8 | 8 | |
| Modulus factor | 1.135 | 1.150 | 1.15 |
| Connector stiffness k_c [N/m] | 60,681 | 59,932 | 60,000 |
| Rayleigh α [1/s] / connector damping c_c [N·s/m] | 13.8 / 3.8 | 2.0 / 6.5 | 17.3 / 3.0 |

Bayesian optimization needs about **3× fewer FE runs** for the stiffness stage and recovers the stiffness parameters more accurately. Both methods match U_rms to well below the 2 % measurement noise, but with **different damping splits**: from U_rms at three points, Rayleigh and connector damping are not separately identifiable.

### 2. Fatigue-life surrogates — `results/parity_plots.png`, `results/learning_curve.png`

Target: log10(fatigue life in hours), 2,000 samples, 20 % held out for testing.

| Model | Training samples | R² | RMSE [decades of life] | Within a factor of 2 |
|---|---|---|---|---|
| **GPR** (Matérn 5/2, ARD) | 1,000 | > 0.9999 | **0.007** (about 2 % in life) | 100 % |
| Neural network (MLP 3×64, tanh) | 1,600 | 0.9993 | 0.059 | 99.5 % |
| XGBoost | 1,600 | 0.988 | 0.253 | 80.8 % |
| Random forest | 1,600 | 0.951 | 0.505 | 44.8 % |

- GPR with only 250 simulations (RMSE 0.027) already beats XGBoost trained on 1,600.
- The GPR 95 % interval contains 99.3 % of the test points (slightly conservative).
- Log-transformed inputs cut the GPR error from 0.023 to 0.007 decades.
- One physics evaluation takes about 55 ms here; one GPR prediction about 0.02 ms.

### 3. Worst-case-temperature design — `results/design_optimization.png`

Case: 60 mm bracket, 150 g module, calibrated damping, 10–200 Hz at 1 g RMS, life target 2,000 h anywhere between −40 and 105 °C.

- 9,030 thickness–temperature combinations screened with the GPR in seconds.
- Governing temperature: **105 °C**. Lightest design: **t = 6.56 mm** (GPR mean − 2σ: 2,015 h).
- Physics-model check at 6.56 mm: **2,131 h** at the worst temperature, so the target is met.

### 4. PINN modal identification — `results/pinn_modal_id.png`

60 samples of a 50 ms free decay with 2 % noise; starting values from an uncalibrated model with the default 4 % damping.

| | Natural frequency [Hz] | Damping ratio |
|---|---|---|
| True | 64.00 | 0.0220 |
| Starting guess | 58.0 | 0.040 |
| PINN (JAX) | 64.03 | 0.0217 |
| Least squares | 63.98 | 0.0223 |

Both methods land within 0.05 % on frequency and 1.5 % on damping.

## What I learned

- **Physics first, then ML.** Working in log space (life ∝ PSD^(−k/2), Basquin) turns the power laws into nearly linear relations, which is why GPR and a small neural network fit almost perfectly, while tree models (random forest, XGBoost) show a staircase error because they cannot interpolate smoothly.
- **GPR is the right tool when every sample is an expensive FE run.** It is data-efficient and gives an uncertainty, which can be used directly for conservative design (mean − 2σ) and for choosing the next simulation in Bayesian optimization.
- **Damping is hard to identify uniquely.** With U_rms at only three points, Rayleigh damping and connector damping trade off against each other: both optimizers match U_rms within a fraction of a percent with different damping splits. This mirrors what I saw in the thesis and is why more measurement points or FRF-shape information are needed.
- **Hot is critical for this polymer.** The modulus drops when hot, which slightly lowers the stress, but the fatigue strength drops faster, so the worst case sits at the maximum temperature.
- **PINNs are useful for inverse problems.** Here a closed-form solution exists, so least squares is just as accurate; the PINN approach becomes interesting when the governing equations have no closed-form solution or the data are sparse.

## Limitations

- The structure is a 1D beam with lumped masses, not a 3D housing. In a real model each FE run takes minutes to hours, which is where the evaluation savings matter.
- Material, S–N and temperature values are illustrative, not test data. Mean stress, frequency effects on polymer fatigue and self-heating are ignored.
- The "experiment" is synthetic, so model-form error (a real structure that the model cannot represent exactly) is not included.

## Next steps

- Replace the beam with an Abaqus model driven from Python and repeat the Bayesian calibration.
- Multi-fidelity GPR (cheap beam model + few 3D runs) and operator learning (e.g. DeepONet) for full-field responses.
- Fit the S–N curve and its temperature dependence to coupon test data.
