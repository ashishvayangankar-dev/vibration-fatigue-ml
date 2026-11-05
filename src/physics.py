"""
Physics model: a polymer mounting bracket carrying a module through a connector,
under random base excitation (shaker / vehicle), with temperature-dependent material.

It mirrors the structure of a headlamp vibration-correlation workflow in a compact form:
  - bracket: Euler-Bernoulli beam FE (Hermite elements), clamped at the screw boss
  - housing section: lumped mass at the bracket tip
  - module: lumped mass attached to the housing by a connector (spring k_c + dashpot c_c)
  - damping: Rayleigh damping C = alpha*M + beta*K on the bracket + connector dashpot
  - load: band-limited white-noise base acceleration, 10-200 Hz (given as g RMS)
  - outputs: natural frequencies, relative-displacement RMS (U_rms) at measurement points,
             root bending stress PSD -> Dirlik rainflow PDF -> Basquin S-N + Miner -> fatigue life

Because the connector dashpot makes the damping non-proportional, the frequency response
is computed by a direct (complex) solve at every frequency instead of modal superposition.

All material and connector values are ILLUSTRATIVE (loosely glass-fibre polypropylene-like).
No company data is used.
"""
import numpy as np
from scipy.linalg import eigh
from scipy.special import gamma as gamma_fn

G0 = 9.81                      # m/s^2
RHO = 1120.0                   # kg/m^3
WIDTH = 0.030                  # m
M_HOUSING = 0.10               # kg, housing section carried at the bracket tip
BETA = 1e-8                    # s, stiffness-proportional Rayleigh term (kept tiny)
K_SN = 9.0                     # Basquin exponent: N = N_REF * (S_f / S_a)^K_SN
N_REF = 1e7
F_LO, F_HI = 10.0, 200.0       # Hz, excitation band
FREQ = np.arange(F_LO, F_HI + 1e-9, 0.05)
N_EL = 10

# Nominal ("standard model") connector values
K_C_NOM, C_C_NOM = 5.0e4, 5.0  # N/m, N s/m


def youngs_modulus(T, scale=1.0):
    """Young's modulus [Pa] vs temperature [degC]: polypropylene softens strongly when hot."""
    return scale * 6.5e9 * (1.0 - 0.0055 * (T - 23.0))


def fatigue_strength(T):
    """Stress amplitude [Pa] for N_REF cycles; degrades with temperature."""
    return 18e6 * (1.0 - 0.004 * (T - 23.0))


def assemble(L, t, m_mod, T, alpha, k_c, c_c, e_scale=1.0, n_el=N_EL):
    """Return M, K, C (reduced, clamp removed) and helpers. Last DOF = module translation."""
    E = youngs_modulus(T, e_scale)
    I = WIDTH * t**3 / 12.0
    A = WIDTH * t
    le = L / n_el
    ke = E * I / le**3 * np.array([[12, 6 * le, -12, 6 * le], [6 * le, 4 * le**2, -6 * le, 2 * le**2],
                                   [-12, -6 * le, 12, -6 * le], [6 * le, 2 * le**2, -6 * le, 4 * le**2]])
    me = RHO * A * le / 420.0 * np.array([[156, 22 * le, 54, -13 * le], [22 * le, 4 * le**2, 13 * le, -3 * le**2],
                                          [54, 13 * le, 156, -22 * le], [-13 * le, -3 * le**2, -22 * le, 4 * le**2]])
    nb = 2 * (n_el + 1)
    Kb = np.zeros((nb, nb))
    Mb = np.zeros((nb, nb))
    for e in range(n_el):
        s = slice(2 * e, 2 * e + 4)
        Kb[s, s] += ke
        Mb[s, s] += me
    Kb, Mb = Kb[2:, 2:], Mb[2:, 2:]                 # clamp at the screw boss
    Mb[-2, -2] += M_HOUSING                          # housing section is part of the damped polymer structure
    n = Kb.shape[0] + 1                              # + module DOF
    M = np.zeros((n, n)); K = np.zeros((n, n)); C = np.zeros((n, n))
    M[:-1, :-1], K[:-1, :-1] = Mb, Kb
    C[:-1, :-1] = alpha * Mb + BETA * Kb             # Rayleigh damping of the polymer bracket
    M[-1, -1] = m_mod
    tip, mod = n - 3, n - 1                          # tip translational DOF, module DOF
    for mat, val in ((K, k_c), (C, c_c)):            # connector between bracket tip and module
        mat[tip, tip] += val; mat[mod, mod] += val
        mat[tip, mod] -= val; mat[mod, tip] -= val
    r = np.zeros(n); r[0:-1:2] = 1.0; r[-1] = 1.0  # rigid-body translation (influence vector)
    mid = 2 * (n_el // 2) - 2                        # translational DOF of the mid node
    return dict(M=M, K=K, C=C, r=r, E=E, le=le, dofs=dict(mid=mid, tip=tip, module=mod))


def frf(sys):
    """Relative displacement per unit base acceleration at every frequency: shape (n_freq, n_dof)."""
    w = 2 * np.pi * FREQ
    A = sys["K"][None] - (w**2)[:, None, None] * sys["M"][None] + 1j * w[:, None, None] * sys["C"][None]
    rhs = -(sys["M"] @ sys["r"])
    return np.linalg.solve(A, np.broadcast_to(rhs, (len(w), len(rhs)))[..., None])[..., 0]


def analyse(L, t, m_mod, T, alpha, grms, k_c=K_C_NOM, c_c=C_C_NOM, e_scale=1.0, fatigue=True):
    """
    L, t [m]; m_mod [kg]; T [degC]; alpha [1/s]; grms [g RMS of 10-200 Hz white noise];
    k_c [N/m]; c_c [N s/m]; e_scale: modulus calibration factor.
    """
    sys = assemble(L, t, m_mod, T, alpha, k_c, c_c, e_scale)
    w2 = eigh(sys["K"], sys["M"], eigvals_only=True, subset_by_index=[0, 1])
    fn = np.sqrt(w2) / (2 * np.pi)

    psd = (grms * G0) ** 2 / (F_HI - F_LO)          # flat acceleration PSD, (m/s^2)^2/Hz
    U = frf(sys)
    d = sys["dofs"]
    urms = {k: float(np.sqrt(np.trapezoid(np.abs(U[:, i]) ** 2 * psd, FREQ))) * 1e3 for k, i in d.items()}  # mm

    out = dict(f1=fn[0], f2=fn[1], urms_mid_mm=urms["mid"], urms_tip_mm=urms["tip"], urms_module_mm=urms["module"])
    if fatigue:
        le, E = sys["le"], sys["E"]
        curv = 6.0 / le**2 * U[:, 0] - 2.0 / le * U[:, 1]   # root curvature (w1 = theta1 = 0 at the clamp)
        S = np.abs(E * (t / 2) * curv) ** 2 * psd / 1e12     # stress PSD, MPa^2/Hz
        m0, m1, m2, m4 = (np.trapezoid(FREQ**k * S, FREQ) for k in (0, 1, 2, 4))
        out.update(sigma_rms=np.sqrt(m0),
                   life_h=1.0 / dirlik_damage_rate(m0, m1, m2, m4, T) / 3600.0,
                   life_nb_h=1.0 / narrowband_damage_rate(m0, m2, T) / 3600.0)
    return out


def _cycles(S_a, T):
    Sf = fatigue_strength(T) / 1e6
    return N_REF * (Sf / np.maximum(S_a, 1e-12)) ** K_SN


def dirlik_damage_rate(m0, m1, m2, m4, T):
    """Dirlik (1985) empirical rainflow-range PDF + Palmgren-Miner -> damage per second."""
    xm = (m1 / m0) * np.sqrt(m2 / m4)
    g = m2 / np.sqrt(m0 * m4)                        # irregularity factor
    D1 = 2 * (xm - g**2) / (1 + g**2)
    R = (g - xm - D1**2) / (1 - g - D1 + D1**2)
    D2 = (1 - g - D1 + D1**2) / (1 - R)
    D3 = 1 - D1 - D2
    Q = 1.25 * (g - D3 - D2 * R) / D1
    Z = np.linspace(1e-6, 40.0, 20000)
    pZ = (D1 / Q) * np.exp(-Z / Q) + (D2 * Z / R**2) * np.exp(-Z**2 / (2 * R**2)) + D3 * Z * np.exp(-Z**2 / 2)
    S_range = 2 * np.sqrt(m0) * Z
    return np.sqrt(m4 / m2) * np.trapezoid(pZ / _cycles(S_range / 2.0, T), Z)


def narrowband_damage_rate(m0, m2, T):
    """Narrow-band (Rayleigh peaks) closed form, a conservative check on Dirlik."""
    Sf = fatigue_strength(T) / 1e6
    return np.sqrt(m2 / m0) * np.sqrt(2 * m0) ** K_SN * gamma_fn(1 + K_SN / 2) / (N_REF * Sf**K_SN)
