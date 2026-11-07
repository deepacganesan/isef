"""Noise-corrected Kramers: average the drift over per-neuron fluctuations.
Mean-field drift along the memory: dm/dt = -m + a * E_z[tanh(m + z)],  z ~ N(0, s^2),
with s^2 = sigma^2 / 2 (stationary per-neuron spread). Potential U_eff = -integral of drift."""
import numpy as np

r = np.load("idp_results.npz")
sigma, D = float(r["sigma"]), float(r["D_true"])
s = sigma / np.sqrt(2)
gh_x, gh_w = np.polynomial.hermite_e.hermegauss(40)
gh_w = gh_w / gh_w.sum()


def f(m, a):                                   # effective drift
    return -m + a * (np.tanh(m[:, None] + s * gh_x) @ gh_w)


def rate_eff(a):
    m = np.linspace(0, 4, 40001)
    F = f(m, a)
    U = -np.concatenate([[0], np.cumsum((F[1:] + F[:-1]) / 2 * np.diff(m))])
    idx = np.where((F[:-1] > 0) & (F[1:] <= 0) & (m[:-1] > 0.05))[0]   # stable fixed point
    if idx.size == 0:
        return np.nan, 0.0                     # no well left: the memory is already gone
    i = idx[0]
    dU = U[0] - U[i]
    h = m[1] - m[0]
    k_well = (U[i + 1] - 2 * U[i] + U[i - 1]) / h ** 2
    k_top = abs(2 * (U[1] - U[0]) / h ** 2)          # symmetric potential: U''(0)
    return np.sqrt(k_well * k_top) / (2 * np.pi) * np.exp(-dU / D), m[i]


E_sech2 = (1 / np.cosh(s * gh_x) ** 2) @ gh_w
print(f"noise-shifted collapse point a_c = 1 / E[sech^2(z)] = {1 / E_sech2:.3f}  (deterministic: 1.000)")
rows = []
for a, r_sim, r_k, frac in r["kramers_val"]:
    r_e, m_e = rate_eff(a)
    rows.append((a, r_sim, r_k, r_e))
    print(f"a={a:.2f}  sim={r_sim:.3g}  Kramers(deterministic)={r_k:.3g}  Kramers(noise-corrected)={r_e:.3g}")
np.save("kramers_corrected.npy", np.array(rows))
