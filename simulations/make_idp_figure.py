"""Figure + summary numbers for Experiment 2 (IDP). Run after idp_ews.py."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

r = np.load("idp_results.npz")
c, alpha0 = r["c"], r["alpha0"]
S = r["a"].shape[0]
INK, INK2, SURF, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"


def flat(k):
    return r[k].reshape(-1)


a = flat("a")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2})
fig, axes = plt.subplots(2, 3, figsize=(14, 8.2), facecolor=SURF)
for ax in axes.flat:
    ax.set_facecolor(SURF)
    ax.grid(True, color=GRID, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def collapse_line(ax):
    ax.axvline(1.0, color=ORANGE, lw=1.5, ls="--")


# A: patient-level decline vs c
ax = axes[0, 0]
rec = r["recall10"].mean(0)                                # (C, K) averaged over networks
for k in range(len(alpha0)):
    ax.plot(c, rec[:, k], color=BLUE, lw=1, alpha=0.35)
ax.plot(c, rec.mean(1), color=BLUE, lw=2.4, label="all memories")
ax.invert_xaxis()
ax.set_xlabel("c  (fraction of network still supporting memory)")
ax.set_title("A  Visible symptom: share of memories recalled\nfrom a 10%-corrupted cue (faint = each memory)",
             loc="left", fontsize=10.5, color=INK)

# B: calibration curve = basin radius vs effective strength
ax = axes[0, 1]
ok = np.isfinite(flat("R"))
ax.scatter(a[ok], flat("R")[ok], s=12, color=BLUE, alpha=0.5, lw=0, label="measured")
aa = np.linspace(1.0, 3.6, 200)
ax.plot(aa, 0.5 * (1 - 1 / aa), color=INK, lw=1.6, label=r"theory $\frac{1}{2}(1-1/a)$")
collapse_line(ax)
ax.set_xlabel(r"effective memory strength $a = c\,\beta_\mu$")
ax.set_title("B  Basin radius R (the calibration curve)\nsmooth and invertible all the way down",
             loc="left", fontsize=10.5, color=INK)
ax.legend(frameon=False, fontsize=9, loc="lower right")

# C: barrier height, two candidate energies
ax = axes[0, 2]
ok = np.isfinite(flat("dU"))
ax.scatter(a[ok], flat("dU")[ok], s=12, color=ORANGE, alpha=0.6, lw=0, label="Hopfield energy E(0) − E(x*)")
ok2 = np.isfinite(flat("dU_1d"))
ax.scatter(a[ok2], flat("dU_1d")[ok2], s=12, color=BLUE, alpha=0.6, lw=0, label="drift potential U(0) − U(m*)")
collapse_line(ax)
ax.set_yscale("log")
ax.set_ylim(1e-4, 10)
ax.set_xlabel(r"effective memory strength $a$")
ax.set_title("C  Barrier height ΔV per neuron\n(only the drift potential belongs in Kramers)",
             loc="left", fontsize=10.5, color=INK)
ax.legend(frameon=False, fontsize=8.5, loc="lower right")

# D: Kramers validation
ax = axes[1, 0]
val = r["kramers_val"]                       # a, r_sim, r_kramers, frac escaped
ax.plot(val[:, 0], val[:, 1], color=BLUE, lw=2.2, marker="s", ms=6, label="simulated escapes")
ax.plot(val[:, 0], val[:, 2], color=INK2, lw=1.4, marker="o", ms=4, ls=":",
        label="Kramers, deterministic landscape")
try:
    kc = np.load("kramers_corrected.npy")
    okc = np.isfinite(kc[:, 3])
    ax.plot(kc[okc, 0], kc[okc, 3], color=ORANGE, lw=1.8, marker="o", ms=5,
            label="Kramers, noise-corrected landscape")
    ax.axvline(1.113, color=ORANGE, lw=1, ls="--")
    ax.text(1.115, 1e-18, "noise-shifted\ncollapse point", color=ORANGE, fontsize=8.5)
except FileNotFoundError:
    pass
ax.set_yscale("log")
ax.set_ylim(1e-23, 1)
ax.set_xlabel(r"effective memory strength $a$")
ax.set_ylabel("escape rate (per model time unit)")
ax.set_title("D  Does Kramers predict actual memory loss?\n(64–128 noisy runs per point)",
             loc="left", fontsize=10.5, color=INK)
ax.legend(frameon=False, fontsize=8.5, loc="lower left")

# E: critical slowing down, recovery time after a pulse
ax = axes[1, 1]
aa = np.linspace(1.005, 3.6, 300)
mm = np.array([0.0] * len(aa))
for i, ai in enumerate(aa):
    m = ai
    for _ in range(500):
        m = ai * np.tanh(m)
    mm[i] = m
ax.plot(aa, 1 / (1 - aa / np.cosh(mm) ** 2), color=INK, lw=1.6, label="theory")
for key, col, lab in [("tau_det_m", AQUA, "noise-free, memory overlap"),
                      ("tau_al_m", BLUE, "8 noisy pulses, memory overlap"),
                      ("tau_e", ORANGE, "8 noisy pulses, virtual EEG")]:
    y = flat(key)
    ok = np.isfinite(y)
    ax.scatter(a[ok], y[ok], s=12, color=col, alpha=0.6, lw=0, label=lab)
collapse_line(ax)
ax.set_yscale("log")
ax.set_ylim(0.2, 300)
ax.set_xlabel(r"effective memory strength $a$")
ax.set_title("E  Recovery time τ after a photic-style pulse\n(critical slowing down)",
             loc="left", fontsize=10.5, color=INK)
ax.legend(frameon=False, fontsize=8.5, loc="upper right")

# F: do the two halves agree?
ax = axes[1, 2]
geo = flat("R")
rows = []
for key, col, lab in [("tau_al_m", BLUE, "memory overlap"), ("tau_e", ORANGE, "virtual EEG")]:
    y = flat(key)
    ok = np.isfinite(y) & np.isfinite(geo)
    rho, p = spearmanr(geo[ok], y[ok])
    rows.append((lab, rho, p, ok.sum()))
    ax.scatter(geo[ok], y[ok], s=12, color=col, alpha=0.6, lw=0,
               label=f"{lab}: Spearman ρ = {rho:.2f}")
ax.set_yscale("log")
ax.set_xlabel("basin radius R  (geometry half)")
ax.set_ylabel("recovery time τ  (dynamics half)")
ax.set_title("F  Do the two halves of the plan agree?",
             loc="left", fontsize=10.5, color=INK)
ax.legend(frameon=False, fontsize=9)

fig.suptitle(f"Experiment 2: IDP network under progressive damage  (N={int(r['N'])}, "
             f"{len(alpha0)} memories, {S} networks, noise σ={float(r['sigma'])})",
             x=0.01, ha="left", color=INK, fontsize=12.5)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig("fig2_idp_two_halves.png", dpi=170, facecolor=SURF)

# ------------------------------------------------------------- summary numbers
print("Agreement (geometry R vs dynamics tau):")
for lab, rho, p, n in rows:
    print(f"  {lab:<16} rho={rho:+.2f}  p={p:.2g}  n={n}")
okR = np.isfinite(flat("R"))
err = flat("R")[okR] - 0.5 * (1 - 1 / a[okR])
print(f"Basin radius vs IDP theory: mean |error| = {np.mean(np.abs(err)):.3f}")
ok = np.isfinite(flat("tau_det_m")) & (a > 1.05)
rel = flat("tau_det_m")[ok] / flat("tau_theory")[ok]
print(f"Noise-free tau / theory (a>1.05): median {np.median(rel):.2f}")
okD = np.isfinite(flat("D_hat"))
print(f"Plan's noise estimator D_hat: median {np.median(flat('D_hat')[okD]):.2e}  vs true {float(r['D_true']):.2e}")
for k in ["tau_e", "tau_al_m", "tau_al_e"]:
    y = flat(k)
    alive = np.isfinite(flat("m_det"))
    print(f"{k:<9} usable fits: {np.isfinite(y).sum()}/{alive.sum()}")
P = flat("P_true")
okP = np.isfinite(P)
lo = a[okP & (P > 0.05)].max() if np.any(okP & (P > 0.05)) else np.nan
print(f"P(collapse within horizon) > 5% only for a <= {lo:.3f}")
val = r["kramers_val"]
print("Kramers check (a, sim rate, Kramers rate, ratio):")
for row in val:
    print(f"  a={row[0]:.2f}  sim={row[1]:.3g}  K={row[2]:.3g}  ratio={row[1] / row[2]:.2f}  escaped {row[3]:.0%}")
print("saved fig2_idp_two_halves.png")
