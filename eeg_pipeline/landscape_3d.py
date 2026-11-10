"""3-D energy landscape of the IDP network for one moment of each person.

The network has 380 neurons, so its landscape is 380-dimensional. We look at a 2-D slice
through it spanned by two healthy memories (default: the healthy 'alpha' memory and the
healthy 'slow' memory, picked by content), and plot the network's energy over that slice:

    E(x) = -1/2 v^T W(u) v + sum_i G(v_i),   v = tanh(x),   G(v) = integral of artanh
    W(u) = sum_mu alpha_mu(u) xi_mu xi_mu^T,  alpha_mu = gamma * max(0, xi_mu . u_hat)

Valleys = memories that exist in that moment. The flat bowl at the centre = confusion.
Each memory makes a pair of mirror valleys (x and -x), a standard Hopfield symmetry.

Moment chosen the same way for everyone: the window at the 80th percentile of
alpha_alphaMemory + alpha_slowMemory (one of that person's clearer moments).

Usage: python landscape_3d.py --binary binary --weights weights --annotation annotation.json --out landscape_3d.png
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from idp_weights import unit_inputs, saliency_cos

INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"


def band_levels(p, n_ch):
    return p.reshape(n_ch, 4, 5).mean((0, 2))                 # delta, theta, alpha, beta


def energy_grid(P, alpha, e1, e2, grid, scale):
    """Network energy per neuron at x = scale*(a e1 + b e2) for every (a, b) in the grid."""
    a, b = np.meshgrid(grid, grid)
    X = scale * (a[..., None] * e1 + b[..., None] * e2)       # (g, g, N)
    V = np.tanh(X)
    proj = V @ P.T                                            # (g, g, K)  xi_mu . v
    quad = (proj ** 2 * alpha).sum(-1)                        # v^T W v without the diagonal fix
    diag = ((P ** 2) * alpha[:, None]).sum(0)                 # remove self-connections
    quad -= (V ** 2 * diag).sum(-1)
    G = (X * V - np.logaddexp(X, -X) + np.log(2)).sum(-1)
    return a, b, (-.5 * quad + G) / X.shape[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="binary")
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--annotation", required=True)
    ap.add_argument("--show", nargs=3, default=["01086", "00722", "01167"])
    ap.add_argument("--gamma", type=float, default=None, help="default: memory exists in half of healthy windows")
    ap.add_argument("--out", default="landscape_3d.png")
    a = ap.parse_args()
    ann = {d["serial"]: d for d in json.load(open(a.annotation))["data"]}
    H = np.load(Path(a.weights) / "W_HC.npz")
    P, m = H["prototypes"], H["mean_vector"]; K, N = P.shape
    n_ch = N // 20
    load = lambda k: np.load(Path(a.binary) / f"{k}.npz")["codes"]
    if a.gamma is None:
        best = np.concatenate([saliency_cos(P, unit_inputs(load(h), m)).max(1) for h in H["healthy_ids"]])
        a.gamma = 1 / np.median(best)
    lv = np.array([band_levels(p, n_ch) for p in P])
    i_alpha = int(np.argmax(lv[:, 2] - lv[:, 0]))             # most alpha-up / delta-down memory
    i_slow = int(np.argmax(lv[:, 0] - lv[:, 2]))              # most delta-up / alpha-down memory
    e1 = P[i_alpha] / np.linalg.norm(P[i_alpha])
    e2 = P[i_slow] - (P[i_slow] @ e1) * e1; e2 /= np.linalg.norm(e2)
    scale = np.sqrt(N)                                        # per-neuron amplitude ~1
    grid = np.linspace(-1.3, 1.3, 121)

    fig = plt.figure(figsize=(18, 6.8), facecolor=SURF)
    surfaces = []
    for j, k in enumerate(a.show):
        S = saliency_cos(P, unit_inputs(load(k), m))
        al_all = a.gamma * np.maximum(0, S)
        drive = al_all[:, i_alpha] + al_all[:, i_slow]
        t = int(np.argsort(drive)[int(.8 * (len(drive) - 1))])
        al = al_all[t]
        A, B, E = energy_grid(P, al, e1, e2, grid, scale)
        E = E - E[len(grid) // 2, len(grid) // 2]               # confusion state (centre) = 0
        W = (P.T * al) @ P; np.fill_diagonal(W, 0)
        lam = float(np.linalg.eigvalsh(W)[-1])                  # >1: centre unstable, a valley exists
        surfaces.append((k, al, t, A, B, E, lam))
    zmin = min(E.min() for *_, E, _ in surfaces) * 1.1; zmax = .06
    for j, (k, al, t, A, B, E, lam) in enumerate(surfaces):
        ax = fig.add_subplot(1, 3, j + 1, projection="3d", facecolor=SURF)
        Ec = np.where(E > zmax, np.nan, E)                      # hide the steep outer walls
        ax.plot_surface(A, B, Ec, cmap="viridis", vmin=zmin, vmax=zmax, rstride=2, cstride=2,
                        linewidth=0, antialiased=True, alpha=.95)
        ax.contour(A, B, np.clip(E, None, zmax), levels=16, zdir="z", offset=zmin * 1.3, cmap="viridis", linewidths=.8)
        ax.set_zlim(zmin * 1.3, zmax)
        ax.set_xlabel("toward healthy ALPHA memory", fontsize=9, labelpad=6)
        ax.set_ylabel("toward healthy SLOW memory", fontsize=9, labelpad=6)
        ax.set_zlabel("energy vs. confusion state", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.view_init(elev=32, azim=-58)
        exist = [f"{n} (α={al[i]:.2f})" for n, i in (("alpha", i_alpha), ("slow", i_slow)) if al[i] > 1]
        others = [str(i + 1) for i in range(K) if al[i] > 1 and i not in (i_alpha, i_slow)]
        ax.set_title(f"{k}  {'/'.join(ann[k]['symptom'][:2])}\n"
                     f"α alpha-memory = {al[i_alpha]:.2f},  α slow-memory = {al[i_slow]:.2f}\n"
                     f"largest eigenvalue of W(u) = {lam:.2f}  ({'>1: centre unstable, valleys exist' if lam > 1 else '<1: only the centre, confusion'})"
                     + (f"\n(other memories present off this slice: {', '.join(others)})" if others else ""),
                     fontsize=10, color=INK, loc="left")
    fig.suptitle(f"IDP energy landscape at one moment per person (2-D slice of the 380-D network; "
                 f"γ = {a.gamma:.2f}; valleys = memories that exist, centre = confusion state)",
                 x=.01, ha="left", fontsize=12.5, color=INK)
    fig.subplots_adjust(left=.01, right=.99, top=.8, bottom=.02, wspace=.02)
    fig.savefig(a.out, dpi=150, facecolor=SURF)
    print("saved", a.out, "| alpha memory", i_alpha + 1, "slow memory", i_slow + 1,
          "| cos between them", round(float(P[i_alpha] @ P[i_slow]), 2))


if __name__ == "__main__":
    main()
