"""3-D landscape of a person's OWN IDP network, as measured in run_analysis.py.

For the person with their group's median basin radius:
  network     = their 8 prototypes as +/-1 patterns (sorted by how much time they cover)
  landscape   = W(xi_1): cued with their most common brain state,
                alpha_nu = gamma * max(0, xi_nu . xi_1 / N), W = sum alpha_nu xi_nu xi_nu^T / N
  slice       = plane through memory 1 and memory 2 (their two most common states)
  height      = continuous Hopfield energy per neuron, relative to the confusion state (centre)

Usage: python landscape_own.py --run run75 --out landscape_own.png
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GAMMA = 2.0
INK, SURF = "#0b0b0b", "#fcfcfb"


def energy_plane(Xi, alpha, e1, e2, grid):
    N = Xi.shape[1]
    a, b = np.meshgrid(grid, grid)
    X = np.sqrt(N) * (a[..., None] * e1 + b[..., None] * e2)
    V = np.tanh(X)
    proj = V @ Xi.T / np.sqrt(N)
    quad = (proj ** 2 * alpha).sum(-1) - (V ** 2 * (alpha @ Xi ** 2) / N).sum(-1)
    G = (X * V - np.logaddexp(X, -X) + np.log(2)).sum(-1)
    return a, b, (-.5 * quad + G) / N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="run75")
    ap.add_argument("--out", default="landscape_own.png")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(Path(a.run) / "results.csv")))
    pick = {}
    for g in ("HC", "AD"):
        x = sorted([r for r in rows if r["group"] == g], key=lambda r: float(r["R"]))
        pick[g] = x[len(x) // 2]
    grid = np.linspace(-7, 7, 161)
    surf = []
    for g, r in pick.items():
        d = np.load(Path(a.run) / f"W_{r['id']}.npz")
        Xi = np.where(d["prototypes"] >= 0, 1.0, -1.0)
        N = Xi.shape[1]
        alpha = GAMMA * np.maximum(0, Xi @ Xi[0] / N)
        e1 = Xi[0] / np.sqrt(N)
        e2 = Xi[1] - (Xi[1] @ e1) * e1; e2 /= np.linalg.norm(e2)
        A, B, E = energy_plane(Xi, alpha, e1, e2, grid)
        E -= E[len(grid) // 2, len(grid) // 2]
        # valley bottom along each memory's direction in this plane
        pts = []
        for mu in (0, 1):
            d2 = np.array([Xi[mu] @ e1, Xi[mu] @ e2]) / np.sqrt(N); d2 /= np.linalg.norm(d2)
            rr = np.linspace(0, 8, 321)
            ray = []
            for t in rr:
                x = np.sqrt(N) * (t * d2[0] * e1 + t * d2[1] * e2)
                v = np.tanh(x)
                q = ((v @ Xi.T / np.sqrt(N)) ** 2 * alpha).sum() - (v ** 2 * (alpha @ Xi ** 2) / N).sum()
                ray.append((-.5 * q + (x * v - np.logaddexp(x, -x) + np.log(2)).sum()) / N)
            t = rr[int(np.argmin(ray))]
            pts.append((t * d2[0], t * d2[1]))
        surf.append((g, r, A, B, E, alpha, abs(Xi[0] @ Xi[1]) / N, pts))
    zmin = min(s[4].min() for s in surf) * 1.05
    zmax = .25 * abs(zmin)
    fig = plt.figure(figsize=(15, 13), facecolor=SURF)
    for j, (g, r, A, B, E, alpha, ov12, pts) in enumerate(surf):
        ax2 = fig.add_subplot(2, 2, j + 3, facecolor=SURF)
        lv = np.linspace(zmin, zmax, 31)
        cf = ax2.contourf(A, B, np.clip(E, zmin, zmax), levels=lv, cmap="viridis")
        ax2.contour(A, B, np.clip(E, zmin, zmax), levels=lv[::2], colors="white", linewidths=.4, alpha=.6)
        for (px, py), lab, c in ((pts[0], "memory 1", "#eb6834"), (pts[1], "memory 2", "#2a78d6")):
            for sgn in (1, -1):
                ax2.scatter([sgn * px], [sgn * py], color=c, s=60, edgecolor="white", lw=1.2, zorder=5)
            ax2.annotate(lab, (px, py), xytext=(8, 6), textcoords="offset points", color="white",
                         fontsize=10, fontweight="bold")
        ax2.scatter([0], [0], marker="x", color="white", s=60, zorder=5)
        ax2.annotate("confusion", (0, 0), xytext=(8, -14), textcoords="offset points", color="white", fontsize=9)
        ax2.set_aspect("equal")
        ax2.set_xlabel("toward memory 1 (most common state)"); ax2.set_ylabel("toward memory 2 (orthogonal part)")
        ax2.set_title("seen from above: dark = low energy (valleys); dots = valley bottoms\n"
                      "(each memory also has a mirror valley on the opposite side)", loc="left", fontsize=10)
        fig.colorbar(cf, ax=ax2, fraction=.04, pad=.02, label="energy vs. confusion state")
        ax = fig.add_subplot(2, 2, j + 1, projection="3d", facecolor=SURF)
        Ec = np.where(E > zmax, np.nan, E)
        ax.plot_surface(A, B, Ec, cmap="viridis", vmin=zmin, vmax=zmax, rstride=2, cstride=2,
                        linewidth=0, antialiased=True)
        ax.contour(A, B, np.clip(E, None, zmax), levels=18, zdir="z", offset=zmin * 1.25,
                   cmap="viridis", linewidths=.8)
        z0 = zmin * 1.25
        for (px, py), lab, c, dy in ((pts[0], "memory 1", "#eb6834", -.35), (pts[1], "memory 2", "#2a78d6", .25)):
            ax.scatter([px], [py], [z0], color=c, s=45, depthshade=False, zorder=5)
            ax.text(px + .1, py + dy, z0, lab, color=c, fontsize=9)
        ax.set_zlim(zmin * 1.25, zmax)
        ax.set_xlabel("toward memory 1 (most common state)", fontsize=9, labelpad=6)
        ax.set_ylabel("toward memory 2 (orthogonal part)", fontsize=9, labelpad=6)
        ax.set_zlabel("energy vs. confusion state", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.view_init(elev=30, azim=-60)
        name = "healthy" if g == "HC" else "Alzheimer's"
        ax.set_title(f"{name}: {r['id']} (age {r['age']}), the group's median basin radius\n"
                     f"basin radius R = {float(r['R']):.3f}   depth ΔV = {float(r['dV']):.2f}   "
                     f"memory 1–2 overlap = {ov12:.2f} (dots = valley bottoms)\n"
                     f"saliencies when cued with memory 1: {', '.join(f'{x:.2f}' for x in alpha)}",
                     loc="left", fontsize=10, color=INK)
    fig.suptitle("Each person's own memory network (as measured): landscape when cued with their most "
                 "common brain state, sliced through their two most common states",
                 x=.01, ha="left", fontsize=12.5, color=INK)
    fig.subplots_adjust(left=.04, right=.97, top=.9, bottom=.05, wspace=.12, hspace=.12)
    fig.savefig(a.out, dpi=150, facecolor=SURF)
    print("saved", a.out, {g: r["id"] for g, r in pick.items()})


if __name__ == "__main__":
    main()
