"""Compare weight matrices between groups.

For each group:
  A  overlap between the person's 8 memories (|cos| of the +/-1 prototypes), averaged over
     people. Memories sorted by how much time they cover. Off-diagonal = how much memories
     blur into each other.
  B  eigenvalues of W: one big eigenvalue + small rest = memories merged into one valley;
     an even spread = 8 separate memories.
  C  one typical person's W (the person with the group's median overlap).

Usage (standalone): python compare_weights.py --groups healthy=weights/W_a.npz,... AD=...
Called from run_analysis.py automatically.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
GCOL = {"healthy": "#2a78d6", "MCI": "#eda100", "AD": "#eb6834"}
BANDS = ["delta", "theta", "alpha", "beta"]


def signs(P):
    return np.where(P >= 0, 1.0, -1.0)


def overlap_matrix(P):
    X = signs(P)
    return np.abs(X @ X.T) / X.shape[1]


def spectrum(P):
    X = signs(P)
    ev = np.linalg.eigvalsh(X.T @ X / X.shape[1])[::-1][: len(X)]   # nonzero eigenvalues
    return ev / ev.sum()


def reorder(W, n_ch):
    idx = np.arange(W.shape[0]).reshape(n_ch, 4, 5).transpose(1, 0, 2).reshape(-1)
    return W[np.ix_(idx, idx)]


def figure(groups, out):
    """groups: {name: [paths to npz with 'W' and 'prototypes']}"""
    data = {g: [np.load(p) for p in ps] for g, ps in groups.items()}
    G = list(data)
    fig, axes = plt.subplots(3, len(G), figsize=(5.2 * len(G), 14), facecolor=SURF,
                             gridspec_kw=dict(hspace=.45, wspace=.3))
    axes = np.atleast_2d(axes).reshape(3, len(G))
    vmaxW = np.percentile(np.abs(np.concatenate([d["W"].ravel() for g in G for d in data[g]])), 99.5)
    ev_all = {}
    for j, g in enumerate(G):
        O = np.mean([overlap_matrix(d["prototypes"]) for d in data[g]], axis=0)
        off = O[np.triu_indices(len(O), 1)].mean()
        ax = axes[0, j]
        im = ax.imshow(O, cmap="Greys", vmin=0, vmax=1)
        for (r, c), v in np.ndenumerate(O):
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if v > .55 else INK)
        ax.set_xticks(range(len(O))); ax.set_yticks(range(len(O)))
        ax.set_xticklabels(range(1, len(O) + 1)); ax.set_yticklabels(range(1, len(O) + 1))
        ax.set_title(f"A  {g} (n={len(data[g])}): overlap between the 8 memories\n"
                     f"average off-diagonal = {off:.2f}  (higher = memories blur together)",
                     loc="left", fontsize=10)
        ev_all[g] = np.array([spectrum(d["prototypes"]) for d in data[g]])
        # C typical person's W
        ovs = [overlap_matrix(d["prototypes"])[np.triu_indices(8, 1)].mean() for d in data[g]]
        d = data[g][int(np.argsort(ovs)[len(ovs) // 2])]
        n_ch = d["W"].shape[0] // 20
        ax = axes[2, j]
        ax.imshow(reorder(d["W"], n_ch), cmap="RdBu_r", norm=TwoSlopeNorm(0, -vmaxW, vmaxW),
                  interpolation="nearest")
        blk = n_ch * 5
        for i in range(1, 4):
            ax.axhline(i * blk - .5, color=INK, lw=.6); ax.axvline(i * blk - .5, color=INK, lw=.6)
        tk = [blk * (i + .5) for i in range(4)]
        ax.set_xticks(tk); ax.set_xticklabels(BANDS, fontsize=8.5)
        ax.set_yticks(tk); ax.set_yticklabels(BANDS, fontsize=8.5); ax.tick_params(length=0)
        ax.set_title(f"C  {g}: a typical person's W (median overlap)", loc="left", fontsize=10)
    for j, g in enumerate(G):
        ax = axes[1, j]
        for gg in G:
            E = ev_all[gg]
            ax.plot(range(1, E.shape[1] + 1), E.mean(0), marker="o", color=GCOL.get(gg, INK2),
                    lw=2.2 if gg == g else 1, alpha=1 if gg == g else .45, label=gg)
            if gg == g:
                ax.fill_between(range(1, E.shape[1] + 1), np.percentile(E, 25, 0), np.percentile(E, 75, 0),
                                color=GCOL.get(gg, INK2), alpha=.15, lw=0)
        ax.axhline(1 / 8, color=INK2, ls=":", lw=1)
        ax.text(8, 1 / 8 + .01, "8 fully separate memories", ha="right", fontsize=8, color=INK2)
        ax.set_xlabel("eigenvalue rank"); ax.set_ylabel("share of W's total strength")
        ax.set_title(f"B  {g}: how W's strength is split\n(first value high = memories merged into one valley)",
                     loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=8.5)
        ax.spines[["top", "right"]].set_visible(False); ax.grid(True, color="#e4e3df", lw=.6)
    for ax in axes.flat:
        ax.set_facecolor(SURF)
    fig.savefig(out, dpi=150, facecolor=SURF, bbox_inches="tight")
    return out
