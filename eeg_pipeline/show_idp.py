"""Figure for the IDP (input-driven) weight matrices on the test set.

Usage: python show_idp.py --idp idp --out idp_weights.png
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from idp_weights import W_of
from show_weights import reorder_W, BANDS, K_BITS

INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
GCOL = {"healthy": "#2a78d6", "MCI": "#eda100", "AD": "#eb6834"}


def group_of(sym):
    s = set(sym)
    return "healthy" if "normal" in s else "AD" if "ad" in s else "MCI" if "mci" in s else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idp", default="idp")
    ap.add_argument("--show", nargs=3, default=["01086", "00722", "01167"])
    ap.add_argument("--theta", type=float, default=0.5, help="existence threshold 1/gamma for panel B")
    ap.add_argument("--out", default="idp_weights.png")
    a = ap.parse_args()
    D = {f.stem: np.load(f) for f in Path(a.idp).glob("*.npz")}
    D = {k: d for k, d in D.items() if group_of(list(d["symptom"]))}
    gamma = 1 / a.theta

    fig = plt.figure(figsize=(16, 10), facecolor=SURF)
    gs = fig.add_gridspec(2, 3, hspace=.42, wspace=.3)
    # A: saliency over time for three people
    sub = gs[0, :].subgridspec(1, 3, wspace=.12)
    for j, k in enumerate(a.show):
        d = D[k]; S = d["saliency_cos"][:150].T
        ax = fig.add_subplot(sub[j])
        im = ax.imshow(S, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(0, -1, 1), interpolation="nearest")
        ax.contour(S > a.theta, levels=[.5], colors=INK, linewidths=.8)
        ax.set_yticks(range(S.shape[0])); ax.set_yticklabels([f"memory {i + 1}" for i in range(S.shape[0])] if j == 0 else [], fontsize=8.5)
        ax.set_xlabel("window (2 s)", color=INK2)
        ax.set_title(f"{'ABC'[j]}  {k}  ({'/'.join(map(str, d['symptom'][:2]))})", loc="left", fontsize=10)
    cb = fig.colorbar(im, ax=fig.axes[-1], fraction=.04, pad=.02); cb.ax.tick_params(labelsize=8)
    fig.text(.12, .935, "How strongly each 2-s window drives each of the 8 healthy memories (ξ·û, red = strongly). "
             f"Outlined = above θ = {a.theta}: that memory exists as a valley in W(u) for that window.",
             fontsize=10.5, color=INK)
    # D, E: W(u) for one window of the healthy person and one of the AD person
    for j, k in enumerate((a.show[0], a.show[2])):
        d = D[k]; S = d["saliency_cos"]; t = int(np.argsort(S.max(1))[len(S) // 2])     # a typical window
        W = W_of(d["prototypes"], S[t], gamma)
        n_ch = W.shape[0] // (len(BANDS) * K_BITS)
        ax = fig.add_subplot(gs[1, j])
        v = np.percentile(np.abs(W), 99.5) or 1
        ax.imshow(reorder_W(W, n_ch), cmap="RdBu_r", norm=TwoSlopeNorm(0, -v, v), interpolation="nearest")
        blk = n_ch * K_BITS
        for i in range(1, len(BANDS)):
            ax.axhline(i * blk - .5, color=INK, lw=.6); ax.axvline(i * blk - .5, color=INK, lw=.6)
        tk = [blk * (i + .5) for i in range(len(BANDS))]
        ax.set_xticks(tk); ax.set_xticklabels(BANDS, fontsize=8.5); ax.set_yticks(tk); ax.set_yticklabels(BANDS, fontsize=8.5)
        ax.tick_params(length=0)
        al = gamma * np.maximum(0, S[t].astype(float))
        alive = [int(i) + 1 for i in np.where(al > 1)[0]]
        ax.set_title(f"{'DE'[j]}  W(u) for one typical window of {k}  (γ = {gamma:g})\n"
                     f"α = [{', '.join(f'{x:.2f}' for x in al)}]\n"
                     f"memories that exist (α > 1): {', '.join(map(str, alive)) or 'none'}", loc="left", fontsize=9)
    # F: share of windows in which at least one healthy memory exists, across thresholds
    ax = fig.add_subplot(gs[1, 2])
    th = np.linspace(0, 1, 101)
    for k, d in sorted(D.items()):
        g = group_of(list(d["symptom"])); best = d["saliency_cos"].max(1)
        ax.plot(th, [(best > x).mean() for x in th], color=GCOL[g], lw=1.6, alpha=.85)
    for g, c in GCOL.items():
        ax.plot([], [], color=c, lw=2, label=g)
    ax.axvline(a.theta, color=INK2, ls=":", lw=1)
    ax.set_xlabel("existence threshold θ = 1/γ"); ax.set_ylabel("share of windows where ≥1\nhealthy memory exists")
    ax.set_title("F  How often a healthy memory is available\n(one line per person)", loc="left", fontsize=9.5)
    ax.legend(frameon=False, fontsize=9); ax.grid(True, color="#e4e3df", lw=.6)
    ax.spines[["top", "right"]].set_visible(False)
    for ax_ in fig.axes:
        ax_.set_facecolor(SURF)
    fig.savefig(a.out, dpi=150, facecolor=SURF, bbox_inches="tight")
    print("saved", a.out)


if __name__ == "__main__":
    main()
