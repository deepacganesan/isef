"""Explainer figure: what the input-driven (IDP) network does with real EEG windows.

For one healthy, one MCI and one AD person:
  1. one typical window's saliency for each of the 8 healthy memories
  2. the energy landscape that window creates along each memory's direction
     (1-D reduction  U(m) = m^2/2 - alpha log cosh m : a valley only if alpha > 1)
  3. over time, which memory the network would fall into (deepest valley), or
     'confusion' when no memory exists

All three people are scored against the same full healthy model so memory numbers
mean the same thing in every row (illustration only; for analysis use idp_weights.py,
which scores healthy people leave-one-out). gamma is set so that a memory exists in
half of all healthy windows.

Usage: python explain_idp.py --binary binary --weights weights --annotation annotation.json --out idp_explained.png
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from idp_weights import unit_inputs, saliency_cos
import build_weights as BW

BANDS = ["δ", "θ", "α", "β"]
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
MEMC = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a5cd6", "#d6457a", "#3fb8c4", "#7a6a55"]
CONF = "#e4e3df"


def label(p, n_ch):
    lv = p.reshape(n_ch, 4, 5).mean((0, 2))
    up, dn = int(np.argmax(lv)), int(np.argmin(lv))
    return f"{BANDS[up]}↑ {BANDS[dn]}↓"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="binary")
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--annotation", required=True)
    ap.add_argument("--show", nargs=3, default=["01086", "00722", "01167"])
    ap.add_argument("--out", default="idp_explained.png")
    a = ap.parse_args()
    ann = {d["serial"]: d for d in json.load(open(a.annotation))["data"]}
    H = np.load(Path(a.weights) / "W_HC.npz")
    P, m = H["prototypes"], H["mean_vector"]; K = len(P)
    n_ch = P.shape[1] // 20
    names = [f"{i + 1}  {label(P[i], n_ch)}" for i in range(K)]
    load = lambda k: np.load(Path(a.binary) / f"{k}.npz")["codes"]
    healthy = [h for h in H["healthy_ids"]]
    best_h = np.concatenate([saliency_cos(P, unit_inputs(load(h), m)).max(1) for h in healthy])
    gamma = 1 / np.median(best_h)

    fig, axes = plt.subplots(3, 3, figsize=(16, 11), facecolor=SURF,
                             gridspec_kw=dict(width_ratios=[1, 1.15, 1.6], hspace=.55, wspace=.32))
    mm = np.linspace(-2.5, 2.5, 301)
    for r, k in enumerate(a.show):
        S = saliency_cos(P, unit_inputs(load(k), m))
        alpha = gamma * np.maximum(0, S)
        t = int(np.argsort(S.max(1))[len(S) // 2])                # a typical window
        al = alpha[t]
        tag = f"{k} ({'/'.join(ann[k]['symptom'][:2])})"
        # 1 saliency bars
        ax = axes[r, 0]
        ax.barh(range(K), al, color=[MEMC[i] if al[i] > 1 else CONF for i in range(K)])
        ax.axvline(1, color=INK, lw=1.2, ls="--")
        ax.set_yticks(range(K)); ax.set_yticklabels(names, fontsize=8.5); ax.invert_yaxis()
        ax.set_xlim(0, max(2.2, al.max() * 1.1))
        ax.set_xlabel("saliency α  (memory exists if α > 1)", fontsize=9)
        ax.set_title(f"{tag}\n1. one typical moment: how much it\nlights up each healthy memory", loc="left", fontsize=9.5)
        # 2 energy landscape
        ax = axes[r, 1]
        for i in range(K):
            U = .5 * mm ** 2 - al[i] * np.log(np.cosh(mm))
            on = al[i] > 1
            ax.plot(mm, U, color=MEMC[i] if on else "#b0afa9", lw=2.2 if on else .9,
                    label=f"memory {i + 1}" if on else None, zorder=3 if on else 1)
        ax.axvline(0, color=INK2, lw=.6, ls=":")
        ax.set_ylim(-.35, 1.2); ax.set_xlim(-2.5, 2.5)
        ax.set_xlabel("position along the memory's direction  (0 = confusion state)", fontsize=9)
        ax.set_ylabel("energy", fontsize=9)
        n_on = int((al > 1).sum())
        ax.set_title(f"2. the landscape this moment creates:\n"
                     f"{n_on} valle{'y' if n_on == 1 else 'ys'} (memories that exist)"
                     + ("" if n_on else ", ball rolls to 0"), loc="left", fontsize=9.5)
        if n_on:
            ax.legend(frameon=False, fontsize=8, loc="upper center")
        # 3 timeline
        ax = axes[r, 2]
        n = min(150, len(S))
        state = np.where(alpha[:n].max(1) > 1, alpha[:n].argmax(1), K)
        ax.imshow(state[None, :], aspect="auto", cmap=ListedColormap(MEMC[:K] + [CONF]),
                  vmin=-.5, vmax=K + .5, interpolation="nearest")
        ax.axvline(t, color=INK, lw=1.5) if t < n else None
        ax.set_yticks([])
        ax.set_xlabel("window (2 s each)", fontsize=9)
        full = np.where(alpha.max(1) > 1, alpha.argmax(1), K)
        share = np.bincount(full, minlength=K + 1) / len(full)
        top = [f"mem {i + 1}: {share[i]:.0%}" for i in np.argsort(-share[:K])[:2]]
        ax.set_title(f"3. moment by moment: which memory it falls into  (gray = confusion, none exists)\n"
                     f"confusion {share[K]:.0%} of windows · most used: {', '.join(top)}"
                     f" · black line = the moment in 1–2", loc="left", fontsize=9.5)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=MEMC[i], label=names[i]) for i in range(K)] + [Patch(color=CONF, label="confusion (no memory exists)")],
               loc="lower right", bbox_to_anchor=(.99, -.035), ncol=5, frameon=False, fontsize=9,
               title="memory colors in column 3", title_fontsize=9)
    for ax in axes.flat:
        ax.set_facecolor(SURF)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle(f"How the IDP network reacts to real EEG: each 2-s window reshapes the landscape   "
                 f"(γ = {gamma:.2f}, set so a memory exists in half of healthy windows)",
                 x=.01, ha="left", fontsize=12.5, color=INK)
    fig.savefig(a.out, dpi=150, facecolor=SURF, bbox_inches="tight")
    print("saved", a.out, " gamma", round(gamma, 2))


if __name__ == "__main__":
    main()
