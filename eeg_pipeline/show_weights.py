"""Look inside the weight matrices: W_HC, the healthy prototypes, patient W's, and
how each person's brain states compare with the healthy model.

Usage: python show_weights.py --weights weights --binary binary --annotation annotation.json
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import build_weights as BW

BANDS = ["delta", "theta", "alpha", "beta"]
K_BITS = 5
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
GCOL = {"healthy": "#2a78d6", "MCI": "#eda100", "AD": "#eb6834"}


def group_of(sym):
    s = set(sym)
    return "healthy" if "normal" in s else "AD" if "ad" in s else "MCI" if "mci" in s else None


def band_major(v, n_ch):
    """Reorder a 380-vector (channel->band->bit) to band->channel->bit."""
    return v.reshape(n_ch, len(BANDS), K_BITS).transpose(1, 0, 2).reshape(-1)


def reorder_W(W, n_ch):
    idx = band_major(np.arange(W.shape[0]), n_ch)
    return W[np.ix_(idx, idx)]


def draw_W(ax, W, n_ch, title, vmax):
    ax.imshow(reorder_W(W, n_ch), cmap="RdBu_r", norm=TwoSlopeNorm(0, -vmax, vmax),
              interpolation="nearest")
    blk = n_ch * K_BITS
    for i in range(1, len(BANDS)):
        ax.axhline(i * blk - .5, color=INK, lw=.6); ax.axvline(i * blk - .5, color=INK, lw=.6)
    t = [blk * (i + .5) for i in range(len(BANDS))]
    ax.set_xticks(t); ax.set_xticklabels(BANDS, fontsize=8.5, color=INK2)
    ax.set_yticks(t); ax.set_yticklabels(BANDS, fontsize=8.5, color=INK2)
    ax.tick_params(length=0)
    ax.set_title(title, loc="left", fontsize=10, color=INK)


def best_match(P, Q, w=None):
    """Weighted mean over P's prototypes of the best |cosine| with any of Q's."""
    s = np.abs(P @ Q.T).max(1)
    return float(np.average(s, weights=w))


def own_overlap(P):
    S = np.abs(P @ P.T)
    return float(S[np.triu_indices(len(P), 1)].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--binary", default="binary")
    ap.add_argument("--annotation", required=True)
    ap.add_argument("--show", nargs=3, default=["01086", "00722", "01167"])
    ap.add_argument("--out", default="weight_matrices.png")
    a = ap.parse_args()
    ann = {d["serial"]: d for d in json.load(open(a.annotation))["data"]}
    H = np.load(Path(a.weights) / "W_HC.npz")
    K = int(H["K"]); healthy = list(H["healthy_ids"])
    n_ch = H["W"].shape[0] // (len(BANDS) * K_BITS)
    ch = [n.split("_")[0] for n in H["feature_names"][:: len(BANDS) * K_BITS]]

    codes = {f.stem: np.load(f)["codes"] for f in Path(a.binary).glob("*.npz") if f.stem != "thresholds"}
    pats = {f.stem: np.load(f) for f in Path(a.weights).glob("*.npz") if f.stem != "W_HC"}

    rows = []
    for k, d in sorted(pats.items()):
        g = group_of(ann[k]["symptom"])
        if g is None:
            continue
        if k in healthy:                       # leave-one-out: rebuild W_HC without this person
            ref = BW.build_healthy(codes, [h for h in healthy if h != k], K, int(H["n_per_subject"]))
            pd = BW.build_patient(codes[k], ref["mean_vector"], K)
            P, sizes, Wref, Pref = pd["prototypes"], pd["cluster_sizes"], ref["W"], ref["prototypes"]
            W = pd["W"]
        else:
            P, sizes, W, Wref, Pref = d["prototypes"], d["cluster_sizes"], d["W"], H["W"], H["prototypes"]
        rows.append(dict(id=k, g=g, match=best_match(P, Pref, sizes),
                         wdist=float(np.linalg.norm(W - Wref) / np.linalg.norm(Wref)),
                         overlap=own_overlap(P)))

    print(f"{'id':<6}{'group':<9}{'match to healthy states':>24}{'|W - W_HC| / |W_HC|':>22}"
          f"{'overlap among own prototypes':>30}")
    for r in sorted(rows, key=lambda r: ("healthy", "MCI", "AD").index(r["g"])):
        print(f"{r['id']:<6}{r['g']:<9}{r['match']:>24.2f}{r['wdist']:>22.2f}{r['overlap']:>30.2f}")
    print("\ngroup means:")
    for g in ("healthy", "MCI", "AD"):
        R = [r for r in rows if r["g"] == g]
        print(f"  {g:<8} match {np.mean([r['match'] for r in R]):.2f}   W distance "
              f"{np.mean([r['wdist'] for r in R]):.2f}   own-prototype overlap {np.mean([r['overlap'] for r in R]):.2f}   (n={len(R)})")

    # ------------------------------------------------------------------ figure
    fig = plt.figure(figsize=(16, 10.5), facecolor=SURF)
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=.35, wspace=.5)
    vmax = np.percentile(np.abs(H["W"]), 99.5)
    ax = fig.add_subplot(gs[0, 0])
    draw_W(ax, H["W"], n_ch, f"A  W_HC (healthy model)\n{len(healthy)} people, K={K} prototypes, 380×380", vmax)
    # healthy prototypes decoded to channel x band (mean over the 5 bits, + = above healthy average)
    sub = gs[0, 1:].subgridspec(2, 4, hspace=.55, wspace=.15)
    lim = np.abs(H["prototypes"]).max()
    for i in range(K):
        axp = fig.add_subplot(sub[i // 4, i % 4])
        img = H["prototypes"][i].reshape(n_ch, len(BANDS), K_BITS).mean(-1).T     # bands x channels
        axp.imshow(img, cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
        axp.set_yticks(range(len(BANDS))); axp.set_yticklabels(BANDS if i % 4 == 0 else [], fontsize=8)
        axp.set_xticks(range(n_ch)); axp.set_xticklabels(ch, rotation=90, fontsize=6)
        axp.tick_params(length=0)
        axp.set_title(f"state {i + 1}: {int(H['cluster_sizes'][i])} windows", fontsize=9, loc="left")
    fig.text(0.29, 0.93, "B  The K=8 healthy prototypes (red = more power than the healthy average, "
             "blue = less)", fontsize=10.5, color=INK)
    for j, k in enumerate(a.show):
        d = pats[k]
        draw_W(fig.add_subplot(gs[1, j]), d["W"], n_ch,
               f"{'CDE'[j]}  W for {k} ({'/'.join(map(str, d['symptom'][:2]))})", vmax)
    ax = fig.add_subplot(gs[1, 3])
    for gi, g in enumerate(("healthy", "MCI", "AD")):
        R = [r for r in rows if r["g"] == g]
        ax.scatter([gi] * len(R), [r["match"] for r in R], s=70, color=GCOL[g], edgecolor=SURF, lw=1.5, zorder=3)
        ax.plot([gi - .25, gi + .25], [np.mean([r["match"] for r in R])] * 2, color=INK, lw=2)
    ax.set_xticks(range(3)); ax.set_xticklabels(["healthy\n(left out)", "MCI", "AD"])
    ax.set_ylabel("how well this person's states\nmatch the healthy states (|cos|)")
    ax.set_title("F  Match to the healthy model", loc="left", fontsize=10)
    ax.set_xlim(-.6, 2.6); ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", color="#e4e3df", lw=.6); ax.set_facecolor(SURF)
    fig.savefig(a.out, dpi=150, facecolor=SURF, bbox_inches="tight")
    print("saved", a.out)


if __name__ == "__main__":
    main()
