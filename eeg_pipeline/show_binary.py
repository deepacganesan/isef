"""Draw binary matrices (features x windows, as in the research plan).

Usage: python show_binary.py binary/01086.npz binary/00722.npz binary/01167.npz --out binary_matrices.png
Black = +1 (bit on), white = -1. Rows are grouped by band so the structure is visible.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BANDS = ["delta", "theta", "alpha", "beta"]
K = 5


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--max-epochs", type=int, default=150)
    p.add_argument("--out", default="binary_matrices.png")
    a = p.parse_args()
    SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
    fig, axes = plt.subplots(1, len(a.files), figsize=(5.2 * len(a.files), 8.6), facecolor=SURF)
    for ax, f in zip(np.atleast_1d(axes), a.files):
        d = np.load(f)
        codes = d["codes"][: a.max_epochs]                  # (epochs, 380)
        n_ch = codes.shape[1] // (len(BANDS) * K)
        ch = [n.split("_")[0] for n in d["feature_names"][:: len(BANDS) * K]]
        # stored order is channel -> band -> bit; show band -> channel -> bit
        img = codes.reshape(len(codes), n_ch, len(BANDS), K).transpose(2, 1, 3, 0)
        img = img.reshape(len(BANDS) * n_ch * K, len(codes))  # (380, epochs) like the plan
        ax.imshow(img > 0, aspect="auto", cmap="Greys", interpolation="nearest")
        block = n_ch * K
        for i in range(1, len(BANDS)):
            ax.axhline(i * block - 0.5, color="#eb6834", lw=1.5)
        ax.set_yticks([block * (i + 0.5) for i in range(len(BANDS))])
        ax.set_yticklabels([f"{b}\n(19 ch × 5 bits)" for b in BANDS], fontsize=9.5, color=INK2)
        ax.tick_params(left=False)
        ax.set_xlabel(f"window (2-s epoch), first {len(codes)} of {len(d['codes'])}", color=INK2)
        sym = "/".join(str(s) for s in d["symptom"][:2])
        on = (d["codes"] > 0).reshape(len(d["codes"]), n_ch, len(BANDS), K).mean((0, 1, 3))
        ax.set_title(f"{f.split('/')[-1][:-4]}  {sym}, age {int(d['age'])}\nbits on: "
                     + "  ".join(f"{b[:1].upper()}{b[1:3]} {v:.0%}" for b, v in zip(BANDS, on)),
                     loc="left", fontsize=10.5, color=INK)
    fig.suptitle("Thermometer-coded EEG: 380 bits per window (black = +1 / on)",
                 x=0.01, ha="left", fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(a.out, dpi=150, facecolor=SURF)
    print("saved", a.out)


if __name__ == "__main__":
    main()
