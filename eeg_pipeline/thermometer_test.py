"""Compare the plan's thermometer coding with healthy-reference coding on the test set.

Usage: python thermometer_test.py --features features --annotation annotation.json
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import thermometer as TM

BANDS = ["delta", "theta", "alpha", "beta"]          # gamma dropped (muscle / noise)
K = 5
MIN_EPOCHS = 60
POST = ["O1", "O2", "P3", "P4", "Pz"]


def group_of(sym):
    s = set(sym)
    return ("healthy" if "normal" in s else "AD" if "ad" in s else
            "MCI" if "mci" in s else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="features")
    p.add_argument("--annotation", required=True)
    p.add_argument("--fig", default="thermometer_comparison.png")
    a = p.parse_args()
    ann = {d["serial"]: d for d in json.load(open(a.annotation))["data"]}

    subj = {}
    for f in sorted(Path(a.features).glob("*.npz")):
        d = np.load(f)
        g = group_of(ann[f.stem]["symptom"])
        if g is None or len(d["rel_power"]) < MIN_EPOCHS:
            continue
        subj[f.stem] = dict(g=g, feat=TM.select(d["rel_power"], d["bands"], BANDS),
                            ch=list(d["ch_names"]))
    ch = next(iter(subj.values()))["ch"]
    post = [ch.index(c) for c in POST]
    healthy = [s["feat"] for s in subj.values() if s["g"] == "healthy"]
    th = TM.fit_reference_thresholds(healthy, K=K, per_subject=100)
    n_ch, n_b = len(ch), len(BANDS)
    N = n_ch * n_b * K
    print(f"subjects with >= {MIN_EPOCHS} epochs: " +
          ", ".join(f"{g} {sum(s['g'] == g for s in subj.values())}" for g in ("healthy", "MCI", "AD")))
    print(f"pattern length N = {n_ch} channels x {n_b} bands x {K} bits = {N}")

    res = {}
    for name, enc in (("plan", lambda f: TM.encode_plan(f, K)), ("ref", lambda f: TM.encode_ref(f, th))):
        codes = {k: enc(s["feat"]) for k, s in subj.items()}
        allc = np.concatenate(list(codes.values()))
        on = (allc > 0).mean(0)
        dead = np.mean((on == 0) | (on == 1))                 # bits that never change
        nearly = np.mean((on < 0.02) | (on > 0.98))
        # overlap between patterns from DIFFERENT people (Hopfield crosstalk)
        rng = np.random.default_rng(1)
        keys = list(codes)
        ov = []
        for _ in range(3000):
            i, j = rng.choice(len(keys), 2, replace=False)
            x = codes[keys[i]][rng.integers(len(codes[keys[i]]))].astype(float)
            y = codes[keys[j]][rng.integers(len(codes[keys[j]]))].astype(float)
            ov.append(x @ y / N)
        lvl = {k: TM.decode_level(c, n_ch, n_b, K)[:, post].mean((0, 1)) for k, c in codes.items()}
        res[name] = dict(codes=codes, dead=dead, nearly=nearly, overlap=float(np.mean(ov)), lvl=lvl)

    print(f"\n{'':<34}{'plan (own max)':>16}{'ref (healthy)':>15}")
    print(f"{'bits that never change':<34}{res['plan']['dead']:>16.0%}{res['ref']['dead']:>15.0%}")
    print(f"{'bits stuck >98% one way':<34}{res['plan']['nearly']:>16.0%}{res['ref']['nearly']:>15.0%}")
    print(f"{'mean overlap, different people':<34}{res['plan']['overlap']:>16.2f}{res['ref']['overlap']:>15.2f}")
    print("\nposterior share of bits ON (theta / alpha), group means:")
    for g in ("healthy", "MCI", "AD"):
        ks = [k for k, s in subj.items() if s["g"] == g]
        row = f"  {g:<10}"
        for name in ("plan", "ref"):
            th_ = np.mean([res[name]["lvl"][k][1] for k in ks]); al = np.mean([res[name]["lvl"][k][2] for k in ks])
            row += f"   {name}: θ {th_:.2f} α {al:.2f}"
        print(row)
    # how well does each code preserve the person's actual theta/alpha level?
    from scipy.stats import spearmanr
    true_ta = {k: s["feat"][:, post, 1].mean() / s["feat"][:, post, 2].mean() for k, s in subj.items()}
    for name in ("plan", "ref"):
        code_ta = {k: res[name]["lvl"][k][1] - res[name]["lvl"][k][2] for k in subj}
        rho = spearmanr([true_ta[k] for k in subj], [code_ta[k] for k in subj])[0]
        print(f"rank correlation, real theta/alpha vs coded theta-minus-alpha ({name}): {rho:+.2f}")

    # -------------------------------------------------------------- figure
    INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
    COL = {"healthy": "#2a78d6", "MCI": "#eda100", "AD": "#eb6834"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURF,
                             gridspec_kw=dict(width_ratios=[1.15, 1, 1]))
    ax = axes[0]
    o1 = ch.index("O1")
    pooled = np.concatenate([TM.subsample(h, 100, np.random.default_rng(0)) for h in healthy])[:, o1, 2]
    ax.hist(pooled, bins=40, color="#cfcdc6", edgecolor=SURF, label="healthy epochs (reference)")
    for t in th[o1, 2]:
        ax.axvline(t, color=INK, lw=1)
    for g in ("healthy", "AD"):                      # the group's most typical person
        ks = [k for k, s in subj.items() if s["g"] == g]
        meds = np.array([np.median(subj[k]["feat"][:, o1, 2]) for k in ks])
        k = ks[int(np.argmin(np.abs(meds - np.median(meds))))]
        ax.axvline(np.median(subj[k]["feat"][:, o1, 2]), color=COL[g], lw=2.5, ls="--",
                   label=f"median epoch, {g} {k}")
    ax.set_xlabel("relative alpha power at O1")
    ax.set_title("A  Reference thresholds (black) for one feature:\nalpha at O1, 5 bits", loc="left", fontsize=10.5)
    ax.legend(frameon=False, fontsize=8.5)
    for i, (name, title) in enumerate((("plan", "B  Plan's coding (% of own max)"),
                                       ("ref", "C  Healthy-reference coding"))):
        ax = axes[i + 1]
        for g in ("healthy", "MCI", "AD"):
            ks = [k for k, s in subj.items() if s["g"] == g]
            x = [res[name]["lvl"][k][2] for k in ks]
            y = [res[name]["lvl"][k][1] for k in ks]
            ax.scatter(x, y, s=60, color=COL[g], label=g, edgecolor=SURF, lw=1.5, zorder=3)
        ax.plot([0, 1], [0, 1], color="#e4e3df", lw=1, zorder=1)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("share of posterior ALPHA bits on")
        ax.set_ylabel("share of posterior THETA bits on")
        ax.set_title(title + "\none dot = one person (mean over epochs)", loc="left", fontsize=10.5)
        ax.legend(frameon=False, fontsize=9, loc="upper right")
    for ax in axes:
        ax.set_facecolor(SURF); ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, color="#e4e3df", lw=0.6)
    fig.tight_layout()
    fig.savefig(a.fig, dpi=160, facecolor=SURF)
    print("saved", a.fig)


if __name__ == "__main__":
    main()
