"""Figure + lead-time analysis for Experiment 1. Run after hopfield_ews.py."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

r = np.load("results.npz")
d = r["damage"]
BASE = d <= 0.12                     # "healthy" reference period
COLLAPSE_THR = 0.90                  # recall below this = clinically visible loss

INK, INK2, BLUE, ORANGE, SURF = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#fcfcfb"

# name, label, direction (+1 rises / -1 falls), log scale
METRICS = [
    ("recall",    "Recall from 10%-corrupted cue\n(the visible symptom)", -1, False),
    ("basin",     "Basin of attraction radius\n(hidden)",                 -1, False),
    ("kick_fail", "Fraction of small kicks\nthat knock the memory out",    +1, False),
    ("ac1",       "Lag-1 autocorrelation\nof memory overlap",              +1, False),
    ("eeg_var",   "Virtual-EEG variance\n(log scale)",                    +1, True),
    ("eeg_ac1",   "Virtual-EEG lag-1\nautocorrelation",                    +1, False),
]


def collapse_point(rec):
    idx = np.where(rec < COLLAPSE_THR)[0]
    return d[idx[0]] if idx.size else np.nan


def warning_point(x, direction, log):
    """First damage level where the signal leaves its healthy range and stays
    out for 3 consecutive levels. Healthy range = baseline mean +/- max(3 SD, 10%)."""
    y = np.log10(x) if log else x.copy()
    b = y[BASE]
    b = b[np.isfinite(b)]
    if b.size < 2:
        return np.nan
    if log:
        tol = max(3 * b.std(), 0.5)            # half an order of magnitude
    else:
        tol = max(3 * b.std(), 0.10 * abs(b.mean()), 0.05)
    out = (y - b.mean()) * direction > tol
    for j in range(len(d) - 2):
        if out[j] and out[j + 1] and out[j + 2]:
            return d[j]
    return np.nan


# ------------------------------------------------------------ lead times
S = r["recall"].shape[0]
dc = np.array([collapse_point(r["recall"][s]) for s in range(S)])
rows = []
print(f"Collapse (recall < {COLLAPSE_THR}) at damage: mean {dc.mean():.2f}  per network {np.round(dc, 2)}\n")
print(f"{'signal':<12}{'warns at':>10}{'lead (damage)':>15}{'networks warned early':>24}")
for key, _, sgn, log in METRICS[1:]:
    dw = np.array([warning_point(r[key][s], sgn, log) for s in range(S)])
    lead = dc - dw
    early = int(np.sum(lead > 0))
    rows.append((key, np.nanmean(dw), np.nanmean(lead), early))
    print(f"{key:<12}{np.nanmean(dw):>10.2f}{np.nanmean(lead):>15.2f}{early:>18d} / {S}")
mild = np.array([d[np.argmax(r["recall"][s] < 0.99)] for s in range(S)])
print(f"\nFor comparison, recall first slips below 0.99 at damage {mild.mean():.2f} "
      f"(lead {np.mean(dc - mild):.2f}) - the symptom itself is a late, weak warning.")

# ------------------------------------------------------------ figure
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2})
fig, axes = plt.subplots(2, 3, figsize=(13, 7.4), sharex=True, facecolor=SURF)
for ax, (key, label, sgn, log) in zip(axes.flat, METRICS):
    ax.set_facecolor(SURF)
    y = r[key]
    m = np.nanmean(np.log10(y) if log else y, axis=0)
    lo = np.nanpercentile(np.log10(y) if log else y, 10, axis=0)
    hi = np.nanpercentile(np.log10(y) if log else y, 90, axis=0)
    if log:
        m, lo, hi = 10 ** m, 10 ** lo, 10 ** hi
        ax.set_yscale("log")
    ax.axvspan(dc.min(), dc.max(), color=ORANGE, alpha=0.12, lw=0)
    ax.axvline(dc.mean(), color=ORANGE, lw=1.5, ls="--")
    ax.fill_between(d, lo, hi, color=BLUE, alpha=0.18, lw=0)
    ax.plot(d, m, color=BLUE, lw=2)
    if key != "recall":
        wp = [rw for rw in rows if rw[0] == key][0][1]
        if np.isfinite(wp):
            ax.axvline(wp, color=INK2, lw=1, ls=":")
            ax.text(wp + 0.01, 0.04, "warns", transform=ax.get_xaxis_transform(),
                    color=INK2, fontsize=8.5)
    else:
        ax.axhline(COLLAPSE_THR, color=INK2, lw=0.8, ls=":")
        ax.text(dc.mean() - 0.01, 0.04, "collapse", transform=ax.get_xaxis_transform(),
                color=ORANGE, fontsize=8.5, ha="right")
    ax.set_title(label, fontsize=10.5, color=INK, loc="left")
    ax.grid(True, color="#e4e3df", lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
for ax in axes[1]:
    ax.set_xlabel("Fraction of synapses lost")
fig.suptitle(f"Early-warning signals before memory collapse in a Hopfield network "
             f"(N={int(r['N'])}, P={int(r['P'])}, {S} networks; line = mean, band = 10–90%)",
             x=0.01, ha="left", color=INK, fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig("fig1_early_warning.png", dpi=180, facecolor=SURF)
print("\nsaved fig1_early_warning.png")
