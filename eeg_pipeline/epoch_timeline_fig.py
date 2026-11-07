"""Timeline figure: which parts of each recording became epochs.

Usage: python epoch_timeline_fig.py --edf-dir caueeg-dataset/signal/edf --event-dir caueeg-dataset/event \
           --features features --out epoch_timeline.png 00722 00529 00115
"""
import sys, json, argparse, warnings; warnings.filterwarnings("ignore")
import numpy as np, mne, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import caueeg_preprocess as P
mne.set_log_level("ERROR")
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
C = {"EC": "#2a78d6", "EO": "#cfcdc6", "photic": "#eb6834", "hv": "#eda100", "art": "#b0afa9", "ep": "#1baf7a"}
ap = argparse.ArgumentParser()
ap.add_argument("ids", nargs="+")
ap.add_argument("--edf-dir", required=True); ap.add_argument("--event-dir", required=True)
ap.add_argument("--features", default="features"); ap.add_argument("--out", default="epoch_timeline.png")
A = ap.parse_args()
ids = A.ids
fig, axes = plt.subplots(len(ids), 1, figsize=(13, 2.7 * len(ids)), facecolor=SURF)
for ax, rid in zip(np.atleast_1d(axes), ids):
    raw = mne.io.read_raw_edf(f"{A.edf_dir}/{rid}.edf", preload=False)
    T = raw.n_times / 200
    ica_skip, skip, photic, _ = P.parse_events(f"{A.event_dir}/{rid}.json", 200, T)
    d = np.load(f"{A.features}/{rid}.npz"); on = d["onsets_s"]
    rows = {"eyes": 4, "photic": 3, "hv": 2, "excluded": 1, "epochs used": 0}
    # eyes state from events (signal time)
    raw_ev = sorted((int(s), str(l)) for s, l in json.load(open(f"{A.event_dir}/{rid}.json")))
    gaps, _ = P.pause_gaps(raw_ev)
    eyes, t0 = None, 0.0
    for s, l in raw_ev + [(10**9, "END")]:
        t = min(P.to_signal_sample(s, gaps) / 200, T) if l != "END" else T
        L = l.lower()
        if L in ("eyes closed", "eyes open", "END", "passive eye close", "passive eye open"):
            if eyes: ax.add_patch(Rectangle((t0, 4 - .35), t - t0, .7, color=C[eyes], lw=0))
            if L != "END": eyes, t0 = ("EC" if "close" in L else "EO"), t
    for a, b, why in skip + ica_skip:
        a, b = max(0, a), min(T, b)
        if b <= a: continue
        if why.startswith("photic"):
            ax.add_patch(Rectangle((a, 3 - .35), b - a, .7, color=C["photic"], lw=0, alpha=.35 if why == "photic_session" else 1))
        elif why == "hyperventilation":
            ax.add_patch(Rectangle((a, 2 - .35), b - a, .7, color=C["hv"], lw=0))
        elif why not in ("not_eyes_closed", "eyes_settle"):
            ax.add_patch(Rectangle((a, 1 - .35), b - a, .7, color=C["art"], lw=0))
    for o in on:
        ax.add_patch(Rectangle((o, -.35), 2, .7, color=C["ep"], lw=0, alpha=.35))
    ax.set_xlim(0, T); ax.set_ylim(-.7, 4.7)
    ax.set_yticks(list(rows.values())); ax.set_yticklabels(list(rows.keys()), color=INK2)
    ax.set_facecolor(SURF); ax.spines[["top", "right", "left"]].set_visible(False); ax.tick_params(left=False)
    qc = json.loads(str(d["qc"]))
    ax.set_title(f"{rid}: {len(on)} epochs (2 s, 50% overlap) from {qc['clean_ec_s']:.0f} s of clean eyes-closed rest, out of {T/60:.1f} min",
                 loc="left", fontsize=10.5, color=INK)
np.atleast_1d(axes)[-1].set_xlabel("time in recording (s)", color=INK2)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=C["EC"], label="eyes closed"), Patch(color=C["EO"], label="eyes open"),
                    Patch(color=C["photic"], label="flash train"), Patch(color=C["photic"], alpha=.35, label="photic session (reserved for CSD)"),
                    Patch(color=C["hv"], label="hyperventilation + 60 s"), Patch(color=C["art"], label="artifacts, pause cuts, eyes closed past 60 s"),
                    Patch(color=C["ep"], alpha=.5, label="epochs used for features")],
           loc="lower center", ncol=4, frameon=False, fontsize=9)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(A.out, dpi=150, facecolor=SURF)
print("ok")
