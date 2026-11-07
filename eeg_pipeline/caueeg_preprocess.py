"""
CAUEEG preprocessing: raw EDF + event JSON  ->  per-epoch band-power features.

Pipeline (one recording), in order:
   1. Load EDF, standardize channel names, set the 10-20 montage
   2. Turn the event JSON into annotations (what to skip for ICA, what to skip for features)
   3. Filter: 60 Hz notch + 0.5-45 Hz band-pass
   4. Find bad channels (flat / very noisy) and gross-artifact segments
   5. Average reference
   6. ICA: fit on a 1-45 Hz copy, auto-flag eye / heart / muscle components, remove them
   7. Interpolate bad channels, re-reference
   8. Keep only clean eyes-closed rest (no photic, no hyperventilation, no artifacts, no drowsiness)
   9. Cut 2-s epochs with 50% overlap, drop epochs with leftover artifacts
  10. Hann taper + FFT per epoch (0.5 Hz resolution) -> band power per channel
  11. Save absolute and relative band power + QC info

Usage
  python caueeg_preprocess.py --edf-dir caueeg-dataset/signal/edf \
                              --event-dir caueeg-dataset/event \
                              --out features --jobs 8 [--qc 20]
  (--ids 00001 00002 to run a few; --qc N saves before/after-ICA plots for the first N)

Output per recording: features/<id>.npz  (abs_power, rel_power, onsets_s, ch_names,
bands, photic_blocks, qc) and one summary table features/qc_summary.csv
"""
import argparse
import csv
import json
import os
import re
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")       # one thread per worker process
import numpy as np
import mne

mne.set_log_level("ERROR")

# ------------------------------------------------------------------ settings
LINE_FREQ = 60.0                 # Korea uses 60 Hz mains
BAND_PASS = (0.5, 45.0)          # analysis data
ICA_HP = 1.0                     # ICA is fit on a 1 Hz high-passed copy (fits much better)
EPOCH_S, STEP_S = 2.0, 1.0       # 2-s windows, 50% overlap -> 0.5 Hz resolution
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
TOTAL = (1, 45)                  # denominator for relative power

POST_PHOTIC_S = 5.0              # skip after each flash train (brain is still responding)
POST_SESSION_S = 30.0            # the whole photic session (first train -> last + 30 s) is
                                 # reserved for the CSD half and never used as rest
POST_HV_S = 60.0                 # hyperventilation slows EEG for about a minute after it ends
HV_DEFAULT_S = 180.0             # if an HV-on has no matching HV-off
EYES_SETTLE_S = 2.0              # skip right after eyes close (eye movement + alpha onset)
MAX_EC_S = 60.0                  # use only the first 60 s after each eye closure: in the test set
                                 # theta/alpha rose later in long eyes-closed stretches in 10/13
                                 # recordings (drowsiness mimics AD slowing). None = no limit
ART_MARGIN_S = 3.0               # +/- around technician-marked artifacts (timing is approximate)
PAUSE_MARGIN_S = 2.0             # +/- around pauses (signal is discontinuous there)

GROSS_PTP = 500e-6               # 1-s window peak-to-peak above this = garbage (before ICA)
EPOCH_PTP = 150e-6               # epoch peak-to-peak above this = leftover artifact (after ICA)
FLAT_STD = 0.5e-6                # channel std below this in a 1-s window = flat
FLAT_FRAC = 0.10                 # flat in >10% of windows = bad channel
NOISY_RATIO = 5.0                # median PTP > 5x the channel-median = bad channel
MAX_ICA_EXCLUDE = 5              # never remove more than this many components
SEED = 97

EEG_19 = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
          "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz"]
# old 10-20 names -> modern names used by MNE's montage
NAME_MAP = {"FP1": "Fp1", "FP2": "Fp2", "F3": "F3", "F4": "F4", "C3": "C3", "C4": "C4",
            "P3": "P3", "P4": "P4", "O1": "O1", "O2": "O2", "F7": "F7", "F8": "F8",
            "T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8", "T7": "T7", "T8": "T8",
            "P7": "P7", "P8": "P8", "FZ": "Fz", "CZ": "Cz", "PZ": "Pz"}

# event labels that are bookkeeping, not artifacts
BENIGN = re.compile(r"^(start recording|new montage|new low filters|new sensitivity|"
                    r"hv\s*(fair|good)|hv - dur)", re.I)


# ------------------------------------------------------------------ 1. loading
def load_raw(edf_path):
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    rename, types = {}, {}
    for ch in raw.ch_names:
        key = re.split(r"[-_ ]", ch.strip().upper().replace("EEG ", ""))[0]
        if key in NAME_MAP:
            rename[ch] = NAME_MAP[key]
        elif key in ("EKG", "ECG"):
            rename[ch] = "ECG"
            types["ECG"] = "ecg"
        elif key.startswith("PHOTIC"):
            rename[ch] = "Photic"
            types["Photic"] = "misc"
    raw.rename_channels(rename)
    raw.set_channel_types(types)
    missing = [c for c in EEG_19 if c not in raw.ch_names]
    if missing:
        raise ValueError(f"missing EEG channels {missing}; file has {raw.ch_names}")
    keep = EEG_19 + [c for c in ("ECG", "Photic") if c in raw.ch_names]
    raw.pick(keep)
    raw.set_montage("standard_1020")
    return raw


# ------------------------------------------------------------------ 2. events -> annotations
def pause_gaps(raw_ev):
    """(paused_sample, resumed_sample) pairs, plus a final pause with no resume."""
    gaps, p = [], None
    for s, lab in raw_ev:
        L = lab.lower()
        if L in ("paused", "patient disconnected") and p is None:
            p = s
        elif L in ("recording resumed", "patient connected") and p is not None:
            gaps.append((p, s))
            p = None
    return gaps, p


def to_signal_sample(s, gaps):
    """Event sample (recording clock) -> EDF sample (paused stretches removed)."""
    off = 0
    for a, b in gaps:
        if s >= b:
            off += b - a
        elif s > a:                                 # event during a pause -> the cut point
            return a - off
    return s - off


def photic_alignment(raw, photic_blocks):
    """Share of real flash activity (photic channel) inside the event-marked flash windows,
    and outside them. Correct timing gives roughly 100% / ~1%."""
    if "Photic" not in raw.ch_names or not photic_blocks:
        return np.nan, np.nan
    sf = int(raw.info["sfreq"])
    ph = raw.get_data(picks=["Photic"])[0]
    k = len(ph) // sf
    act = np.ptp(ph[:k * sf].reshape(k, sf), axis=1)
    if np.percentile(act, 99) <= 0:
        return np.nan, np.nan
    act = act > 0.5 * np.percentile(act, 99)
    inside = np.zeros(k, bool)
    for _, on, off, *_ in photic_blocks:
        inside[int(on):int(off)] = True
    if not inside.any() or inside.all():
        return np.nan, np.nan
    return float(act[inside].mean()), float(act[~inside].mean())


def photic_offset(raw, photic_blocks):
    """Median (actual flash-train onset on the photic channel - event onset), seconds.
    ~0 when timing is right. Use the photic channel itself to time flashes for CSD."""
    if "Photic" not in raw.ch_names or not photic_blocks:
        return np.nan
    sf = raw.info["sfreq"]
    ph = np.abs(raw.get_data(picks=["Photic"])[0])
    hi = np.percentile(ph, 99.9)
    if hi <= 0:
        return np.nan
    on = np.where(ph > 0.5 * hi)[0]
    if on.size == 0:
        return np.nan
    starts = on[np.r_[0, np.where(np.diff(on) > 2 * sf)[0] + 1]] / sf
    d = [starts[np.argmin(np.abs(starts - b[1]))] - b[1] for b in photic_blocks]
    d = [x for x in d if abs(x) < 15]
    return float(np.median(d)) if d else np.nan


def parse_events(event_path, sfreq, t_end):
    """Returns (ica_skip, feature_skip, photic_blocks, unknown_labels).
    ica_skip     : intervals that are garbage for everything (pauses, gross artifacts)
    feature_skip : everything that is not clean eyes-closed rest
    Intervals are (start_s, end_s, reason), in SIGNAL time.

    IMPORTANT: CAUEEG event samples count the time the recording was paused, but the
    paused stretches are cut out of the EDF. So every event after a pause is shifted
    by the total paused time before it. We map event time -> signal time first.
    (Verified against the photic channel: 14% -> 100% overlap with real flashes.)"""
    raw_ev = sorted(((int(s), str(lab).strip()) for s, lab in json.load(open(event_path))),
                    key=lambda x: x[0])
    gaps, final_pause = pause_gaps(raw_ev)
    ev = [(to_signal_sample(s, gaps) / sfreq, lab) for s, lab in raw_ev]
    ica_skip, skip = [], []
    # the cut points are signal discontinuities -> skip a margin around each
    for a, _ in gaps:
        j = to_signal_sample(a, gaps) / sfreq
        ica_skip.append((j - PAUSE_MARGIN_S, j + PAUSE_MARGIN_S, "pause_junction"))
    if final_pause is not None:                   # paused and never resumed = end of recording
        j = to_signal_sample(final_pause, gaps) / sfreq
        ica_skip.append((j - PAUSE_MARGIN_S, t_end, "after_final_pause"))
    photic, unknown, flags = [], set(), set()
    eyes = None                                   # unknown until the first eyes event
    eyes_since = 0.0
    hv_on = photic_on = None
    photic_freq = None
    drowsy_on = None

    def close_eyes_interval(t):
        # anything that wasn't eyes-closed between eyes_since and t is excluded
        if eyes != "EC":
            skip.append((eyes_since, t, "not_eyes_closed"))
        else:
            skip.append((eyes_since, min(t, eyes_since + EYES_SETTLE_S), "eyes_settle"))
            if MAX_EC_S is not None and t > eyes_since + MAX_EC_S:
                skip.append((eyes_since + MAX_EC_S, t, "long_eyes_closed"))

    for t, lab in ev:
        L = lab.lower()
        if L in ("eyes closed", "passive eye close") or L in ("eyes open", "passive eye open"):
            new = "EC" if "close" in L else "EO"
            close_eyes_interval(t)
            if drowsy_on is not None:             # drowsiness ends when eye state changes
                skip.append((drowsy_on, t, "drowsy"))
                drowsy_on = None
            eyes, eyes_since = new, t
            if L.startswith("passive"):
                skip.append((t - ART_MARGIN_S, t + ART_MARGIN_S, "artifact"))
        elif L.startswith("photic on"):
            m = re.search(r"([\d.]+)\s*hz", L)
            photic_freq = float(m.group(1)) if m else np.nan
            photic_on = t
        elif L.startswith("photic off"):
            if photic_on is not None:
                skip.append((photic_on, t + POST_PHOTIC_S, "photic"))
                photic.append([photic_freq, photic_on, t, eyes])
                photic_on = None
        elif L.startswith("hv") and L.endswith("on"):
            hv_on = t
        elif L.startswith("hv") and "off" in L:
            if hv_on is not None:
                skip.append((hv_on, t + POST_HV_S, "hyperventilation"))
                hv_on = None
        elif L in ("paused", "recording resumed", "patient disconnected", "patient connected"):
            pass                                   # handled by the time mapping above
        elif L in ("sleep", "sleep try", "snoring") or L.startswith("drowsy"):
            if drowsy_on is None:                  # skip until the eyes state next changes
                drowsy_on = t
        elif L.startswith("seizure"):
            flags.add("seizure_marked")
            if drowsy_on is None:
                drowsy_on = t
        elif L.startswith("power failure"):       # signal break, like a pause
            ica_skip.append((t - 5.0, t + 5.0, "power_failure"))
        elif BENIGN.match(L):
            pass
        else:                                      # technician-marked artifact of any kind
            skip.append((t - ART_MARGIN_S, t + ART_MARGIN_S, "artifact"))
            if not re.search(r"move|artifact|swallow|talk|cough|couch|chew|check|blink|"
                             r"sweat|jerk", L):
                unknown.add(lab)

    close_eyes_interval(t_end)
    if photic_on is not None:
        skip.append((photic_on, t_end, "photic"))
    if hv_on is not None:
        skip.append((hv_on, hv_on + HV_DEFAULT_S + POST_HV_S, "hyperventilation"))
    if drowsy_on is not None:
        skip.append((drowsy_on, t_end, "drowsy"))

    if photic:                                     # whole session off-limits for rest
        skip.append((min(b[1] for b in photic), max(b[2] for b in photic) + POST_SESSION_S,
                     "photic_session"))

    # quiet window after each flash train = time until the next event (for the CSD half)
    times = np.array([t for t, _ in ev])
    for blk in photic:
        nxt = times[times > blk[2] + 1e-6]
        blk.append(float(nxt[0] - blk[2]) if nxt.size else float(t_end - blk[2]))
    return ica_skip, skip, photic, sorted(unknown | flags)


def to_annotations(intervals, prefix, t_end):
    on, du, de = [], [], []
    for a, b, why in intervals:
        a, b = max(0.0, a), min(t_end, b)
        if b > a:
            on.append(a); du.append(b - a); de.append(f"{prefix}{why}")
    return mne.Annotations(on, du, de)


# ------------------------------------------------------------------ 4. bad channels / segments
def window_stats(data, sfreq):
    n = int(sfreq)
    k = data.shape[1] // n
    w = data[:, :k * n].reshape(data.shape[0], k, n)
    return np.ptp(w, axis=2), w.std(axis=2)


def find_bad_channels(raw):
    data = raw.get_data(picks="eeg")
    ptp, sd = window_stats(data, raw.info["sfreq"])
    flat = (sd < FLAT_STD).mean(1) > FLAT_FRAC
    med = np.median(ptp, axis=1)
    noisy = med > NOISY_RATIO * np.median(med)
    names = np.array(raw.copy().pick("eeg").ch_names)
    return list(names[flat | noisy])


def gross_artifact_intervals(raw):
    data = raw.get_data(picks="eeg", exclude="bads")
    ptp, _ = window_stats(data, raw.info["sfreq"])
    bad = np.where((ptp > GROSS_PTP).any(0))[0]
    return [(float(i) - 0.5, float(i) + 1.5, "gross") for i in bad]


# ------------------------------------------------------------------ 6. ICA
def run_ica(raw):
    raw_fit = raw.copy().filter(ICA_HP, None, picks="eeg")
    # after average reference, 19 channels carry only 18 independent signals
    rank = mne.compute_rank(raw_fit.copy().pick("eeg", exclude="bads"), rank=None)["eeg"]
    # Picard = extended Infomax solved much faster (pip install python-picard)
    try:
        import picard  # noqa: F401
        method, params = "picard", dict(ortho=False, extended=True)
    except ImportError:
        method, params = "infomax", dict(extended=True)
    ica = mne.preprocessing.ICA(n_components=int(rank), method=method, fit_params=params,
                                random_state=SEED, max_iter="auto")
    # decim=2 -> fit on 100 Hz samples; plenty for 1-45 Hz data and twice as fast
    ica.fit(raw_fit, picks="eeg", reject_by_annotation=True, decim=2)

    found = {"eye": [], "heart": [], "muscle": []}
    # No EOG channel in CAUEEG -> use the frontal-pole electrodes as the eye reference.
    # Two guards, because after average referencing posterior alpha leaks into Fp1/Fp2:
    #  (1) correlate only at 1-5 Hz (blinks/eye movements), not up into alpha
    #  (2) keep a candidate only if its scalp map is front-heavy
    idx, eog_sc = ica.find_bads_eog(raw_fit, ch_name=["Fp1", "Fp2"], threshold=3.0,
                                    l_freq=1, h_freq=5)
    eog_strength = np.max(np.abs(np.atleast_2d(eog_sc)), axis=0)
    idx = sorted(idx, key=lambda i: -eog_strength[i])          # strongest first
    maps = np.abs(ica.get_components())                 # (channels, components)
    names = ica.ch_names
    front = [names.index(c) for c in ("Fp1", "Fp2", "F7", "F8") if c in names]
    back = [names.index(c) for c in ("O1", "O2", "P3", "P4", "Pz") if c in names]
    found["eye"] = [i for i in idx if maps[front, i].mean() > 2.0 * maps[back, i].mean()]
    if "ECG" in raw_fit.ch_names:
        try:
            idx, sc = ica.find_bads_ecg(raw_fit, ch_name="ECG", method="ctps", threshold="auto")
            found["heart"] = sorted(idx, key=lambda i: -np.abs(sc[i]))
        except Exception:
            pass
    try:
        idx, sc = ica.find_bads_muscle(raw_fit, threshold=0.7)
        found["muscle"] = sorted(idx, key=lambda i: -sc[i])
    except Exception:
        pass

    exclude = []
    for kind in ("eye", "heart", "muscle"):            # priority order for the cap
        for i in found[kind]:
            if i not in exclude and len(exclude) < MAX_ICA_EXCLUDE:
                exclude.append(int(i))
    ica.exclude = exclude
    return ica, {k: [int(i) for i in v] for k, v in found.items()}, int(rank)


# ------------------------------------------------------------------ 10. spectra
def band_powers(epochs):
    sf = epochs.info["sfreq"]
    n = int(round(EPOCH_S * sf))
    spec = epochs.compute_psd(method="welch", fmin=TOTAL[0], fmax=TOTAL[1], n_fft=n,
                              n_per_seg=n, n_overlap=0, window="hann", picks="eeg")
    psd = spec.get_data() * 1e12                       # V^2/Hz -> uV^2/Hz
    f = spec.freqs
    df = f[1] - f[0]
    absp = np.stack([psd[..., (f >= lo) & (f < hi)].sum(-1) * df for lo, hi in BANDS.values()],
                    axis=-1)                           # (epochs, channels, bands)
    total = psd[..., (f >= TOTAL[0]) & (f < TOTAL[1])].sum(-1) * df
    return absp, absp / total[..., None]


# ------------------------------------------------------------------ one recording
def process(rec_id, edf_dir, event_dir, out_dir, qc_plot=False):
    edf = Path(edf_dir) / f"{rec_id}.edf"
    evf = Path(event_dir) / f"{rec_id}.json"
    raw = load_raw(edf)
    sf, t_end = raw.info["sfreq"], raw.times[-1]

    ica_skip, feat_skip, photic, unknown = parse_events(evf, sf, t_end)
    align_in, align_out = photic_alignment(raw, photic)     # timing sanity check
    flash_offset = photic_offset(raw, photic)

    # 3. filter
    raw.notch_filter(LINE_FREQ, picks="eeg")
    raw.filter(*BAND_PASS, picks="eeg")

    # 4. bad channels + gross artifacts
    raw.info["bads"] = find_bad_channels(raw)
    gross = gross_artifact_intervals(raw)
    raw.set_annotations(to_annotations(ica_skip + gross, "BAD_", t_end))

    # 5. average reference (bad channels left out of the average)
    raw.set_eeg_reference("average", projection=False)

    # 6. ICA
    raw_before = raw.copy() if qc_plot else None
    ica, found, rank = run_ica(raw)
    ica.apply(raw)

    # 7. interpolate bad channels, re-reference over all 19
    n_bad = len(raw.info["bads"])
    bads = list(raw.info["bads"])
    if n_bad:
        raw.interpolate_bads(reset_bads=True)
        raw.set_eeg_reference("average", projection=False)

    # 8. keep only clean eyes-closed rest
    raw.set_annotations(to_annotations(ica_skip + gross + feat_skip, "BAD_", t_end))

    # 9. epochs
    epochs = mne.make_fixed_length_epochs(raw, duration=EPOCH_S, overlap=EPOCH_S - STEP_S,
                                          reject_by_annotation=True, preload=True)
    n_clean_windows = len(epochs)
    epochs.pick("eeg")
    if len(epochs):
        epochs.drop_bad(reject=dict(eeg=EPOCH_PTP))
    onsets = epochs.events[:, 0] / sf - raw.first_time if len(epochs) else np.array([])

    # 10. band power
    if len(epochs):
        absp, relp = band_powers(epochs)
    else:
        absp = relp = np.zeros((0, len(EEG_19), len(BANDS)))

    clean_ec_s = float(_clean_seconds(feat_skip + ica_skip + gross, t_end))
    qc = dict(id=rec_id, duration_s=round(t_end, 1), clean_ec_s=round(clean_ec_s, 1),
              windows_in_clean_ec=n_clean_windows, epochs_kept=len(epochs),
              epochs_dropped_ptp=n_clean_windows - len(epochs),
              bad_channels=";".join(bads), ica_rank=rank,
              ica_removed=";".join(map(str, ica.exclude)),
              ica_eye=";".join(map(str, found["eye"])),
              ica_heart=";".join(map(str, found["heart"])),
              ica_muscle=";".join(map(str, found["muscle"])),
              gross_artifact_s=len(gross), n_photic_blocks=len(photic),
              photic_align_in=round(align_in, 3), photic_align_out=round(align_out, 3),
              photic_offset_s=round(flash_offset, 2),
              unknown_event_labels=";".join(unknown))

    np.savez_compressed(Path(out_dir) / f"{rec_id}.npz",
                        abs_power=absp.astype(np.float32), rel_power=relp.astype(np.float32),
                        onsets_s=onsets.astype(np.float32), ch_names=np.array(EEG_19),
                        bands=np.array(list(BANDS)), band_edges=np.array(list(BANDS.values())),
                        photic_blocks=np.array([[b[0], b[1], b[2], b[4]] for b in photic]
                                               ).reshape(-1, 4),
                        photic_eyes=np.array([str(b[3]) for b in photic]),
                        qc=json.dumps(qc))
    if qc_plot:
        _qc_figure(raw_before, raw, epochs, ica, rec_id, out_dir, feat_skip + ica_skip + gross)
    return qc


def _clean_seconds(intervals, t_end):
    mask = np.ones(int(np.ceil(t_end * 10)) + 1, bool)       # 0.1 s grid
    for a, b, _ in intervals:
        mask[max(0, int(a * 10)):max(0, int(b * 10))] = False
    return mask.sum() / 10


def _qc_figure(before, after, epochs, ica, rec_id, out_dir, skip):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sf = after.info["sfreq"]
    # 10-s window with the most eye activity in the before-ICA frontal channels
    fp = before.get_data(picks=["Fp1", "Fp2"]).mean(0)
    w = int(10 * sf)
    k = len(fp) // w
    start = int(np.argmax(np.ptp(fp[:k * w].reshape(k, w), axis=1))) * w
    t = np.arange(w) / sf
    chans = ["Fp1", "Fp2", "F3", "F4"]
    fig, axes = plt.subplots(len(chans) + 1, 1, figsize=(11, 9), facecolor="#fcfcfb")
    for ax, ch in zip(axes, chans):
        b = before.get_data(picks=[ch])[0, start:start + w] * 1e6
        a = after.get_data(picks=[ch])[0, start:start + w] * 1e6
        ax.plot(t, b, color="#b0afa9", lw=0.8, label="before ICA")
        ax.plot(t, a, color="#2a78d6", lw=0.9, label="after ICA")
        ax.set_ylabel(f"{ch} (µV)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="upper right", fontsize=8)
    axes[0].set_title(f"{rec_id}: removed ICA components {ica.exclude}", loc="left")
    ax = axes[-1]
    if len(epochs):
        sp = epochs.compute_psd(method="welch", fmin=1, fmax=45, n_fft=400, n_per_seg=400,
                                n_overlap=0, window="hann")
        p = sp.get_data().mean(0) * 1e12
        for ch, col in (("O1", "#2a78d6"), ("Fz", "#eb6834")):
            ax.semilogy(sp.freqs, p[sp.ch_names.index(ch)], color=col, lw=1.4, label=ch)
        for lo, hi in BANDS.values():
            ax.axvline(lo, color="#e4e3df", lw=0.8)
        ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("frequency (Hz)  —  clean eyes-closed epochs, mean spectrum")
    ax.set_ylabel("µV²/Hz")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / f"{rec_id}_qc.png", dpi=110, facecolor="#fcfcfb")
    plt.close(fig)


# ------------------------------------------------------------------ batch
def _safe(rec_id, a, qc):
    try:
        return process(rec_id, a.edf_dir, a.event_dir, a.out, qc)
    except Exception as e:
        return dict(id=rec_id, error=f"{type(e).__name__}: {e}",
                    trace=traceback.format_exc(limit=2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--edf-dir", required=True)
    p.add_argument("--event-dir", required=True)
    p.add_argument("--out", default="features")
    p.add_argument("--ids", nargs="*")
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--qc", type=int, default=0, help="save QC plots for the first N recordings")
    a = p.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    ids = a.ids or sorted(f.stem for f in Path(a.edf_dir).glob("*.edf"))
    qc_set = set(ids[:a.qc])
    rows = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(_safe, i, a, i in qc_set): i for i in ids}
        for n, fu in enumerate(as_completed(futs), 1):
            r = fu.result()
            rows.append(r)
            msg = r.get("error") or f"{r['epochs_kept']} epochs, ICA removed [{r['ica_removed']}]"
            print(f"[{n}/{len(ids)}] {r['id']}: {msg}", flush=True)
    keys = sorted({k for r in rows for k in r if k != "trace"})
    with open(Path(a.out) / "qc_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id"] + [k for k in keys if k != "id"])
        w.writeheader()
        for r in sorted(rows, key=lambda r: r["id"]):
            w.writerow({k: v for k, v in r.items() if k != "trace"})
    errs = [r for r in rows if "error" in r]
    print(f"done: {len(rows) - len(errs)} ok, {len(errs)} failed -> {a.out}/qc_summary.csv")


if __name__ == "__main__":
    sys.exit(main())
