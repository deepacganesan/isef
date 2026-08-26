"""
CAUEEG -> IDP binary patterns.

EDF + event JSON  ->  (n_windows, 380) matrix of +/-1

Pipeline:
  clean (filter, ref, ICA) -> 2s windows -> Welch PSD -> 5 band powers
  -> centered log-ratio -> thermometer encode -> +/-1 bits
"""

import json
import numpy as np
import mne
from scipy.signal import welch

mne.set_log_level("ERROR")

EEG_19 = ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz",
          "C3", "C4", "Cz", "P3", "P4", "Pz",
          "T3", "T4", "T5", "T6", "O1", "O2"]

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 30), "gamma": (30, 45)}

SFREQ_EVENTS = 200.0   # CAUEEG event timestamps are sample indices at 200 Hz


# ---------------------------------------------------------------- cleaning

def load_clean(edf_path, n_ica_reject=2):
    """Load an EDF, keep the 19 EEG channels, filter, average-reference, ICA.

    Returns (raw, n_rejected). n_ica_reject caps how many components we drop
    so the rejection count is identical for every recording -- this is the
    group-bias guard. Set to None to let the correlation threshold decide.
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True)

    # CAUEEG labels channels "Fp1-AVG" etc. Strip the montage suffix.
    raw.rename_channels({c: c.split("-")[0] for c in raw.ch_names})
    # Case varies across CAUEEG files (FZ vs Fz), so normalise.
    canon = {c.lower(): c for c in EEG_19}
    raw.rename_channels({c: canon[c.lower()] for c in raw.ch_names
                         if c.lower() in canon})
    raw.pick(EEG_19)

    raw.set_montage(mne.channels.make_standard_montage("standard_1020"),
                    on_missing="warn")

    raw.filter(0.1, 45.0, fir_design="firwin")
    raw.notch_filter(60.0)
    raw.set_eeg_reference("average", projection=False)

    # ICA converges badly on slow drift, so fit on a 1 Hz high-passed copy
    # and apply the solution back to the 0.1 Hz data.
    ica_fit_data = raw.copy().filter(1.0, 45.0, fir_design="firwin")

    # rank drops by 1 after average referencing -> 18, not 19
    ica = mne.preprocessing.ICA(n_components=18, method="fastica",
                                max_iter="auto", random_state=97)
    ica.fit(ica_fit_data)

    # Frontal channels stand in for EOG: blinks load heavily on Fp1/Fp2.
    # For the real 1,379-file run, swap this for mne-icalabel, which also
    # catches muscle and heartbeat components.
    bad, scores = ica.find_bads_eog(raw, ch_name=["Fp1", "Fp2"])

    if n_ica_reject is not None:
        # Fixed budget: take the n worst by |correlation|, pad if too few.
        ranked = np.argsort(-np.abs(scores).max(axis=0))
        bad = list(ranked[:n_ica_reject])

    ica.exclude = bad
    ica.apply(raw)
    return raw, len(bad)


# ---------------------------------------------- features: 19x5 -> 95 dims

def band_features(raw, win_sec=2.0, overlap=0.5):
    """Sliding windows -> centered-log-ratio band powers.

    Returns (n_windows, 95). Feature order is channel-major:
    [Fp1_delta, Fp1_theta, ... Fp1_gamma, Fp2_delta, ...]
    """
    sf = raw.info["sfreq"]
    data = raw.get_data()                      # (19, n_samples)
    n_win = int(win_sec * sf)
    step = int(n_win * (1 - overlap))

    starts = range(0, data.shape[1] - n_win + 1, step)
    out = []

    for s in starts:
        seg = data[:, s:s + n_win]
        # nperseg = full window: 2 s gives 0.5 Hz resolution, enough to
        # separate delta from theta.
        freqs, psd = welch(seg, fs=sf, nperseg=n_win, axis=-1)

        powers = np.stack([
            psd[:, (freqs >= lo) & (freqs < hi)].sum(axis=-1)
            for lo, hi in BANDS.values()
        ], axis=-1)                            # (19, 5)

        powers = np.clip(powers, 1e-20, None)
        log_p = np.log(powers)

        # Centered log-ratio. Plain relative power sums to 1, so the five
        # bands are perfectly collinear -- one is always redundant. CLR
        # subtracts the per-channel mean of the logs instead, which carries
        # the same information without the forced dependency.
        clr = log_p - log_p.mean(axis=-1, keepdims=True)

        out.append(clr.ravel())                # (95,)

    return np.array(out)


# ------------------------------------------- binarize: 95 dims -> 380 bits

def fit_thresholds(train_features, n_bits=4):
    """Quartile cut points per feature, learned from TRAINING SUBJECTS ONLY.

    Returns (95, n_bits). Fit this once on your training split and reuse it
    for val/test -- refitting per subject leaks.
    """
    qs = np.linspace(0, 100, n_bits + 2)[1:-1]     # 20,40,60,80 for n_bits=4
    return np.percentile(train_features, qs, axis=0).T


def thermometer(features, thresholds):
    """(n_windows, 95) -> (n_windows, 380) of +/-1.

    Bit i is +1 if the value exceeds threshold i, so bits fill in order and
    similar values stay close in Hamming distance -- which is what the
    Hopfield overlap actually measures.
    """
    bits = features[:, :, None] > thresholds[None, :, :]   # (n, 95, 4)
    return np.where(bits, 1, -1).reshape(len(features), -1).astype(np.int8)


# ------------------------------------------------------- event handling

def clean_eyes_closed_spans(json_path, total_sec, guard_sec=15.0):
    """Eyes-closed spans with photic / artifact / movement windows removed.

    Only needed if you want to restrict the IDP features to resting segments.
    Timestamps in the JSON are SAMPLE INDICES at 200 Hz, not milliseconds.
    """
    ev = [(t / SFREQ_EVENTS, lab) for t, lab in json.load(open(json_path))]

    photic, on = [], None
    for t, lab in ev:
        if lab.startswith("Photic On"):
            on = t
        elif lab == "Photic Off" and on is not None:
            photic.append((on, t)); on = None

    dirty = photic + [(t, t + 5.0) for t, lab in ev
                      if lab in ("artifact", "Move", "swallowing")]

    spans, start = [], None
    for t, lab in ev:
        if lab == "Eyes Closed" and start is None:
            start = t
        elif lab in ("Eyes Open", "Paused") and start is not None:
            spans.append((start, t)); start = None
    if start is not None:
        spans.append((start, total_sec))

    clean = []
    for a, b in spans:
        cuts = sorted([(max(a, x), min(b, y)) for x, y in dirty
                       if x < b and y > a])
        cursor = a
        for x, y in cuts:
            if x - cursor > 1.0:
                clean.append((cursor, x))
            cursor = max(cursor, y)
        if b - cursor > 1.0:
            clean.append((cursor, b))
    return clean


# ------------------------------------------------------------------ demo

if __name__ == "__main__":
    edf = "C:/Users/u28jk/OneDrive/Desktop/caueeg-dataset/signal/edf/00001.edf"
    evj = "C:/Users/u28jk/OneDrive/Desktop/caueeg-dataset/event/00001.json"

    raw, n_rej = load_clean(edf)
    print(f"cleaned: {len(raw.ch_names)} ch, {raw.times[-1]:.0f}s, "
          f"{n_rej} ICA components removed")

    feats = band_features(raw)
    print(f"features: {feats.shape}   (windows, 19 ch x 5 bands)")

    spans = clean_eyes_closed_spans(evj, raw.times[-1])
    print(f"clean eyes-closed: {len(spans)} spans, "
          f"{sum(b - a for a, b in spans):.0f}s total")

    # NOTE: fitting thresholds on this one file is a placeholder so the demo
    # runs. In the real pipeline, fit on the pooled training split.
    thr = fit_thresholds(feats)
    bits = thermometer(feats, thr)
    print(f"binary:   {bits.shape}   values {np.unique(bits)}")

    # Sanity: adjacent windows overlap 50%, so they should be far more
    # similar than random windows. If this gap is small, something upstream
    # is broken.
    adj = np.mean([np.mean(bits[i] == bits[i + 1]) for i in range(len(bits) - 1)])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(bits), (500, 2))
    far = np.mean([np.mean(bits[i] == bits[j]) for i, j in idx])
    print(f"bit agreement: adjacent {adj:.3f} vs random {far:.3f}")



np.save("C:/Users/u28jk/Downloads/bits.npy", bits)
print("saved bits.npy")
    
