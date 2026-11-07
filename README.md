# isef

Code for the ISEF project (Deepac Ganesan, Jay Kim, Oliver Gao).
Data: CAUEEG (Kim, Youn & Paik 2023). The raw data is **not** in this repo. The data use
terms don't allow redistributing it, and `.gitignore` keeps EEG files and anything derived
from them out of git.

## `eeg_pipeline/`: CAUEEG → binary patterns (run in this order)

| Step | Script | What it does |
|---|---|---|
| 1 | `caueeg_preprocess.py` | EDF + event JSON → per-epoch band power. Eyes-closed rest only (first 60 s of each closure), photic session reserved for CSD, HV/artifacts/drowsiness skipped, 60 Hz notch, 0.5–45 Hz, ICA (eye/heart/muscle), 2-s Hann epochs with 50% overlap |
| 2 | `summarize_features.py` | QC + theta/alpha sanity table per recording, joined with diagnoses |
| 3 | `make_binary.py` (uses `thermometer.py`) | Thermometer coding with healthy-reference thresholds → +1/−1 matrix per recording (epochs × 380) |
| – | `show_binary.py` | Draw binary matrices |
| – | `thermometer_test.py` | Compares the plan's coding (% of own max) with healthy-reference coding |
| – | `epoch_timeline_fig.py` | Figure: which parts of a recording became epochs |

```
pip install mne python-picard matplotlib scipy numpy
python eeg_pipeline/caueeg_preprocess.py --edf-dir caueeg-dataset/signal/edf --event-dir caueeg-dataset/event --out features --jobs 8 --qc 30
python eeg_pipeline/summarize_features.py --features features --annotation caueeg-dataset/annotation.json --out features/summary.csv
python eeg_pipeline/make_binary.py --features features --annotation caueeg-dataset/annotation.json --out binary --healthy-ids <training half of healthy ids>
```

### CAUEEG gotchas this code handles
- **Pauses are cut out of the EDF, but event times still count them.** Every event after a pause is shifted by the paused time. The fix is checked against the photic channel (14% → 100% overlap with real flashes). `Patient Disconnected` behaves the same way.
- Event times lag the actual flashes by about 0.1 s. For CSD, time flashes from the photic channel; `photic_offset_s` in the QC table reports the lag.
- No EOG channel: Fp1/Fp2 serve as the eye reference, limited to 1–5 Hz with a front-heavy scalp-map check. Without that, posterior alpha gets removed as an "eye" component.
- Only ~13% of cb_normal recordings have photic stimulation (vs ~93% of MCI/AD).
- Long eyes-closed stretches show drowsiness (theta/alpha rises in 10/13 test recordings), hence the 60 s limit (`MAX_EC_S`).

## `simulations/`: model validation (no real data)
- Experiment 1: `hopfield_ews.py` → `make_figure.py`. Classical Hopfield under synaptic loss, early-warning signals before collapse.
- Experiment 2: `idp_ews.py` → `kramers_noise_corrected.py` → `make_idp_figure.py`. IDP network: basin radius calibration curve, Kramers vs simulated escapes, photic-style recovery time (CSD), agreement of the two halves.
