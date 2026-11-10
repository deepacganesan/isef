# isef

Code for the ISEF project (Deepac Ganesan, Jay Kim, Oliver Gao).
Data: CAUEEG (Kim, Youn & Paik 2023). The raw data is **not** in this repo. The data use
terms don't allow redistributing it, and `.gitignore` keeps EEG files and anything derived
from them out of git.

## `eeg_pipeline/`: CAUEEG → binary patterns → weight matrices → basins (run in this order)

| Step | Script | What it does |
|---|---|---|
| 1 | `caueeg_preprocess.py` | EDF + event JSON → per-epoch band power. Eyes-closed rest only (first 60 s of each closure), photic session reserved for CSD, HV/artifacts/drowsiness skipped, 60 Hz notch, 0.5–45 Hz, ICA (eye/heart/muscle), 2-s Hann epochs with 50% overlap |
| 2 | `summarize_features.py` | QC + theta/alpha sanity table per recording, joined with diagnoses |
| 3 | `make_binary.py` (uses `thermometer.py`) | Thermometer coding with healthy-reference thresholds → +1/−1 matrix per recording (epochs × 380) |
| 4 | `build_weights.py` | Healthy mean → k-means (K=8) → unit prototypes → Hebbian W, for the healthy model (W_HC) and each patient. `--center healthy` (plan) or `own` |
| 5 | `idp_weights.py` | Input-driven version: each EEG window's saliencies α for the healthy memories, so W(u) = Σ α ξξᵀ per window |
| 6 | `basin_radius.py` | Basin radius R and depth ΔV (rim − bottom) of each person's own IDP network. Checked against theory: R = ½(1 − 1/γ) |
| ★ | `run_analysis.py` | Runs steps 1–6 plus the pre-registered tests on a fixed sample (`analysis/sample_*.json`) |
| – | `show_binary.py` | Draw binary matrices |
| – | `show_weights.py`, `compare_weights.py` | Weight matrices, prototype overlap, eigenvalue split by group |
| – | `show_idp.py`, `explain_idp.py` | How EEG windows drive the healthy memories over time |
| – | `landscape_3d.py`, `landscape_own.py` | 3D energy landscapes (IDP drive; each person's own network) |
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

## `analysis/`: pre-registered test, AD vs healthy
- `analysis_plan.md`: hypotheses, settings, and tests, written down before running. Changes are dated and were all made before results.
- `sample_150.json` / `sample_75.json`: the fixed, random, age-matched samples (recording IDs only).
- `run75/`: results for 25 AD vs 25 age-matched healthy (healthy model from 25 separate controls).
  - θ/α separates the groups (p = 0.0015).
  - The basin-radius hypothesis (AD < healthy) was **not** supported: AD basins were larger (median R 0.225 vs 0.134) and deeper (ΔV 1.21 vs 0.73).
  - The AD networks' memories overlap more and merge into one broad valley (`weights_compared.png`, `landscape_own.png`).
  - Post-hoc idea to test next with its own pre-registration: AD is past the tipping point (one deep merged state), so the shallowest basins should be in MCI.
- Per-person result tables are not committed (derived patient data).

```
python eeg_pipeline/run_analysis.py --edf-dir caueeg-dataset/signal/edf --event-dir caueeg-dataset/event --annotation caueeg-dataset/annotation.json --sample analysis/sample_75.json --work run75 --jobs 8
```

## `simulations/`: model validation (no real data)
- Experiment 1: `hopfield_ews.py` → `make_figure.py`. Classical Hopfield under synaptic loss, early-warning signals before collapse.
- Experiment 2: `idp_ews.py` → `kramers_noise_corrected.py` → `make_idp_figure.py`. IDP network: basin radius calibration curve, Kramers vs simulated escapes, photic-style recovery time (CSD), agreement of the two halves.
