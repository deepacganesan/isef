"""
Save the thermometer-coded (+1/-1) binary matrix for every recording.

Usage:
  python make_binary.py --features features --annotation caueeg-dataset/annotation.json --out binary

Writes
  binary/<id>.npz      codes (epochs x 380, int8 of +1/-1), onsets_s, feature_names
  binary/thresholds.npz  the healthy-reference thresholds (19 ch x 4 bands x 5 bits)

Feature order inside each 380-long pattern: channel -> band -> bit, e.g.
  Fp1_delta_b1 ... Fp1_delta_b5, Fp1_theta_b1, ... , Pz_beta_b5
The research plan writes the matrix as features x windows: use codes.T for that.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import thermometer as TM

BANDS = ["delta", "theta", "alpha", "beta"]
K = 5
MIN_EPOCHS = 60


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="features")
    p.add_argument("--annotation", required=True)
    p.add_argument("--out", default="binary")
    p.add_argument("--healthy-ids", nargs="*",
                   help="which healthy recordings set the thresholds (default: all normals "
                        "with enough epochs). For the real study, pass a training half only.")
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ann = {d["serial"]: d for d in json.load(open(a.annotation))["data"]}

    feats, ch = {}, None
    for f in sorted(Path(a.features).glob("*.npz")):
        d = np.load(f)
        if len(d["rel_power"]) == 0:
            continue
        feats[f.stem] = (TM.select(d["rel_power"], d["bands"], BANDS), d["onsets_s"])
        ch = list(d["ch_names"])

    healthy = a.healthy_ids or [k for k in feats if "normal" in ann[k]["symptom"]
                                and len(feats[k][0]) >= MIN_EPOCHS]
    th = TM.fit_reference_thresholds([feats[k][0] for k in healthy], K=K, per_subject=100)
    names = np.array([f"{c}_{b}_b{i + 1}" for c in ch for b in BANDS for i in range(K)])
    np.savez(out / "thresholds.npz", thresholds=th, ch_names=np.array(ch),
             bands=np.array(BANDS), healthy_ids=np.array(healthy))

    for k, (f, onsets) in feats.items():
        codes = TM.encode_ref(f, th)
        np.savez_compressed(out / f"{k}.npz", codes=codes, onsets_s=onsets,
                            feature_names=names, symptom=np.array(ann[k]["symptom"]),
                            age=ann[k]["age"], enough_epochs=len(codes) >= MIN_EPOCHS)
        print(f"{k}: {codes.shape[0]:>4} epochs x {codes.shape[1]} bits   "
              f"{'/'.join(ann[k]['symptom'][:2])}{'' if len(codes) >= MIN_EPOCHS else '   (< 60 epochs)'}")
    print(f"thresholds fit on {len(healthy)} healthy recordings: {' '.join(healthy)}")


if __name__ == "__main__":
    main()
