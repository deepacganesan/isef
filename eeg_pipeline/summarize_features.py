"""Per-recording QC + spectral sanity table, joined with CAUEEG diagnoses.

Usage: python summarize_features.py --features features --annotation caueeg-dataset/annotation.json
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def group_of(symptoms):
    s = set(symptoms)
    if "normal" in s:
        return "normal-smi" if "smi" in s else "normal"
    if "ad" in s:
        return "AD"
    if "mci" in s:
        return "MCI"
    if "vd" in s:
        return "VaD"
    return "/".join(sorted(s))[:18]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="features")
    p.add_argument("--annotation", required=True)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    ann = {d["serial"]: d for d in json.load(open(a.annotation))["data"]}
    qc_rows = {r["id"]: r for r in csv.DictReader(open(Path(a.features) / "qc_summary.csv"))}
    order = {"normal": 0, "normal-smi": 1, "MCI": 2, "AD": 3}
    rows = []
    for f in sorted(Path(a.features).glob("*.npz")):
        rid = f.stem
        d = np.load(f)
        qc = json.loads(str(d["qc"]))
        rel = d["rel_power"]
        ch = list(d["ch_names"])
        post = [ch.index(c) for c in ("O1", "O2", "P3", "P4", "Pz")]
        info = ann.get(rid, {"age": None, "symptom": ["?"]})
        r = dict(id=rid, group=group_of(info["symptom"]), age=info["age"],
                 epochs=qc["epochs_kept"], clean_s=qc["clean_ec_s"],
                 ica_removed=len([x for x in qc["ica_removed"].split(";") if x]),
                 eye=qc["ica_eye"], heart=qc["ica_heart"], muscle=qc["ica_muscle"],
                 bad_ch=qc["bad_channels"], align_in=qc.get("photic_align_in"),
                 align_out=qc.get("photic_align_out"))
        if len(rel):
            m = rel[:, post].mean(axis=(0, 1))        # posterior mean relative power per band
            r.update(delta=m[0], theta=m[1], alpha=m[2], beta=m[3], gamma=m[4],
                     theta_alpha=m[1] / m[2])
        rows.append(r)
    rows.sort(key=lambda r: (order.get(r["group"], 9), r["id"]))
    hdr = (f"{'id':<6}{'group':<19}{'age':>4}{'epochs':>7}{'ICA':>4} {'eye':<7}{'heart':<6}"
           f"{'muscle':<9}{'badch':<8}{'flash in/out':>13}{'delta':>7}{'theta':>7}{'alpha':>7}"
           f"{'beta':>7}{'θ/α':>6}")
    print(hdr)
    for r in rows:
        al = (f"{r['align_in']:.0%}/{r['align_out']:.0%}"
              if r["align_in"] not in (None, "", "nan") and r["align_in"] == r["align_in"] else "  -")
        sp = (f"{r['delta']:7.2f}{r['theta']:7.2f}{r['alpha']:7.2f}{r['beta']:7.2f}{r['theta_alpha']:6.2f}"
              if "alpha" in r else "   (no epochs)")
        print(f"{r['id']:<6}{r['group']:<19}{str(r['age']):>4}{r['epochs']:>7}{r['ica_removed']:>4} "
              f"{r['eye']:<7}{r['heart']:<6}{r['muscle']:<9}{r['bad_ch'][:7]:<8}{al:>13}{sp}")
    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
