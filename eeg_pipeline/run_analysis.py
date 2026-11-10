"""
Run the pre-registered analysis (analysis_plan.md) on the fixed sample (sample_150.json), once.

  EDF + events -> features -> sample with replacement rule -> thermometer thresholds and
  W_HC from the 50 TRAINING controls -> each test person's own prototypes -> basin radius R,
  depth dV -> tests -> results.txt, results.csv, results.png

Usage:
  python run_analysis.py --edf-dir caueeg-dataset/signal/edf --event-dir caueeg-dataset/event \
      --annotation caueeg-dataset/annotation.json --sample sample_150.json --work run150 --jobs 8
"""
import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.stats import mannwhitneyu, chi2

import caueeg_preprocess as PRE
import thermometer as TM
import build_weights as BW
import basin_radius as BR

BANDS = ["delta", "theta", "alpha", "beta"]
K_BITS, K_PROTO, MIN_EPOCHS = 5, 8, 60
POST = ["O1", "O2", "P3", "P4", "Pz"]


def _pre(args):
    rid, edf_dir, event_dir, out = args
    if (Path(out) / f"{rid}.npz").exists():
        return rid, "cached"
    try:
        PRE.process(rid, edf_dir, event_dir, out)
        return rid, "ok"
    except Exception as e:
        return rid, f"error: {e}"


def logit_ll(X, y):
    X = np.column_stack([np.ones(len(y)), X])
    f = lambda b: -np.sum(y * (X @ b) - np.logaddexp(0, X @ b))
    r = minimize(f, np.zeros(X.shape[1]), method="BFGS")
    return -r.fun, r.x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edf-dir", required=True)
    ap.add_argument("--event-dir", required=True)
    ap.add_argument("--annotation", required=True)
    ap.add_argument("--sample", default="sample_150.json")
    ap.add_argument("--work", default="run150")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    W = Path(a.work); (W / "features").mkdir(parents=True, exist_ok=True)
    S = json.load(open(a.sample))
    ann = {d["serial"]: d for d in json.load(open(a.annotation))["data"]}
    pairs = S["pairs"] + S["pair_backups"]
    train_order = S["hc_train"] + S["hc_train_backup"]
    need = sorted({i for p in pairs for i in p} | set(train_order))

    # 1. preprocessing
    with ProcessPoolExecutor(a.jobs) as ex:
        for rid, msg in ex.map(_pre, [(i, a.edf_dir, a.event_dir, W / "features") for i in need]):
            print(f"preprocess {rid}: {msg}", flush=True)
    feat = {}
    for i in need:
        f = W / "features" / f"{i}.npz"
        if f.exists():
            d = np.load(f)
            feat[i] = dict(rel=TM.select(d["rel_power"], d["bands"], BANDS), ch=list(d["ch_names"]))
    ok = lambda i: i in feat and len(feat[i]["rel"]) >= MIN_EPOCHS

    # 2. replacement rule (data quality only)
    n_pairs, n_train = len(S["pairs"]), len(S["hc_train"])          # backups only replace failures
    used_pairs = [p for p in pairs if ok(p[0]) and ok(p[1])][:n_pairs]
    train = [i for i in train_order if ok(i)][:n_train]
    ad = [p[0] for p in used_pairs]; hc = [p[1] for p in used_pairs]
    log = [f"pairs used: {len(used_pairs)} (needed {n_pairs}); training controls: {len(train)} (needed {n_train})",
           f"replaced pairs: {[p for p in S['pairs'] if p not in used_pairs]}"]

    # 3. thresholds + codes + healthy model from TRAINING controls only
    th = TM.fit_reference_thresholds([feat[i]["rel"] for i in train], K=K_BITS, per_subject=100)
    codes = {i: TM.encode_ref(feat[i]["rel"], th) for i in train + ad + hc}
    H = BW.build_healthy(codes, train, K_PROTO)
    np.savez(W / "W_HC.npz", **H, train_ids=np.array(train))

    # 4. per person: own prototypes -> R, dV ; theta/alpha baseline
    ch = feat[ad[0]]["ch"]; post = [ch.index(c) for c in POST]
    rows = []
    for grp, ids in (("HC", hc), ("AD", ad)):
        for i in ids:
            P = BW.build_patient(codes[i], H["mean_vector"], K_PROTO)
            R, dV, Rs, ov, dV0 = BR.person(P["prototypes"], np.random.default_rng(int(i)))
            np.savez(W / f"W_{i}.npz", W=P["W"], prototypes=P["prototypes"], cluster_sizes=P["cluster_sizes"])
            rel = feat[i]["rel"][:, post].mean(1)
            rows.append(dict(id=i, group=grp, age=ann[i]["age"], epochs=len(feat[i]["rel"]), R=R, dV=dV, dV0=dV0,
                             overlap=ov, theta_alpha=float(rel[:, 1].mean() / rel[:, 2].mean())))
            print(f"{grp} {i}: R {R:.3f}  dV {dV:.3f}  θ/α {rows[-1]['theta_alpha']:.2f}", flush=True)
    with open(W / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # 5. pre-registered tests
    get = lambda g, k: np.array([r[k] for r in rows if r["group"] == g])
    out = log + [""]
    for k, name in (("R", "basin radius R (primary)"), ("dV", "depth dV (secondary)"), ("theta_alpha", "theta/alpha (baseline)")):
        x_ad, x_hc = get("AD", k), get("HC", k)
        alt = "greater" if k == "theta_alpha" else "less"
        u = mannwhitneyu(x_ad, x_hc, alternative=alt)
        rb = 2 * u.statistic / (len(x_ad) * len(x_hc)) - 1
        out.append(f"{name}: AD median {np.median(x_ad):.3f} vs HC {np.median(x_hc):.3f}; "
                   f"one-sided Mann-Whitney ({'AD ' + ('<' if alt == 'less' else '>') + ' HC'}) p = {u.pvalue:.4f}; rank-biserial r = {rb:+.2f}")
    y = np.array([1.0 if r["group"] == "AD" else 0.0 for r in rows])
    z = lambda v: (v - v.mean()) / v.std()
    ta = z(np.log(np.array([r["theta_alpha"] for r in rows]))); Rv = z(np.array([r["R"] for r in rows]))
    ll0, _ = logit_ll(ta[:, None], y); ll1, b = logit_ll(np.column_stack([ta, Rv]), y)
    lr = 2 * (ll1 - ll0)
    out.append(f"added value of R beyond theta/alpha: likelihood-ratio = {lr:.2f}, p = {chi2.sf(lr, 1):.4f}; "
               f"coefficient of R (standardized) = {b[2]:+.2f}")
    open(W / "results.txt", "w").write("\n".join(out) + "\n")
    print("\n".join(out))

    # 6. figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), facecolor="#fcfcfb")
    for ax, (k, lab) in zip(axes, (("R", "basin radius R"), ("dV", "valley depth ΔV"), ("theta_alpha", "theta/alpha"))):
        for gi, (g, c) in enumerate((("HC", "#2a78d6"), ("AD", "#eb6834"))):
            v = get(g, k); jit = np.random.default_rng(gi).uniform(-.12, .12, len(v))
            ax.scatter(gi + jit, v, s=22, color=c, alpha=.75, edgecolor="none")
            ax.plot([gi - .25, gi + .25], [np.median(v)] * 2, color="#0b0b0b", lw=2)
        ax.set_xticks([0, 1]); ax.set_xticklabels([f"healthy (n={len(hc)})", f"AD (n={len(ad)})"])
        ax.set_title(lab, loc="left"); ax.set_facecolor("#fcfcfb"); ax.spines[["top", "right"]].set_visible(False)
        if k == "theta_alpha": ax.set_yscale("log")
    fig.tight_layout(); fig.savefig(W / "results.png", dpi=160, facecolor="#fcfcfb")
    import compare_weights
    compare_weights.figure({g: [W / f"W_{i}.npz" for i in ids] for g, ids in (("healthy", hc), ("AD", ad))},
                           W / "weights_compared.png")


if __name__ == "__main__":
    main()
