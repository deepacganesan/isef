"""
Input-Driven Plasticity (IDP) weight matrices: every EEG window gets its own W(u).

    W(u)      = sum_mu  alpha_mu(u) * xi_mu xi_mu^T          (zero diagonal)
    alpha_mu  = gamma * max(0, xi_mu . u_hat)                (saliency)
    memory mu exists (is a valley) only while alpha_mu > 1  <=>  xi_mu . u_hat > 1/gamma

  xi_mu   the K healthy prototypes (unit length) from the healthy model
  u_hat   one window's +/-1 code, centered with the healthy mean, scaled to length 1
  gamma   input gain. Its value is a modeling choice (the plan notes the critical point
          depends on scaling), so results are shown across the threshold theta = 1/gamma.

The stored memories never change; what changes from window to window is which of them
the current brain state "lights up". A person's IDP model is therefore the healthy
prototypes plus one saliency vector per window (windows x K). W(u) for any window is
rebuilt from those with W_of(...). Healthy people are scored against a healthy model
built WITHOUT them (leave-one-out).

Usage: python idp_weights.py --binary binary --weights weights --annotation annotation.json --out idp
"""
import argparse
import json
from pathlib import Path
import numpy as np
import build_weights as BW


def unit_inputs(codes, m):
    U = codes.astype(float) - m
    return U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)


def saliency_cos(P, U_hat):
    """xi_mu . u_hat for every window and memory: (windows, K). alpha = gamma * max(0, this)."""
    return U_hat @ P.T


def W_of(P, cos_row, gamma):
    alpha = gamma * np.maximum(0.0, cos_row)
    W = (P.T * alpha) @ P
    np.fill_diagonal(W, 0.0)
    return W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="binary")
    ap.add_argument("--weights", default="weights", help="folder with W_HC.npz from build_weights.py")
    ap.add_argument("--annotation", required=True)
    ap.add_argument("--out", default="idp")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ann = {d["serial"]: d for d in json.load(open(a.annotation))["data"]}
    H = np.load(Path(a.weights) / "W_HC.npz")
    K, healthy, n_per = int(H["K"]), list(H["healthy_ids"]), int(H["n_per_subject"])
    codes = {f.stem: np.load(f)["codes"] for f in Path(a.binary).glob("*.npz") if f.stem != "thresholds"}
    for k, c in sorted(codes.items()):
        if len(c) < BW.MIN_EPOCHS or not BW.relevant(ann[k]["symptom"]):
            continue
        if k in healthy:                                       # leave-one-out healthy model
            ref = BW.build_healthy(codes, [h for h in healthy if h != k], K, n_per)
            P, m, loo = ref["prototypes"], ref["mean_vector"], True
        else:
            P, m, loo = H["prototypes"], H["mean_vector"], False
        S = saliency_cos(P, unit_inputs(c, m))
        np.savez(out / f"{k}.npz", saliency_cos=S.astype(np.float32), prototypes=P, mean_vector=m,
                 leave_one_out=loo, symptom=np.array(ann[k]["symptom"]), age=ann[k]["age"])
        print(f"{k}: {len(c):>4} windows x {K} memories   median best match {np.median(S.max(1)):.2f}"
              f"   {'/'.join(ann[k]['symptom'][:2])}{'  (leave-one-out)' if loo else ''}")


if __name__ == "__main__":
    main()
