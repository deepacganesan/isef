"""
Basin radius R and valley depth dV for each person's own IDP network (research plan:
'Representation of Individual Alzheimer Patients' + basin radius).

Per person: their K prototypes (from build_weights.py, healthy-mean centered), turned into
+/-1 patterns. IDP retrieval as in simulations/idp_ews.py:
    cue s       = a stored pattern with a fraction f of its bits flipped
    saliency    = alpha_nu = gamma * max(0, xi_nu . s / N)      (the cue reshapes W)
    W(s)        = sum_nu alpha_nu xi_nu xi_nu^T / N,  zero diagonal
    dynamics    dx/dt = -x + W(s) tanh(x),  start x0 = 0.05 * s  (near the confusion state)
    success     cos(x_final, target) > 0.9 AND overlap tanh(x).target/N > 0.1
R_mu = largest f with >= 50% success; person R = mean over prototypes.
dV_mu (research plan definition) = E(rim) - E(bottom): settle from the clean pattern to x*, then
       flip the same fraction R_mu of x*'s coordinates (the edge of the basin) and measure how much
       the energy rose, per neuron, averaged over trials. Uses the landscape W(xi_mu) of the clean cue.
dV0_mu (exploratory) = [E(0) - E(x*)] / N, depth measured down from the confusion state.

Usage: python basin_radius.py --weights weights --out basin.csv [--ids ...]
"""
import argparse
import csv
from pathlib import Path
import numpy as np

GAMMA = 2.0
F_GRID = np.round(np.arange(0.025, 0.5001, 0.025), 3)
TRIALS = 20
DT, T = 0.05, 60.0


def run(Xi, A, X0):
    """Integrate a batch. Xi: (K, N) +/-1, A: (B, K) saliencies, X0: (B, N)."""
    N = Xi.shape[1]
    X = X0.copy()
    diag = (A[:, :, None] * Xi[None] ** 2).sum(1) / N          # self-connection to remove
    for _ in range(int(T / DT)):
        V = np.tanh(X)
        X += DT * (-X + ((V @ Xi.T) * A) @ Xi / N - diag * V)
    return X


def energy(X, Xi, a):
    N = Xi.shape[1]
    V = np.tanh(X)
    quad = ((V @ Xi.T) ** 2 * a).sum(-1) / N - ((V ** 2) * (a @ Xi ** 2) / N).sum(-1)
    G = (X * V - np.logaddexp(X, -X) + np.log(2)).sum(-1)
    return (-.5 * quad + G) / N


def person(prototypes, rng):
    Xi = np.where(prototypes >= 0, 1.0, -1.0)
    K, N = Xi.shape
    sal = lambda S: GAMMA * np.maximum(0.0, S @ Xi.T / N)
    R, dV, dV0 = [], [], []
    for mu in range(K):
        r = 0.0
        for f in F_GRID:
            S = np.tile(Xi[mu], (TRIALS, 1))
            k = int(round(f * N))
            for t in range(TRIALS):
                S[t, rng.choice(N, k, replace=False)] *= -1
            X = run(Xi, sal(S), 0.05 * S)
            cos = (X @ Xi[mu]) / (np.linalg.norm(X, axis=1) * np.sqrt(N) + 1e-12)
            m = np.tanh(X) @ Xi[mu] / N                        # must actually sit in the valley,
            if np.mean((cos > 0.9) & (m > 0.1)) >= 0.5:       # not just point at it while decaying to 0
                r = f
            else:
                break
        R.append(r)
        a = sal(Xi[mu][None])[0]
        x = run(Xi, a[None], 0.05 * Xi[mu][None])
        e_bottom = energy(x, Xi, a)[0]
        dV0.append(float(energy(np.zeros((1, N)), Xi, a)[0] - e_bottom))
        k = int(round(r * N))
        if k == 0:
            dV.append(0.0)
        else:
            rim = np.tile(x[0], (TRIALS, 1))
            for t in range(TRIALS):
                rim[t, rng.choice(N, k, replace=False)] *= -1
            dV.append(float(np.mean(energy(rim, Xi, a)) - e_bottom))
    overlap = np.abs(Xi @ Xi.T / N)[np.triu_indices(K, 1)].mean() if K > 1 else 0.0
    return float(np.mean(R)), float(np.mean(dV)), R, float(overlap), float(np.mean(dV0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--out", default="basin.csv")
    ap.add_argument("--ids", nargs="*")
    a = ap.parse_args()
    files = [f for f in sorted(Path(a.weights).glob("*.npz")) if f.stem != "W_HC"
             and (not a.ids or f.stem in a.ids)]
    rows = []
    for f in files:
        d = np.load(f)
        R, dV, Rs, ov, dV0 = person(d["prototypes"], np.random.default_rng(int(f.stem)))
        rows.append(dict(id=f.stem, R=round(R, 4), dV=round(dV, 5), dV0=round(dV0, 5), overlap=round(ov, 3),
                         R_per_prototype=";".join(f"{x:.3f}" for x in Rs)))
        print(f"{f.stem}: R = {R:.3f}  dV = {dV:.4f}  prototype overlap = {ov:.2f}   per prototype {Rs}", flush=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
