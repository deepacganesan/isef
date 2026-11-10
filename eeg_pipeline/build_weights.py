"""
Binary patterns -> prototypes -> Hebbian weight matrices (research plan, in order).

Healthy model W_HC
  1. take every healthy recording's +/-1 codes (epochs x N)
  2. subsample each healthy person to the same number of windows
  3. pool them and compute the healthy mean vector  m  (one value per bit)
  4. subtract m from every window                     (deviation from typical)
  5. k-means with K clusters  -> K centroids          (typical healthy brain states)
  6. scale each centroid to unit length               -> prototypes xi_1..xi_K
  7. Hebbian rule  W = sum_mu xi_mu xi_mu^T, zero diagonal

Each patient
  same steps 4-7 on that person's own windows.
  --center healthy (plan): subtract the healthy mean m. Then every centroid carries the
      person's average offset from healthy, so prototype overlap mostly measures that
      offset (rank corr +0.92 in the test set).
  --center own: subtract the person's own mean. Prototypes then describe the person's
      repertoire of states; the offset is saved separately as `offset`.

Usage:
  python build_weights.py --binary binary --annotation caueeg-dataset/annotation.json --out weights --K 8

Writes weights/W_HC.npz and weights/<id>.npz with W, prototypes (K x N, unit length),
prototype_signs (K x N, +/-1, for bit-flip basin tests), cluster_sizes, mean_vector.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans

MIN_EPOCHS = 60


def relevant(symptoms):
    """Only the groups the study compares: healthy, MCI (not vascular), Alzheimer's."""
    s = set(symptoms)
    if "normal" in s or "ad" in s:
        return True
    return "mci" in s and "mci_vascular" not in s


def healthy_mean(codes_list, n_per, seed=0):
    rng = np.random.default_rng(seed)
    pooled = [c[np.sort(rng.choice(len(c), n_per, replace=False))] if len(c) > n_per else c
              for c in codes_list]
    pooled = np.concatenate(pooled).astype(float)
    return pooled.mean(0), pooled


def prototypes(X_centered, K, seed=0):
    km = KMeans(n_clusters=K, n_init=20, random_state=seed).fit(X_centered)
    C = km.cluster_centers_
    norms = np.linalg.norm(C, axis=1, keepdims=True)
    P = C / np.where(norms > 0, norms, 1)                     # unit length
    sizes = np.bincount(km.labels_, minlength=K)
    order = np.argsort(-sizes)                                # most common state first
    return P[order], sizes[order], km.labels_


def hebbian(P):
    W = P.T @ P                                               # sum of outer products
    np.fill_diagonal(W, 0.0)                                  # no self-connections
    return W


def build_healthy(codes_by_id, healthy_ids, K, n_per=None, seed=0):
    lst = [codes_by_id[h] for h in healthy_ids]
    n_per = n_per or min(len(c) for c in lst)
    m, pooled = healthy_mean(lst, n_per, seed)
    P, sizes, _ = prototypes(pooled - m, K, seed)
    return dict(W=hebbian(P), prototypes=P, prototype_signs=np.sign(P + 1e-12),
                cluster_sizes=sizes, mean_vector=m, n_per_subject=n_per)


def build_patient(codes, m, K, seed=0, center="healthy"):
    X = codes.astype(float)
    own = X.mean(0)
    P, sizes, _ = prototypes(X - (m if center == "healthy" else own), K, seed)
    return dict(W=hebbian(P), prototypes=P, prototype_signs=np.sign(P + 1e-12),
                cluster_sizes=sizes, own_mean=own,
                offset=float(np.linalg.norm(own - m) / np.sqrt(X.shape[1])))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--binary", default="binary")
    p.add_argument("--annotation", required=True)
    p.add_argument("--out", default="weights")
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--center", choices=["healthy", "own"], default="healthy")
    p.add_argument("--healthy-ids", nargs="*", help="default: all normals with >= 60 epochs")
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ann = {d["serial"]: d for d in json.load(open(a.annotation))["data"]}

    codes, names = {}, None
    for f in sorted(Path(a.binary).glob("*.npz")):
        if f.stem == "thresholds":
            continue
        d = np.load(f)
        if len(d["codes"]) >= MIN_EPOCHS and relevant(ann[f.stem]["symptom"]):
            codes[f.stem] = d["codes"]
            names = d["feature_names"]
    healthy = a.healthy_ids or [k for k in codes if "normal" in ann[k]["symptom"]]

    H = build_healthy(codes, healthy, a.K)
    np.savez(out / "W_HC.npz", **H, healthy_ids=np.array(healthy), feature_names=names, K=a.K)
    print(f"W_HC: {len(healthy)} healthy x {H['n_per_subject']} windows -> K={a.K} prototypes, "
          f"W {H['W'].shape}, cluster sizes {H['cluster_sizes'].tolist()}")
    for k, c in codes.items():
        P = build_patient(c, H["mean_vector"], a.K, center=a.center)
        np.savez(out / f"{k}.npz", **P, symptom=np.array(ann[k]["symptom"]), age=ann[k]["age"],
                 n_windows=len(c), K=a.K, center=a.center)
        print(f"{k}: {len(c):>4} windows -> W {P['W'].shape}, cluster sizes {P['cluster_sizes'].tolist()}"
              f"   {'/'.join(ann[k]['symptom'][:2])}")


if __name__ == "__main__":
    main()
