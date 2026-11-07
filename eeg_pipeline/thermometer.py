"""
Thermometer coding: band-power epochs -> +1/-1 patterns for the Hopfield / IDP network.

Each feature (channel x band) gets K bits. Bit k is +1 if the value is at or above
threshold k, else -1, so a bigger value turns on more bits (like mercury in a
thermometer) and neighbouring values get neighbouring codes (unlike plain binary).

Two ways to set the thresholds:

  plan  (research plan version)
        value = power / that recording's own maximum for the feature
        thresholds at 0, 25, 50, 75, 100 % (K=5)
  ref   (recommended)
        thresholds = K evenly spaced percentiles of the HEALTHY reference
        distribution for that feature (k/(K+1), k = 1..K), the same thresholds for
        every person. A typical healthy epoch turns on about half the bits.

Pattern length N = channels x bands x K  (19 x 4 x 5 = 380 without gamma).

Usage as a library:
    th = fit_reference_thresholds(healthy_feature_list, K=5, per_subject=100)
    codes = encode_ref(features, th)          # (epochs, N) of +1/-1
"""
import numpy as np


def select(feat, bands_all, keep_bands):
    idx = [list(bands_all).index(b) for b in keep_bands]
    return feat[:, :, idx]                               # (epochs, channels, kept bands)


def subsample(feat, n, rng):
    """Equal number of epochs per subject so no one dominates the reference."""
    if len(feat) <= n:
        return feat
    return feat[np.sort(rng.choice(len(feat), n, replace=False))]


def fit_reference_thresholds(healthy_feats, K=5, per_subject=100, seed=0):
    """healthy_feats: list of (epochs, channels, bands) arrays, one per healthy subject.
    Returns thresholds (channels, bands, K)."""
    rng = np.random.default_rng(seed)
    pooled = np.concatenate([subsample(f, per_subject, rng) for f in healthy_feats])
    q = np.arange(1, K + 1) / (K + 1)
    return np.moveaxis(np.quantile(pooled, q, axis=0), 0, -1)


def encode_ref(feat, thresholds):
    bits = feat[..., None] >= thresholds[None]           # (epochs, ch, bands, K)
    return np.where(bits, 1, -1).reshape(len(feat), -1).astype(np.int8)


def encode_plan(feat, K=5):
    """Research-plan version: % of the recording's own max, thresholds 0..100%."""
    frac = feat / feat.max(axis=0, keepdims=True)
    thr = np.linspace(0, 1, K)                           # 0, .25, .5, .75, 1
    bits = frac[..., None] >= thr
    return np.where(bits, 1, -1).reshape(len(feat), -1).astype(np.int8)


def decode_level(codes, n_ch, n_bands, K):
    """Fraction of bits on per feature -> (epochs, ch, bands) in [0, 1]."""
    b = codes.reshape(len(codes), n_ch, n_bands, K)
    return (b > 0).mean(-1)
