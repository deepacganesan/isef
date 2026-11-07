"""Python port of the Mathematica IDP (basicidp.ipynb) prototype."""
import csv

import numpy as np
import matplotlib.pyplot as plt

N = 784
DS_PATH = "basic-memory-ds/mnist_processed.csv"


def load_patterns(path=DS_PATH):
    with open(path) as f:
        rows = list(csv.reader(f))
    return np.array(rows, dtype=float)  # (10, 784), values in {-1, 1}


def psi(x):
    return np.tanh(5 * x)


def distort(pattern, percent):
    """pattern: length-N array in {-1, 1}. Randomizes `percent` fraction of entries."""
    data = pattern.copy()
    amount = round(percent * N)
    idx = np.random.choice(N, size=amount, replace=False)
    data[idx] = np.random.choice([-1.0, 1.0], size=amount)
    return data


def build_weights(ds, flat_img, num_patterns=None):
    """num_patterns=None uses every stored pattern; pass 2 to match the
    notebook's `{l, 1, 2}` literally."""
    if num_patterns is None:
        num_patterns = ds.shape[0]
    ds_used = ds[:num_patterns]
    alpha = (ds_used @ flat_img) ** 2  # (num_patterns,)
    w = (ds_used.T * alpha) @ ds_used  # sum_l alpha[l] * outer(ds[l], ds[l])
    np.fill_diagonal(w, 0)
    w /= N
    return w


def euler_integrate(w, x0, dt=0.1, iters=50):
    x = x0.copy()
    for _ in range(iters):
        x = x + dt * (-x + w @ psi(x))
    return x


def classify(pattern, ds):
    scores = ds @ pattern
    return int(np.argmax(scores)), scores


def to_img(ans):
    return np.where(ans >= 0, 1.0, 0.0).reshape(28, 28)


def random_start_recall(w, flat_img, rng, dt=0.1, iters=50):
    x0 = rng.normal(0, 0.01, N)
    settled_state = euler_integrate(w, x0, dt=dt, iters=iters)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, img, title in zip(
        axes,
        [flat_img, x0, settled_state],
        ["Initial Memory With Distortion", "Random Starting Point x0", "Recalled Memory"],
    ):
        ax.imshow(to_img(img), cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    rng = np.random.default_rng(0)
    ds = load_patterns()

    target_digit = 0
    clean = ds[target_digit]
    corrupted = distort(clean, percent=0.4)

    w = build_weights(ds, corrupted)
    recovered = euler_integrate(w, corrupted)

    pred, scores = classify(recovered, ds)
    print(f"target digit: {target_digit}, recovered classifies as: {pred}")
    print("scores per class:", np.round(scores, 2))

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, img, title in zip(
        axes,
        [clean, corrupted, recovered],
        ["clean", "corrupted (40%)", "recovered"],
    ):
        ax.imshow(img.reshape(28, 28), cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    random_start_recall(w, corrupted, rng)


if __name__ == "__main__":
    main()
