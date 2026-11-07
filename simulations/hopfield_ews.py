"""
Experiment 1: Do early-warning signals appear before memory collapse
in a Hopfield network undergoing progressive synaptic loss?

Model
  - Classical Hopfield network, N binary neurons (+1/-1), P random stored patterns
  - Hebbian weights W = (1/N) * sum_mu xi_mu xi_mu^T, zero diagonal
  - Damage = symmetric synaptic loss: a fixed random order of synapse pairs (i,j)
    is deleted progressively, so every damage level is the SAME network further
    along its own decline (a within-subject trajectory, like a single patient)

Per damage level we measure
  Visible symptom
    recall      : overlap reached from a cue with 10% corrupted bits (1.0 = perfect)
  Hidden / early-warning signals
    basin       : largest cue corruption that still recalls the memory
    stability   : mean aligned local field of the stored pattern, xi_i * h_i
    recovery    : sweeps needed to get back to the attractor after a small kick
    kick_fail   : fraction of small kicks the memory never recovers from
    variance    : variance of overlap m(t) under noisy (Glauber) dynamics
    ac1         : lag-1 autocorrelation of m(t) under noisy dynamics
  "Virtual EEG" versions (the bridge to real data)
    eeg_var, eeg_ac1 : same two statistics, but computed on a virtual electrode
                       = average activity of a random 20% of neurons. This signal
                       knows nothing about which memory is stored, like a real
                       scalp electrode.

Run:  python3 hopfield_ews.py      (writes results.npz, then run make_figure.py)
"""
import numpy as np
from numba import njit

# ---------------------------------------------------------------- parameters
N = 400             # neurons
P = 30              # stored patterns  -> load alpha = P/N = 0.075
SEEDS = 6           # independent networks ("patients")
DAMAGE = np.round(np.linspace(0.0, 0.92, 47), 3)   # fraction of synapses removed
TEST_PATTERNS = 8   # patterns probed per damage level
CUE_NOISE = 0.10    # corruption used for the recall test
BASIN_GRID = np.round(np.arange(0.02, 0.50, 0.02), 2)
BASIN_TRIALS = 4
KICK = 0.05         # perturbation size for the recovery test
T_NOISE = 0.15      # temperature for the stochastic (Glauber) traces
TRACE_SWEEPS = 300
BURN_IN = 50


# ---------------------------------------------------------------- dynamics
@njit(cache=True)
def relax(W, s, max_sweeps, seed):
    """Zero-temperature asynchronous dynamics until no neuron flips.
    Returns final state and number of sweeps used (fractional)."""
    np.random.seed(seed)
    n = s.shape[0]
    h = W @ s
    flips_total = 0
    for sweep in range(max_sweeps):
        order = np.random.permutation(n)
        flips = 0
        for k in range(n):
            i = order[k]
            new = 1.0 if h[i] >= 0 else -1.0
            if new != s[i]:
                h += 2.0 * new * W[:, i]
                s[i] = new
                flips += 1
        flips_total += flips
        if flips == 0:
            return s, sweep
    return s, max_sweeps


@njit(cache=True)
def recovery_time(W, s, xi, target_m, max_sweeps, seed):
    """Asynchronous T=0 dynamics after a kick; time (in sweeps, fractional)
    until overlap with xi gets back within 0.01 of target_m."""
    np.random.seed(seed)
    n = s.shape[0]
    h = W @ s
    m = (s @ xi) / n
    if m >= target_m - 0.01:
        return 0.0
    for sweep in range(max_sweeps):
        order = np.random.permutation(n)
        for k in range(n):
            i = order[k]
            new = 1.0 if h[i] >= 0 else -1.0
            if new != s[i]:
                h += 2.0 * new * W[:, i]
                m += 2.0 * new * xi[i] / n
                s[i] = new
                if m >= target_m - 0.01:
                    return sweep + (k + 1) / n
    return np.nan                              # never came back: kick knocked it out


@njit(cache=True)
def glauber_trace(W, s, xi, v, T, sweeps, seed):
    """Heat-bath dynamics at temperature T. Every sweep records the overlap
    with the memory and a virtual-electrode reading v . s"""
    np.random.seed(seed)
    n = s.shape[0]
    h = W @ s
    out = np.empty(sweeps)
    eeg = np.empty(sweeps)
    for sw in range(sweeps):
        for k in range(n):
            i = np.random.randint(n)
            p_up = 1.0 / (1.0 + np.exp(-2.0 * h[i] / T))
            new = 1.0 if np.random.random() < p_up else -1.0
            if new != s[i]:
                h += 2.0 * new * W[:, i]
                s[i] = new
        out[sw] = (s @ xi) / n
        eeg[sw] = v @ s
    return out, eeg


# ---------------------------------------------------------------- helpers
def corrupt(xi, frac, rng):
    s = xi.copy()
    idx = rng.choice(xi.size, int(round(frac * xi.size)), replace=False)
    s[idx] *= -1
    return s


def overlap(a, b):
    return float(a @ b) / a.size


def build_network(seed):
    rng = np.random.default_rng(seed)
    xi = rng.choice([-1.0, 1.0], size=(P, N))
    W_full = (xi.T @ xi) / N
    np.fill_diagonal(W_full, 0.0)
    iu, ju = np.triu_indices(N, k=1)
    order = rng.permutation(iu.size)          # fixed order synapses die in
    return xi, W_full, iu[order], ju[order]


def damaged(W_full, iu, ju, d):
    W = W_full.copy()
    k = int(round(d * iu.size))
    W[iu[:k], ju[:k]] = 0.0
    W[ju[:k], iu[:k]] = 0.0                   # symmetric loss keeps an energy function
    return W


def pooled_stats(traces):
    """Variance and lag-1 autocorrelation pooled over several traces (each
    demeaned separately). Pooling keeps the estimate defined even when a single
    healthy trace barely moves."""
    if not traces:
        return np.nan, np.nan
    num = den = 0.0
    n = 0
    for tr in traces:
        x = tr - tr.mean()
        num += float(x[:-1] @ x[1:])
        den += float(x @ x)
        n += x.size
    return den / n, (num / den if den > 0 else np.nan)


def measure(W, xi_all, v, rng):
    rec, basin, stab, rtime, kfail, var, ac1 = [], [], [], [], [], [], []
    m_traces, e_traces = [], []
    for mu in range(TEST_PATTERNS):
        xi = xi_all[mu]
        sd = int(rng.integers(1 << 30))

        # visible symptom: recall from a 10%-corrupted cue
        s, _ = relax(W, corrupt(xi, CUE_NOISE, rng), 50, sd)
        rec.append(overlap(s, xi))

        # stability of the stored pattern itself (local field margin)
        h = W @ xi
        stab.append(float(np.mean(xi * h)) / max(float(np.std(xi * h)), 1e-12))

        # basin radius: largest corruption with majority successful recall
        r = 0.0
        for f in BASIN_GRID:
            ok = 0
            for t in range(BASIN_TRIALS):
                s, _ = relax(W, corrupt(xi, f, rng), 50, sd + 7 * t + 1)
                ok += overlap(s, xi) > 0.95
            if ok * 2 >= BASIN_TRIALS:
                r = f
            else:
                break
        basin.append(r)

        # find the attractor nearest the pattern, then kick it and time recovery
        a, _ = relax(W, xi.copy(), 50, sd + 99)
        m_star = overlap(a, xi)
        if m_star > 0.5:
            for t in range(4):
                rt = recovery_time(W, corrupt(a, KICK, rng), xi, m_star, 50, sd + 3 + t)
                kfail.append(float(np.isnan(rt)))
                rtime.append(rt)
        else:
            kfail.append(1.0)                  # memory already gone

        # noisy dynamics around the memory: variance and lag-1 autocorrelation
        tr, eeg = glauber_trace(W, xi.copy(), xi, v, T_NOISE, BURN_IN + TRACE_SWEEPS, sd + 5)
        tr, eeg = tr[BURN_IN:], eeg[BURN_IN:]
        if tr.min() > 0.3:                     # still near the memory the whole time
            m_traces.append(tr)
            e_traces.append(eeg)
    # need most memories still alive, otherwise the noisy statistics are undefined
    if len(m_traces) < TEST_PATTERNS // 2:
        m_traces, e_traces = [], []
    mvar, mac1 = pooled_stats(m_traces)
    evar, eac1 = pooled_stats(e_traces)
    f = lambda v: float(np.nanmean(v)) if np.any(np.isfinite(v)) else np.nan
    g = lambda v: float(np.nanmedian(v)) if np.any(np.isfinite(v)) else np.nan
    return [f(rec), f(basin), f(stab), g(rtime), f(kfail), mvar, mac1, evar, eac1]


def main():
    keys = ["recall", "basin", "stability", "recovery", "kick_fail", "variance", "ac1",
            "eeg_var", "eeg_ac1"]
    res = {k: np.full((SEEDS, DAMAGE.size), np.nan) for k in keys}
    for seed in range(SEEDS):
        xi, W_full, iu, ju = build_network(seed)
        rng = np.random.default_rng(1000 + seed)
        region = rng.choice(N, N // 5, replace=False)    # neurons under the electrode
        v = np.zeros(N); v[region] = 1.0 / region.size
        for j, d in enumerate(DAMAGE):
            vals = measure(damaged(W_full, iu, ju, d), xi, v, rng)
            for k, val in zip(keys, vals):
                res[k][seed, j] = val
        print(f"network {seed + 1}/{SEEDS} done", flush=True)
    np.savez("results.npz", damage=DAMAGE, N=N, P=P, **res)
    print("saved results.npz")


if __name__ == "__main__":
    main()
