"""
Experiment 2: Do the two halves of the research plan agree inside an IDP model?

Model (Input-Driven Plasticity Hopfield network, after Betteti et al. 2025)
    dx/dt = -x + W(c) tanh(x) + noise
    W(u,c) = (1/N) * sum_nu  a_nu(u) * xi_nu xi_nu^T
    a_nu(u) = c * beta_nu * max(0, xi_nu . u / N)      (input-driven saliency)
  - xi_nu  : stored prototypes (+1/-1), K of them
  - beta_nu: intrinsic strength of each memory (some start more vulnerable)
  - u      : the current input (perfect cue = xi_mu, or a corrupted cue)
  - c      : fraction of the network still supporting memory (1 = intact)
  For a perfect cue the active memory has effective strength a = c * beta_mu.
  It exists only while a > 1; below that its valley flattens into the
  confusion state at the origin.
  Damage never chooses WHICH memory dies: weaker ones cross a = 1 first.

Geometry half (structural)
  R      basin radius: largest fraction of flipped bits that still recalls
  dU     barrier height per neuron, U(0) - U(m*), along the memory direction
  D_hat  noise level estimated the way the plan does it: Var(m) * curvature
  P      Kramers escape rate -> survival analysis -> P(collapse within horizon)

Dynamics half (critical slowing down)
  A "photic" pulse (the same random flash pattern each time, 8 repeats) kicks the
  network; we fit the decay constant tau of the recovery in
    - the memory overlap m(t)          (ground truth, not measurable in a patient)
    - a virtual EEG electrode e(t)     (mean activity of 20% of neurons)

Run:  python3 idp_ews.py
"""
import numpy as np
from scipy.optimize import curve_fit

N = 200
ALPHA0 = np.array([1.4, 1.7, 2.0, 2.4, 2.9, 3.5])   # memory strengths, weakest first
K = ALPHA0.size
SIGMA = 0.5                       # per-neuron noise amplitude
D_TRUE = SIGMA ** 2 / (2 * N)     # noise felt along a memory direction (1-D reduction)
DT = 0.05
SEEDS = 4
C_GRID = np.round(np.arange(1.0, 0.249, -0.025), 3)
BASIN_GRID = np.round(np.arange(0.05, 0.501, 0.025), 3)
BASIN_TRIALS = 6
HORIZON = 1000.0                  # model time units for P(collapse)
PULSES, KAPPA, T_PULSE = 8, 1.0, 1.0
T_REC, T_PRE = 30.0, 20.0


# ------------------------------------------------------------------ model
def drift(X, Xi, a):
    """-x + W tanh(x) for a batch of states X (B,N); W is low rank so use Xi."""
    V = np.tanh(X)
    return -X + ((V @ Xi.T) * a) @ Xi / N


def integrate(X, Xi, a, T, rng=None, sigma=0.0, I=None):
    steps = int(round(T / DT))
    for _ in range(steps):
        X = X + DT * drift(X, Xi, a)
        if I is not None:
            X = X + DT * I
        if sigma > 0:
            X = X + sigma * np.sqrt(DT) * rng.standard_normal(X.shape)
    return X


def energy(X, Xi, a):
    """Hopfield continuous energy: -1/2 v^T W v + sum_i G(v_i), v = tanh x,
    G(v) = int_0^v artanh = x tanh x - log cosh x (written in x for stability)."""
    V = np.tanh(X)
    quad = ((V @ Xi.T) ** 2 * a).sum(1) / N
    G = (X * V - np.logaddexp(X, -X) + np.log(2)).sum(1)
    return -0.5 * quad + G


def fixed_point_1d(a):
    """Solve m = a tanh(m) (memory amplitude in the 1-D reduction)."""
    if a <= 1:
        return 0.0
    m = a
    for _ in range(500):
        m = a * np.tanh(m)
    return m


def U1d(m, a):
    return 0.5 * m ** 2 - a * np.log(np.cosh(m))


def kramers_rate(a, D):
    """Escape rate out of the memory well over the confusion state at m = 0."""
    if a <= 1:
        return np.inf
    m = fixed_point_1d(a)
    k_well = 1 - a / np.cosh(m) ** 2            # curvature at the memory
    k_top = abs(1 - a)                          # curvature at the barrier top
    dU = U1d(0, a) - U1d(m, a)
    return np.sqrt(k_well * k_top) / (2 * np.pi) * np.exp(-dU / D)


def corrupt_batch(xi, frac, n, rng):
    S = np.tile(xi, (n, 1))
    k = int(round(frac * xi.size))
    for r in range(n):
        S[r, rng.choice(xi.size, k, replace=False)] *= -1
    return S


def cos_sim(X, xi):
    return (X @ xi) / (np.linalg.norm(X, axis=1) * np.sqrt(xi.size) + 1e-12)


def exp_tau(t, y, t0):
    """Fit y = A exp(-t/tau) on t >= t0 (after the fast transients die out)."""
    msk = t >= t0
    try:
        p, _ = curve_fit(lambda tt, A, tau: A * np.exp(-tt / tau), t[msk], y[msk],
                         p0=(y[msk][0], 3.0), bounds=([-np.inf, 0.05], [np.inf, 500]),
                         maxfev=5000)
        return p[1] if 0.06 < p[1] < 499 else np.nan     # fit pinned at a bound = failed
    except Exception:
        return np.nan


# ------------------------------------------------------------------ one patient
def saliency(Xi, u, c):
    return c * ALPHA0 * np.maximum(0.0, (Xi @ u) / N)


def recall_from_cue(Xi, xi, c, f, trials, rng, amp=0.05):
    """IDP retrieval: the corrupted cue sets the saliency (the landscape) and the
    state starts near the confusion state. Success = the state grows into the memory."""
    S = corrupt_batch(xi, f, trials, rng)
    A = c * ALPHA0 * np.maximum(0.0, (S @ Xi.T) / N)          # (trials, K)
    Y = integrate(amp * S, Xi, A, 60.0)
    ok = (cos_sim(Y, xi) > 0.9) & ((Y @ xi) / N > 0.1)
    return ok.mean()


def measure(Xi, c, region, flash, rng):
    out = {k: np.full(K, np.nan) for k in
           ["a", "exists", "m_det", "recall10", "avail", "R", "R_theory", "dU", "dU_1d",
            "D_hat", "P_true", "P_hat", "var_e", "ac1_e",
            "tau_theory", "tau_det_m", "tau_det_e", "tau_m", "tau_e", "tau_al_m", "tau_al_e"]}

    for mu in range(K):
        xi = Xi[mu]
        a = saliency(Xi, xi, c)                 # resting input = this memory
        out["a"][mu] = a[mu]
        X = integrate(3.0 * xi[None, :], Xi, a, 25.0)
        exists = cos_sim(X, xi)[0] > 0.9 and (X[0] @ xi) / N > 0.05
        out["exists"][mu] = float(exists)
        out["R_theory"][mu] = max(0.0, 0.5 * (1 - 1 / a[mu]))
        # visible symptom: recall from a 10%-corrupted cue
        out["recall10"][mu] = recall_from_cue(Xi, xi, c, 0.10, 4, rng)
        if not exists:
            continue
        xstar = X
        m_det = (X[0] @ xi) / N
        out["m_det"][mu] = m_det

        # barrier: confusion state (origin) minus the memory, per neuron
        out["dU"][mu] = (energy(np.zeros((1, N)), Xi, a)[0] - energy(xstar, Xi, a)[0]) / N
        out["dU_1d"][mu] = U1d(0, a[mu]) - U1d(fixed_point_1d(a[mu]), a[mu])

        # basin radius (the corrupted cue sets the saliency, as in IDP)
        R = 0.0
        for f in BASIN_GRID:
            if recall_from_cue(Xi, xi, c, f, BASIN_TRIALS, rng) >= 0.5:
                R = f
            else:
                break
        out["R"][mu] = R

        # noisy resting state: availability, plan's D estimator, EEG var / AC1
        Xr = xstar.copy()
        T_run, every = 300.0, 10
        steps = int(T_run / DT)
        ms, es = [], []
        for s in range(steps):
            Xr = Xr + DT * drift(Xr, Xi, a) + SIGMA * np.sqrt(DT) * rng.standard_normal(Xr.shape)
            if s % every == 0:
                ms.append((Xr[0] @ xi) / N)
                es.append(np.tanh(Xr[0, region]).mean())
        ms, es = np.array(ms), np.array(es)
        in_basin = ms > 0.5 * m_det
        out["avail"][mu] = in_basin.mean()
        if in_basin.all():
            m0 = m_det
            curv = 1 - a[mu] / np.cosh(m0) ** 2            # 1-D curvature at the memory
            out["D_hat"][mu] = np.var(ms) * curv           # plan's estimator
            x = es - es.mean()
            out["var_e"][mu] = np.var(es)
            out["ac1_e"][mu] = (x[:-1] @ x[1:]) / (x @ x)

        # Kramers + survival analysis
        r_true = kramers_rate(a[mu], D_TRUE)
        out["P_true"][mu] = 1 - np.exp(-r_true * HORIZON)
        if np.isfinite(out["D_hat"][mu]):
            r_hat = kramers_rate(a[mu], out["D_hat"][mu])
            out["P_hat"][mu] = 1 - np.exp(-r_hat * HORIZON)

        # photic pulses: same flash pattern each time
        m0 = fixed_point_1d(a[mu])
        out["tau_theory"][mu] = 1.0 / (1 - a[mu] / np.cosh(m0) ** 2)
        rec_steps = int(T_REC / DT)
        # (i) noise-free pulse: ground truth for the fitting method
        Xd = integrate(xstar.copy(), Xi, a, T_PULSE, I=KAPPA * flash)
        tm, em = [], []
        e_star = np.tanh(xstar[0, region]).mean()
        for s in range(rec_steps):
            if s % 5 == 0:
                tm.append((Xd[0] @ xi) / N - m_det)
                em.append(np.tanh(Xd[0, region]).mean() - e_star)
            Xd = Xd + DT * drift(Xd, Xi, a)
        t = np.arange(len(tm)) * 5 * DT
        out["tau_det_m"][mu] = exp_tau(t, np.array(tm), 3.0)
        out["tau_det_e"][mu] = exp_tau(t, np.array(em), 3.0)
        # (ii) realistic: noisy, 8 pulses averaged (CAUEEG averages ~7.8 sessions)
        def noisy_pulse(stim):
            Xp = np.tile(xstar[0], (PULSES, 1))
            Xp = integrate(Xp, Xi, a, T_PRE, rng, SIGMA)          # independent noisy starts
            base_m = (Xp @ xi) / N
            base_e = np.tanh(Xp[:, region]).mean(1)
            Xp = integrate(Xp, Xi, a, T_PULSE, rng, SIGMA, I=KAPPA * stim)
            tm, em = [], []
            for s in range(rec_steps):
                if s % 5 == 0:
                    tm.append(((Xp @ xi) / N - base_m).mean())
                    em.append((np.tanh(Xp[:, region]).mean(1) - base_e).mean())
                Xp = Xp + DT * drift(Xp, Xi, a) + SIGMA * np.sqrt(DT) * rng.standard_normal(Xp.shape)
            if not np.all(((Xp @ xi) / N) > 0):                  # pulses knocked the memory out
                return np.nan, np.nan
            return exp_tau(t, np.array(tm), 0.5), exp_tau(t, np.array(em), 0.5)
        # random flash pattern: the stimulus has nothing to do with the memory
        out["tau_m"][mu], out["tau_e"][mu] = noisy_pulse(flash)
        # best case: a stimulus that directly disturbs the memory (pushes it toward collapse)
        out["tau_al_m"][mu], out["tau_al_e"][mu] = noisy_pulse(-xi)
    return out


def escape_validation(rng, a_list, trials=128, T_max=3000.0):
    """Simulated mean escape time vs Kramers prediction, single network."""
    g = np.random.default_rng(7)
    Xi = g.choice([-1.0, 1.0], size=(K, N))
    mu = 0
    rows = []
    for a_target in a_list:
        c = a_target / ALPHA0[mu]
        a = saliency(Xi, Xi[mu], c)
        x0 = integrate(3.0 * Xi[mu:mu + 1], Xi, a, 25.0)
        m_star = (x0[0] @ Xi[mu]) / N
        X = np.tile(x0[0], (trials, 1))
        t_esc = np.full(trials, np.nan)
        alive = np.ones(trials, bool)
        steps = int(T_max / DT)
        for s in range(steps):
            X = X + DT * drift(X, Xi, a) + SIGMA * np.sqrt(DT) * rng.standard_normal(X.shape)
            if s % 4 == 0:
                m = (X @ Xi[mu]) / N
                hit = alive & (m < -0.5 * m_star)
                t_esc[hit] = s * DT
                alive &= ~hit
                if not alive.any():
                    break
        frac = np.mean(np.isfinite(t_esc))
        # censored data: exponential MLE for the escape rate
        total_time = np.nansum(t_esc) + np.sum(~np.isfinite(t_esc)) * T_max
        r_sim = np.sum(np.isfinite(t_esc)) / total_time
        rows.append((a_target, r_sim, kramers_rate(a_target, D_TRUE), frac))
        print(f"  a={a_target:.2f}  simulated rate={r_sim:.3g}  Kramers={rows[-1][2]:.3g}  "
              f"escaped {frac:.0%}", flush=True)
    return np.array(rows)


def main():
    keys = None
    allres = []
    for seed in range(SEEDS):
        rng = np.random.default_rng(seed)
        Xi = rng.choice([-1.0, 1.0], size=(K, N))
        region = rng.choice(N, N // 5, replace=False)
        flash = rng.choice([-1.0, 1.0], size=N)          # the photic stimulus pattern
        per_c = []
        for c in C_GRID:
            per_c.append(measure(Xi, c, region, flash, rng))
        keys = list(per_c[0].keys())
        allres.append({k: np.stack([p[k] for p in per_c]) for k in keys})   # (C, K)
        print(f"network {seed + 1}/{SEEDS} done", flush=True)
    res = {k: np.stack([r[k] for r in allres]) for k in keys}                 # (S, C, K)
    print("Kramers validation:")
    val = escape_validation(np.random.default_rng(99),
                            [1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.08])
    np.savez("idp_results.npz", c=C_GRID, alpha0=ALPHA0, N=N, sigma=SIGMA,
             D_true=D_TRUE, horizon=HORIZON, kramers_val=val, **res)
    print("saved idp_results.npz")


if __name__ == "__main__":
    main()
