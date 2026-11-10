# Analysis plan: basin radius in AD vs healthy (written before running the 150-recording sample)

Date fixed: 2026-11-07, before any of these 150 recordings were processed.

## Sample (`sample_150.json`, seed 2026)
- Ages 62–78. AD = any `ad` label. Healthy = `cb_normal` only (SMI kept out of "healthy").
- Recordings need ≥ 90 s of estimated clean eyes-closed rest (from the event files only).
- AD drawn in random order. Each is paired with an unused healthy control within ±1 year.
  The first 50 pairs form the sample. The healthy test group is 69.9 ± 4.5 y and AD is
  70.0 ± 4.6 y (p = 0.90).
- A separate 50 healthy controls (training set, 65.0 ± 2.6 y) set the thermometer
  thresholds, the healthy mean, and W_HC. They are never used as test subjects.
- Any recording with < 60 epochs after preprocessing is replaced by the next backup
  (pairs are replaced together). Only data quality decides this, never results.
- Known limitation: AD patients aged 73–78 rarely have an age-matched control in
  CAUEEG, so this AD group is younger than CAUEEG's AD population overall.

## Fixed settings (no changes after seeing results)
Preprocessing as in `caueeg_preprocess.py`. Bands δ θ α β (gamma dropped). K = 5 bits.
K = 8 prototypes. Healthy-mean centering (as in the research plan). γ = 2.
Recall = cosine with the target > 0.9 and overlap tanh(x)·target/N > 0.1. 20 trials per corruption level. Corruption grid
2.5%–50% in 2.5% steps.

## Measures (per person)
- **Basin radius R (primary).** Use the person's own 8 prototypes, sign-binarized.
  Corrupt a prototype by flipping a fraction f of its bits. The corrupted cue sets the
  IDP saliencies, α_ν = γ·max(0, ξ_ν·s/N), and the start state. R for that prototype is
  the largest f with ≥ 50% correct recall. The person's R is the plain mean over the 8.
- **Depth ΔV (secondary, research-plan definition).** Settle from the clean prototype
  (bottom, x*). Then flip a fraction R of x*'s coordinates, which is the rim of the basin.
  ΔV = E(rim) − E(bottom) per neuron, in the landscape the clean cue creates. Mean over
  the 8 prototypes.
- Exploratory only: depth measured from the confusion state, [E(0) − E(x*)] / N.
- Changed 2026-11-09 before any of the 150 were run: ΔV switched from the
  confusion-state version to the plan's rim-minus-bottom definition, and trials went
  from 6 to 20.
- **Theta/alpha ratio (baseline).** Posterior relative theta / alpha, per person.

## Tests (decided now)
1. Primary: R is smaller in AD than in matched healthy (one-sided Mann–Whitney, α = 0.05).
   Effect size: rank-biserial r.
2. Secondary: ΔV is smaller in AD (same test).
3. Added value: logistic regression, AD vs healthy ~ θ/α, then ~ θ/α + R. Likelihood-ratio
   test for adding R. If R only works through θ/α, say so plainly.
4. Whatever comes out is reported, including a null result.

## Sample size change (2026-11-09, before any results)
Reduced to the first 25 age-matched pairs and the first 25 training controls of the same random draw (`sample_75.json`), for practical reasons. With 25 per group, only fairly large differences (rank-biserial r of about 0.35 or more) are likely to be detected. A null result at this size is weaker evidence than at 50.
