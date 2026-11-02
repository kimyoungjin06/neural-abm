# Information-spread crossover prediction v1 (experiment 3, stage 1): result

Date: 2026-10-30

Status: `E3_THE_RULE_BEATS_THE_BASELINES_BUT_MISSES_BY_MORE_THAN_THE_BOUND`

Design: [info-spread-crossover-prediction-v1.md](../../../docs/research/info-spread-crossover-prediction-v1.md),
frozen at `871ace34cda273bce086a232c70f77c9b5e8a254`.
Machine-readable record:
[info_spread_crossover_prediction_v1_20261030.summary.json](info_spread_crossover_prediction_v1_20261030.summary.json).

## Execution

- **Host:** GPU-Server's CPU, with 36 workers and one BLAS thread per
  process. The code was shipped as a git bundle and checked out at the
  freeze.
- **Wall time:** 3,475 s for the development run and 3,509 s for the
  reproduction run, inside the quoted 40 to 120 minutes.
- **Reproduction:** `measurement.jsonl`, `cells.jsonl`, and `summary.json`
  are byte identical.
- **Data read:** none; the worlds are synthetic.

## What was measured

From learner runs with one fixed sign, on worlds the comparison did not use:

| Visibility | Dd, between-run heterogeneity | V, within-run variance |
|---|---:|---:|
| 0.2 | 0.0925 | 0.0320 |
| 0.5 | 0.1014 | 0.0133 |
| 1.0 | 0.1034 | 0.0067 |

- **Within-run variance.** V falls in proportion to the visibility.
- **Heterogeneity.** Dd changes little with visibility.

## The registered score

The crossing is where L0 − T0-k (share precision on unverified items) rises
above zero to stay there.

| Visibility | k | Observed | Rule | Constant | Table size only |
|---|---:|---:|---:|---:|---:|
| 0.2 | 2 | 0.060 | 0.122 | 0.175 | 0.088 |
| 0.2 | 4 | 0.158 | 0.255 | 0.175 | 0.175 |
| 0.2 | 8 | 0.185 | 0.421 | 0.175 | 0.249 |
| 0.5 | 2 | 0.015 | 0.041 | 0.175 | 0.088 |
| 0.5 | 4 | 0.052 | 0.077 | 0.175 | 0.175 |
| 0.5 | 8 | 0.073 | 0.102 | 0.175 | 0.249 |
| 1.0 | 2 | 0.011 | 0.021 | 0.175 | 0.088 |
| 1.0 | 4 | 0.044 | 0.038 | 0.175 | 0.175 |
| 1.0 | 8 | 0.052 | 0.050 | 0.175 | 0.249 |

| Predictor | Mean absolute error (9 series) | Series within 0.05 |
|---|---:|---:|
| **The rule** | 0.0549 | 6 of 9 |
| Constant (0.175) | 0.1052 | 2 of 9 |
| Table size only | 0.0987 | 2 of 9 |

**The rule beats both baselines, but its mean error, 0.0549, misses the
registered bound of 0.05.**

| Expectation | Result |
|---|---|
| X1: at p = 0 every table with k ≥ 2 matches or beats L0 | Held |
| X2: the crossing does not fall as k grows | Held |
| X3: the crossing does not rise with visibility | Held |
| X4: the rule predicts the crossover | **Failed**: 0.0549 > 0.05 |
| X5: at p = 0, L0 − T0-1 is indistinguishable | Held |

## The comparison

Mean cell difference L0 − T0-k, with the group rule's label: + and −
material, 0 indistinguishable.

| Visibility | k | p = 0 | p = 0.05 | p = 0.1 | p = 0.15 | p = 0.2 | p = 0.25 | p = 0.3 | p = 0.4 | p = 0.5 |
|---|---:|---|---|---|---|---|---|---|---|---|
| 0.2 | 1 | −0.001 0 | +0.026 + | +0.068 + | +0.084 + | +0.111 + | +0.115 + | +0.128 + | +0.134 + | +0.134 + |
| 0.2 | 2 | −0.028 − | −0.006 0 | +0.022 + | +0.031 + | +0.042 + | +0.065 + | +0.069 + | +0.080 + | +0.082 + |
| 0.2 | 4 | −0.044 − | −0.028 − | −0.012 0 | −0.003 0 | +0.014 0 | +0.038 + | +0.040 + | +0.058 + | +0.066 + |
| 0.2 | 8 | −0.047 − | −0.033 − | −0.020 − | −0.011 0 | +0.005 0 | +0.021 + | +0.024 + | +0.047 + | +0.057 + |
| 0.5 | 1 | +0.004 0 | +0.032 + | +0.072 + | +0.085 + | +0.112 + | +0.120 + | +0.134 + | +0.144 + | +0.142 + |
| 0.5 | 2 | −0.005 0 | +0.013 + | +0.037 + | +0.045 + | +0.057 + | +0.087 + | +0.088 + | +0.108 + | +0.109 + |
| 0.5 | 4 | −0.015 − | −0.001 0 | +0.017 + | +0.025 + | +0.040 + | +0.067 + | +0.069 + | +0.086 + | +0.091 + |
| 0.5 | 8 | −0.018 − | −0.006 0 | +0.008 0 | +0.015 + | +0.027 + | +0.046 + | +0.048 + | +0.073 + | +0.080 + |
| 1.0 | 1 | +0.003 0 | +0.031 + | +0.069 + | +0.082 + | +0.111 + | +0.118 + | +0.133 + | +0.142 + | +0.139 + |
| 1.0 | 2 | −0.004 0 | +0.014 + | +0.035 + | +0.044 + | +0.056 + | +0.081 + | +0.084 + | +0.106 + | +0.107 + |
| 1.0 | 4 | −0.012 − | +0.002 0 | +0.019 + | +0.028 + | +0.041 + | +0.065 + | +0.070 + | +0.090 + | +0.094 + |
| 1.0 | 8 | −0.013 − | −0.001 0 | +0.013 + | +0.020 + | +0.032 + | +0.049 + | +0.053 + | +0.079 + | +0.087 + |

The first material advantage of the learner (the O26-style onset):

| Visibility | k = 1 | k = 2 | k = 4 | k = 8 |
|---|---|---|---|---|
| 0.2 | 0.05 | 0.1 | 0.25 | 0.25 |
| 0.5 | 0.05 | 0.05 | 0.1 | 0.15 |
| 1.0 | 0.05 | 0.05 | 0.1 | 0.1 |

- **A larger table delays the onset.** This is the table-size half of O29.
- **More within-run evidence brings it forward.**
- **O26 replicates on new worlds.** At visibility 0.5, the four-donor row is
  close to O26's Regime A linear group: −0.015, +0.017, and +0.067 at
  p = 0, 0.1, and 0.25, against O26's −0.017, +0.010, and +0.069.
  - O26's p = 0.1 interval covered zero; this one does not.
  - So the onset reads 0.1 here and 0.25 in O26.

## Post hoc, descriptive, not registered

- **The misses come from scarce evidence.**
  - At visibility 1.0 the rule is within 0.010 of every observed crossing,
    on either side.
  - At 0.5 it is late in all three series, by at most 0.030.
  - At 0.2 it is late by 0.062 to 0.236.
  - Without visibility 0.2, its mean error would be 0.017.
- **What the rule gets right.**
  - It orders the crossings across table sizes and evidence levels as
    observed.
  - It gives the near-tie of the one-donor table at p = 0.
- **Why it might be late at low evidence (not tested).**
  - Its error is a squared error on uniform probe items, which weights
    items far from the share cutoff.
  - A learner's noise there costs less share precision than the squared
    error says.

## What this means

- **Two measurable quantities order the crossover.** Between-run
  heterogeneity and within-run evidence order it across table sizes and
  evidence levels, and they place it well where evidence is ample.
- **The plug-in rule is not accurate enough to call predictive** by the
  registered bound. It is late when evidence is scarce.
- **Stage 2 premise.** The plan was to apply the rule to OpenAlex discovery
  episodes. There labels are few per episode and visible only five quarters
  after entry (experiment 4), so stage 2 would sit in the regime where the
  rule misses.

## Boundary

- **Synthetic, one generator.** One veracity relation, Regime A, and the
  exact linear learner.
- **One kind of run-specific information:** a sign drawn once per run.
- **Squared-error proxy.** The rule uses a squared-error proxy on a fixed
  probe; it is scored against share precision on unverified items.
- **Scope.** No empirical row, no LLM call, and no change to a generic NABM
  API.
