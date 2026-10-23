# Information-spread policy ladder: prior-calibration study v1 (O28)

Date: 2026-10-22

Status: `PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION`

Machine-readable contract:
[`info_spread_policy_ladder_prior_calibration_v1.yaml`](../../experiments/configs/info_spread_policy_ladder_prior_calibration_v1.yaml)

The user chose O28, the first-ranked open item of
[consolidation v7](pivot-policy-ladder-consolidation-v7.md), on 2026-10-22
(N21).

## Question

O25 scored the registered arms by the agents' own utility on unverified
items: +1 per true share and −c per false share. It found that starting the
learner from an LLM prior instead of the hand rule F1 lowers that utility:

- L1 − L0 in Regime A, 6 of 6 study groups;
- L1 − L0 in Regime B, 4 of 6;
- L1u − L0 in Regime B, 4 of 6.

These are the three **costed cases**. L1u − L0 in Regime A was
`MIXED_OR_NONE`.

**The question.** Does the cost come from the prior's calibration or from
the model's answers?

Every LLM prior went through D3b's calibration rule: one intercept shift so
that its mean probability over the design cohort is the base rate 0.4. F1
was never calibrated that way.

**Calibration on the cohort, read before this freeze.** These values come
from the recorded weights.

| Prior | Regime | Mean probability | Logit sd | Share of profiles at p ≥ 0.5 |
|---|---|---:|---:|---:|
| F1 | A / B | 0.689 / 0.674 | 1.43 / 1.47 | 0.76 / 0.75 |
| P0 | A / B | 0.400 / 0.400 | 0.06 / 3.40 | 0.00 / 0.40 |
| P0u | A / B | 0.400 / 0.400 | 2.50 / 7.71 | 0.39 / 0.40 |

## Calibration and direction

A linear prior's logit over the design cohort has two calibration
properties:
- its **location**, the mean probability over the cohort;
- its **steepness**, the population standard deviation of the logit.

Its **direction** is its weight vector up to a positive scale. For an LLM
prior, the direction is what the model's answers contribute.

**Recalibrating** prior D to the calibration of prior C keeps D's direction
and takes C's location and steepness:
1. Multiply D's weights by sd(cohort · C's weights) / sd(cohort · D's
   weights).
2. Set the intercept by D3b's bisection, so that the mean probability over
   the cohort equals C's. D's own intercept is not used.

## Arms

For each LLM prior Q (P0, and the unclipped P0u) and each regime, crossing
direction with calibration gives a 2 × 2.

| | F1's calibration | Q's calibration |
|---|---|---|
| **F1's direction** | L0 (recorded, study v2) | L0_pcal or L0_ucal (new) |
| **Q's direction** | L1_hcal or L1u_hcal (new) | L1 or L1u (recorded, D5 and O24) |

Each new learner arm is study v2's learner, with its seeds, started from the
recalibrated prior. The four recalibrated priors (P0_hcal, F1_pcal, P0u_hcal,
F1_ucal) also run as fixed arms, for the calibration check below.

No LLM is called. P0 and P0u are rebuilt exactly from D5's first registered
elicitation.

## Comparisons

For each Q and regime, there are two routes from L0 to the recorded L1 (or
L1u), and each route crosses one factor at a time. Each comparison is the
arm with the LLM component minus the arm with the hand component.

| Comparison | P0 | P0u |
|---|---|---|
| Calibration, given the LLM's answers | L1 − L1_hcal | L1u − L1u_hcal |
| Calibration, given the hand rule's answers | L0_pcal − L0 | L0_ucal − L0 |
| Answers, given F1's calibration | L1_hcal − L0 | L1u_hcal − L0 |
| Answers, given the LLM's calibration | L1 − L0_pcal | L1u − L0_ucal |

Along either route, the two steps add up to O25's L1 − L0 (or L1u − L0).

## Measure and checks

**Measure.** Utility per agent on unverified items, as in O25.
- **The new arms** are measured with O25's utility module.
- **Recorded utility.** Regime A's comes from the recorded counts. Regime
  B's comes from O25's development cells, after each of their rows is
  checked against its record.
- **Reported, not judged.** Every learner arm's unverified shares and share
  precision.

**Design.** Study v2's 12 study groups: two regimes, two learner families,
and the linear, smooth, and axis worlds. Each group has 20 cells of 16
replicates. The null world is not run, because O25 showed it cannot be
judged on utility.

**Checks that block reading.**
- **Parity.** At replicate 0 of every learner task, L0 is recomputed. It
  must equal its recorded counts, precisions, reach, decisions, and utility
  exactly.
- **Consistency.** In every group, the mean cell difference L1 − L0, and
  L1u − L0, must equal O25's recorded value.

**Calibration check.** It decides whether a case can be attributed.
- Recalibration acts on the cohort, so the study checks that it also moves
  how much a prior shares in the world.
- **The test.** For each recalibrated fixed prior and regime, its mean
  unverified share count over the study cells must lie closer to its
  calibration source's than to its direction source's.
- **If it fails.** If either recalibrated prior of Q fails in a regime, that
  case is `CALIBRATION_CHECK_FAILED`.

## Rule and dispositions

**The group rule.**
- **Bootstrap.** 2,000 cell resamples, 95% interval, rule tag 28.
- **Minimum effect.** O25's recorded value per regime, 0.5895 in Regime A
  and 0.1549 in Regime B. It is not recomputed.
- **Material.** The interval excludes zero and the mean is at least the
  minimum effect.

**Per comparison and regime:**
- `MOSTLY_POSITIVE` if positive in at least 4 of 6 study groups;
- `MOSTLY_NEGATIVE` if negative in at least 4 of 6;
- `MIXED_OR_NONE` otherwise.

**Attribution per Q and regime.**

| Attribution | Calibration comparisons `MOSTLY_NEGATIVE` | Answers comparisons `MOSTLY_NEGATIVE` |
|---|---|---|
| `CALIBRATION` | Both | Not both |
| `ANSWERS` | Not both | Both |
| `BOTH` | Both | Both |
| `NEITHER_CONSISTENTLY` | Not both | Not both |

**Overall.** It is read over the three costed cases, which the runner reads
from O25's record and checks against this list. P0u/A is reported but not
judged.
- `O28_CALIBRATION_CHECK_FAILED` if any costed case failed the calibration
  check;
- otherwise `O28_<ATTRIBUTION>_IN_EVERY_COSTED_CASE` if the three cases
  share one attribution;
- otherwise `O28_ATTRIBUTION_DIFFERS_ACROSS_COSTED_CASES`.

**Expected before the run.** `CALIBRATION` in every costed case. This is
stated so that it can be rejected. On the cohort, D3b's shift places every
LLM prior at a mean probability of 0.40, against F1's 0.67 to 0.69. At c = 1
the LLM priors share 0 to 40% of profiles, against F1's 75%.

## Runs and record

1. **Pilot.** A full-load pilot (2 cells per group) for the time estimate,
   which is quoted as a range. Its summary is not opened.
2. **Development and reproduction runs.**
   - On GPU-Server, on GPU 2 with 24 workers.
   - `cells.jsonl` and `summary.json` must match byte for byte.
3. **Record.** A summary and a result test.

## Boundary

- **Location and steepness move together.** This study does not separate
  which of the two matters.
- **P0's Regime A direction is mostly noise.** P0/A distilled with R² 0.026,
  so at F1's steepness its direction is largely the clipped readout's noise.
  P0u gives the cleaner reading of the model's answers.
- **One utility,** the sharer's own. The reach of false items is not scored
  (O31).
- **Synthetic worlds,** with no empirical data and no LLM call.
