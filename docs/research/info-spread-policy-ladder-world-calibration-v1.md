# Information-spread policy ladder: world-calibration study v1 (O33)

Date: 2026-10-24

Status: `PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION`

Machine-readable contract:
[`info_spread_policy_ladder_world_calibration_v1.yaml`](../../experiments/configs/info_spread_policy_ladder_world_calibration_v1.yaml)

The user chose O33, the first-ranked open item of
[consolidation v8](pivot-policy-ladder-consolidation-v8.md), on 2026-10-24
(N25).

## Question

O28 split each LLM prior into two parts:
- **its direction,** the model's answers;
- **its calibration,** its location and steepness over D3b's design cohort.

It crossed the two parts against the hand rule F1 and found:

- **C63.** The utility cost of D5's prior inside the learner comes from its
  calibration, in both regimes.
- **C64.** For the unclipped prior in Regime B, the answers and the
  calibration interact. The answers lower utility at F1's calibration in
  6 of 6 groups, and raise it at their own calibration in 4 of 6.

**Why repeat it.** O28 matched the recalibration on the cohort, and in the
world it overshot (C66).
- An LLM direction at F1's calibration shared 1.35 to 1.48 times as much as
  F1.
- F1's direction at an LLM calibration shared 20 to 46% as much as the LLM
  prior.
- So the comparisons at a recalibrated corner also carried a difference in
  how much the starting prior shares.

**The question.** Do C63 and C64 hold when each recalibrated prior's
location is matched to its source's share rate in the world?

## World calibration

**What is kept and what is changed.** Each recalibrated prior keeps O28's
weights: its direction at its calibration source's steepness over the
cohort. Only its intercept is set here.

**Calibration worlds.**
- They come from study v2's generator and profile, at world seeds
  15000–15011: four per study world (linear, smooth, axis).
- Each runs two stream replicates, so there are 24 calibration runs per
  regime.
- No registered study uses these seeds. The study cells are at 7000–7019.

**The statistic.** The fixed prior's mean unverified shares per run. These
are its decided shares, from the policy and the exploration floor, of items
whose fact-check is not public at the decision.

**The target.** The calibration source's mean (F1, P0, or P0u) on the same
24 runs.

**The intercept.**
1. **Keep rule.** If O28's cohort-matched intercept already shares within 2%
   of the target, it is kept. This is expected for F1 at P0's Regime A
   calibration: on the study cells, O28's version shared exactly like P0,
   because neither shares beyond the exploration floor.
2. **Otherwise, bisection.** The intercept is the largest one in [−40, 40]
   whose share count does not exceed the target, found in 32 steps.
3. **Reporting.** Every target, method, intercept, and achieved count is
   reported.

**The calibration runs inside each registered run,** before any study task.
So the development and reproduction runs must reach the same intercepts.

## Arms and comparisons

O28's 2 × 2, with the world-calibrated priors in its two new corners:

| | F1's calibration | Q's calibration |
|---|---|---|
| **F1's direction** | L0 (recorded) | L0_pcalw or L0_ucalw (new) |
| **Q's direction** | L1_hcalw or L1u_hcalw (new) | L1 or L1u (recorded) |

- **The fixed priors** of the new corners also run, for the world-match
  check.
- **Comparisons.** O28's eight, with the new arms: calibration along both
  routes, and answers along both routes, for P0 and for P0u.
- **Measure, recorded inputs, and blocking checks.** The same as O28's:
  - utility per agent on unverified items;
  - recorded utility from study v2, D5, O24, and O25;
  - L0 parity at replicate 0 of every learner task;
  - consistency of every group's L1 − L0 and L1u − L0 with O25's record.
- **Design.** Study v2's 12 study groups, with 20 cells of 16 replicates.

## Rule and dispositions

**The group rule.**
- **Bootstrap.** 2,000 cell resamples, 95% interval, rule tag 33.
- **Minimum effect.** O25's recorded value per regime, 0.5895 in Regime A
  and 0.1549 in Regime B.

**Per comparison and regime:**
- `MOSTLY_POSITIVE` if positive in at least 4 of 6 study groups;
- `MOSTLY_NEGATIVE` if negative in at least 4 of 6;
- `MIXED_OR_NONE` otherwise.

**The world-match check.** It is out of sample, on the study cells.
- **The test.** Each world-calibrated fixed prior's mean unverified share
  count must lie within 10% of its calibration source's.
- **If it fails.** If either world-calibrated prior of Q fails in a regime,
  that case is `WORLD_MATCH_FAILED`.

**Attribution per Q and regime.** O28's rule:

| Attribution | Calibration comparisons `MOSTLY_NEGATIVE` | Answers comparisons `MOSTLY_NEGATIVE` |
|---|---|---|
| `CALIBRATION` | Both | Not both |
| `ANSWERS` | Not both | Both |
| `BOTH` | Both | Both |
| `NEITHER_CONSISTENTLY` | Not both | Not both |

**The claims tested.**
- **C63 holds** if P0/A and P0/B are both `CALIBRATION`.
- **C64 holds** if P0u/B is `NEITHER_CONSISTENTLY`.

**Overall.**
- `O33_WORLD_MATCH_FAILED` if any costed case (P0/A, P0/B, P0u/B) failed the
  world-match check;
- otherwise `O33_C63_AND_C64_HOLD`, `O33_ONLY_C63_HOLDS`,
  `O33_ONLY_C64_HOLDS`, or `O33_NEITHER_HOLDS`.

The O28 attribution of each case is reported beside the new one.

**Expected before the run.** These are stated so that they can be rejected.
- **C63 holds.** In Regime A, F1 at P0's calibration is expected to be
  kept, since O28's version already shared like P0. If so, one of the two
  calibration routes is unchanged there.
- **C64 is not predicted.** Its strongest part, the answers lowering utility
  at F1's calibration, is the comparison the overshoot affected most.

## Runs and record

1. **Pilot.** A full-load pilot (2 cells per group), with its world
   calibration in full, for the time estimate. Its summary is not opened.
2. **Development and reproduction runs.**
   - On GPU-Server, on GPU 2 with 24 workers.
   - `cells.jsonl` and `summary.json` must match byte for byte.
3. **Record.** A summary and a result test.

**The time estimate.** It scales the pilot's MLP task time by the full
run's concurrency on the GPU. The full run puts 24 MLP tasks on the GPU at
once, against the pilot's 12. This follows the lesson from O28.

## Boundary

- **Only the location is matched in the world.** The steepness stays O28's,
  matched on the cohort.
- **The match is on how many unverified items are shared,** not on which.
- **One utility,** the sharer's own (O31).
- **Synthetic worlds,** with no empirical data and no LLM call.
