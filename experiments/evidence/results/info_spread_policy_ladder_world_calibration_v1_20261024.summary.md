# Information-spread policy ladder: world-calibration study v1 (O33)

Date: 2026-10-24

Status: `O33_NEITHER_HOLDS`

Design: [world-calibration study v1](../../../docs/research/info-spread-policy-ladder-world-calibration-v1.md),
frozen at `a926c25` before the run.

## What ran

O28's 2 × 2 was repeated, with the location of each recalibrated prior
matched in the world instead of on the design cohort.

**World calibration.**
- **What changed.** Each recalibrated prior kept O28's weights. Its
  intercept was set on 24 calibration runs per regime, at world seeds
  15000–15011, so that it shares as many unverified items as its source.
- **The intercepts.** F1 at P0's Regime A calibration kept O28's intercept,
  as expected, since neither shares beyond the exploration floor. The other
  seven were set by bisection.
- **The match on the calibration runs.** Every target was met within 0.02%.
- **Time.** The step took 6 minutes in the development run and 7 in the
  reproduction.

**The world-match check passes for all 8 priors, out of sample.** On the
study cells, each world-calibrated fixed prior shares 0.984 to 1.001 times
as many unverified items as its calibration source. In O28 the same ratios
ran from 0.20 to 1.48.

**Checks.**
- **Parity.** L0 recomputed in all 240 learner tasks equalled its record.
- **Consistency.** All 24 group totals equalled O25's record.

**Runs.**
- GPU-Server, on GPU 2 with 24 workers.
- **Times.** 270 minutes for each run, against a quoted 4–5 hours. The
  quote was the pilot scaled by O28's measured run.
- `cells.jsonl` and `summary.json` match byte for byte between the two runs.

## Registered result

The measure is utility per agent on unverified items, with O25's minimum
effects. Each cell gives the range of the mean cell Δ over the regime's 6
study groups. The labels follow group order: linear (axis, linear, smooth),
then MLP (axis, linear, smooth).

| Case | Calibration, LLM answers | Calibration, hand answers | Answers, F1 calibration | Answers, LLM calibration | O33 | O28 |
|---|---|---|---|---|---|---|
| **P0/A** (costed) | −8.0 to −4.6 (− − − − − −) | −15.3 to −6.4 (− − − − − −) | −7.4 to −0.1 (− 0 − − − −) | −0.0 to +0.3 (0 0 0 0 0 0) | `CALIBRATION` | `CALIBRATION` |
| **P0/B** (costed) | −6.3 to +1.0 (− − − − + 0) | −4.0 to +2.4 (− − − − + +) | −1.8 to −0.5 (− − − − − −) | −4.6 to −0.6 (− − − − 0 −) | `BOTH` | `CALIBRATION` |
| **P0u/B** (costed) | −5.5 to +4.5 (− − − − + +) | −1.8 to +4.7 (− − + 0 + +) | −2.0 to −0.6 (− − − − − −) | −5.7 to −0.7 (− − − − 0 −) | `ANSWERS` | `NEITHER_CONSISTENTLY` |
| P0u/A (reported) | −2.1 to +0.8 (− − 0 − − +) | −5.2 to +0.6 (− 0 0 − − 0) | −0.2 to +1.9 (0 + + 0 0 0) | +0.8 to +2.9 (+ + + + + 0) | `NEITHER_CONSISTENTLY` | `NEITHER_CONSISTENTLY` |

**Overall: `O33_NEITHER_HOLDS`.**
- **C63 does not hold as registered.** Its Regime A half holds, with
  `CALIBRATION` unchanged. In Regime B, D5's prior is now `BOTH`.
- **C64 does not hold.** The unclipped prior in Regime B is now `ANSWERS`.

**The expectation fails in Regime B.** C63 was expected to hold.

## Reading

Within this design:

- **In Regime A, the calibration result for D5's prior is robust.**
  - With the location matched in the world, giving the learner P0's
    calibration still lowers utility along both routes, in 6 of 6 groups
    each.
  - P0's answers still make no difference at its own calibration.
- **In Regime B, once the share rates match, the LLM's answers lower
  utility too.**
  - **For D5's prior,** along both routes: 6 of 6 and 5 of 6 groups. Its
    calibration still lowers utility along both routes (4 of 6 each), so
    the case is `BOTH`.
  - **For the unclipped prior,** also along both routes: 6 of 6 and 5 of 6.
    Its calibration's effect is mixed given F1's answers.
- **O28's Regime B sign for the unclipped prior's answers came from the
  cohort match.**
  - In O28, at the prior's own calibration, its answers raised utility in 4
    of 6 Regime B groups.
  - Once F1 is matched to that prior's share rate in the world, the same
    comparison is negative in 5 of 6.
  - Under the cohort match, F1 at that calibration had shared only 36% as
    much as the prior.
- **How the answers lower utility in Regime B (descriptive, not judged).**
  At matched calibration, a learner started from an LLM direction shares
  between 6% fewer and 0.3% more unverified items than one started from
  F1's direction. Its unverified precision is up to 0.013 lower: from
  −0.013 to +0.001 against F1's direction.
- **In Regime A, the unclipped prior's answers help at its own
  calibration.** They do so in 5 of 6 groups, as in O28 (4 of 6).

So, with share rates matched in the world:
- the cost of D5's prior is its calibration in Regime A, and both its
  calibration and the model's answers in Regime B;
- for the unclipped prior in Regime B, the cost is the model's answers.

O28's Regime B attributions depended on the cohort match.

**Boundary.**
- **Only the location was matched in the world.** The steepness stayed
  O28's, matched on the cohort.
- **The match is on how many unverified items are shared,** not on which.
- **Utility is the sharer's own.**
- **The worlds are synthetic.**
