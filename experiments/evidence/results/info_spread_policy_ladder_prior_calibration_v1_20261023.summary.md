# Information-spread policy ladder: prior-calibration study v1 (O28)

Date: 2026-10-23

Status: `O28_ATTRIBUTION_DIFFERS_ACROSS_COSTED_CASES`

Design: [prior-calibration study v1](../../../docs/research/info-spread-policy-ladder-prior-calibration-v1.md),
frozen at `555fd49` before the run.

## What ran

Each LLM prior's direction (the model's answers) was crossed with its
calibration (location and steepness over D3b's design cohort), against the
hand rule F1. This was done for each regime and for both LLM priors, P0 and
P0u.

**New arms.**
- **Four learner arms.** Study v2's learner, with its seeds, started from a
  recalibrated prior:
  - L1_hcal and L1u_hcal: the LLM priors' answers at F1's calibration;
  - L0_pcal and L0_ucal: F1's answers at the LLM priors' calibration.
- **Four fixed priors,** the recalibrated priors themselves, for the
  calibration check.
- **Design.** All ran in study v2's 12 study groups, with 20 cells per group
  and 16 replicates per cell.

**Recorded arms.** L0, L1, and L1u were read from their records.
- Regime A utility came from the recorded counts.
- Regime B utility came from O25's development cells.

**Checks.**
- **Parity.** L0 was recomputed at replicate 0 of all 240 learner tasks. It
  equalled its record in every task, including its utility.
- **Consistency.** In every one of the 24 group totals (L1 − L0 and L1u −
  L0), the mean cell difference equalled O25's recorded value.

**Runs.**
- GPU-Server, on GPU 2 with 24 workers.
- **Times.** 271 minutes for the development run and 272 for the
  reproduction.
- **The estimate missed.**
  - Quoted from the pilot: 2.5–3.5 hours.
  - Revised after the first wave: 4–5 hours.
  - Why: the pilot ran 12 MLP tasks on the GPU at once. The full run starts
    with 24, and each took 2.1 times as long.
- `cells.jsonl` and `summary.json` match byte for byte between the two runs.

## The calibration check

**All 8 recalibrated fixed priors pass the registered rule.** Each shares
closer to its calibration source than to its direction source. The rule is
satisfied, but in the world the recalibration overshoots in both directions:

| Recalibrated prior | Regime A shares | Regime B shares | Calibration source (A / B) | Direction source (A / B) |
|---|---:|---:|---|---|
| P0_hcal | 464k | 529k | F1: 343k / 381k | P0: 0.5k / 189k |
| P0u_hcal | 497k | 566k | F1: 343k / 381k | P0u: 229k / 223k |
| F1_pcal | 0.5k | 87k | P0: 0.5k / 189k | F1: 343k / 381k |
| F1_ucal | 45k | 80k | P0u: 229k / 223k | F1: 343k / 381k |

Shares are mean unverified shares per run.

**Why the overshoot matters.** The cohort draws item features and contacts
uniformly, but the world does not. So "F1's calibration" makes an LLM
direction share 1.35 to 1.48 times as much as F1 itself. An "LLM
calibration" makes F1's direction share less than the LLM prior, or in P0/A
as little.

## Registered result

The measure is utility per agent on unverified items, with O25's minimum
effects: 0.5895 in Regime A and 0.1549 in Regime B. Each cell gives the range
of the mean cell Δ over the regime's 6 study groups. The labels follow
group order: linear (axis, linear, smooth), then MLP (axis, linear, smooth).

| Case | Calibration, LLM answers | Calibration, hand answers | Answers, F1 calibration | Answers, LLM calibration | Attribution |
|---|---|---|---|---|---|
| **P0/A** (costed) | −8.2 to −3.8 (− − − − − −) | −15.3 to −6.4 (− − − − − −) | −7.9 to −0.6 (− − − − 0 −) | −0.0 to +0.3 (0 0 0 0 0 0) | `CALIBRATION` |
| **P0/B** (costed) | −5.1 to +0.6 (− − − − 0 0) | −10.1 to +3.3 (− − − − + +) | −3.1 to −0.3 (− − − − 0 0) | −3.2 to +1.9 (+ + + 0 − −) | `CALIBRATION` |
| **P0u/B** (costed) | −3.1 to +4.7 (− − 0 − + +) | −10.5 to +2.8 (− − − − + +) | −4.4 to −1.0 (− − − − − −) | −0.2 to +3.0 (+ + + 0 + 0) | `NEITHER_CONSISTENTLY` |
| P0u/A (reported) | −6.7 to +3.5 (+ 0 + − − −) | −13.6 to +1.4 (− − − − 0 +) | −4.7 to +6.3 (− 0 − − + +) | −0.1 to +11.6 (+ + + + 0 0) | `NEITHER_CONSISTENTLY` |

**The expectation fails for one costed case.** `CALIBRATION` was expected
in every costed case. It holds for D5's prior in both regimes, but not for
the unclipped prior in Regime B.

## Reading

Within this design:

- **For D5's prior, the cost is its calibration.**
  - Along both routes, giving the learner P0's calibration lowers utility:
    in 6 and 6 of 6 Regime A groups, and 4 and 4 of 6 Regime B groups.
  - At P0's own calibration, its answers make no material difference in
    Regime A, and a mixed one in Regime B.
  - In Regime A this is expected. P0/A's logit sd over the cohort is 0.06,
    so at that calibration the direction hardly matters.
- **For the unclipped prior, the two factors interact.**
  - Its answers lower utility at F1's calibration: 6 of 6 Regime B groups.
  - They raise it at its own calibration: 4 of 6 groups in each regime.
  - The calibration lowers utility given F1's answers (4 of 6) and is mixed
    given the model's.
  - So the model's answers pay off only with the selective calibration
    they came with.
  - One caveat: at F1's calibration the LLM direction shares about 1.5
    times as much as F1 in the world. That comparison carries the overshoot
    too.
- **Descriptive, not judged.** Compared with the same answers at F1's
  calibration, learners started from an LLM calibration share fewer
  unverified items, at about the same or higher precision:

  | Regime | Fewer unverified shares | Precision difference |
  |---|---|---|
  | A | 9–25% | −0.003 to +0.032 |
  | B | 6–15% | −0.000 to +0.023 |

  This fits the selectivity explanation of O25.
- **Two Regime B groups go the other way.** In partial MLP on the linear
  and smooth worlds, the LLM calibration raises utility in 3 of the 4
  calibration comparisons. These are the two groups where O25 found L1u −
  L0 positive.

So the utility cost of D5's prior inside the learner traces to how it was
calibrated (D3b's shift to the base rate), not to the model's answers. For
the unclipped prior, neither factor alone carries it.

**Boundary.**
- **Location and steepness moved together.**
- **The recalibration was matched on the cohort, and it overshoots in the
  world** (see the calibration check).
- **P0/A's direction is mostly the clipped readout's noise.**
- **Utility is the sharer's own.**
- **The worlds are synthetic.**
