# Pivot peer-outcome contrast v1 (recommended experiment 4): result

Date: 2026-10-30

Status: `E4_NOT_INTERPRETABLE_THE_ESTIMATOR_FLAGS_STUDY2_FROZEN_ARM`

Design: [pivot-peer-outcome-contrast-v1.md](../../../docs/research/pivot-peer-outcome-contrast-v1.md),
frozen at `586f33bff0b50b4d7363555bb8d718d008be369a`.
Machine-readable record:
[pivot_peer_outcome_contrast_v1_20261030.summary.json](pivot_peer_outcome_contrast_v1_20261030.summary.json).

## Execution

- **Host:** the workstation, with 32 bootstrap workers.
- **Wall time:** 1,384 s for the development run and 1,487 s for the
  reproduction run. That is about 24 minutes each, inside the quoted 10 to
  60 minutes. Most of it is the bootstrap.
- **Reproduction:** all three identity artifacts are byte identical.
- **Data read:** the 24 discovery episodes only. No confirmatory or G4
  outcome row was read.

### Checks before any outcome

- **Study 2 parity:** the slots reproduced all 270 tracked replicate runs
  exactly. The frozen arm was identical at delays 0 and 3 in all 90 runs.
- **N11b parity:** the rebuilt rows of 177,254 exposed authors equal N11b's,
  and every check is 0.
- **Convergence:** every fit converged, in the full samples and in every
  bootstrap draw.

## The registered decision

The estimator flags learning in Study 2's frozen arm, which does not learn.
By the registered rule, the data's visibility DiD is not interpreted.

| Expectation | Result |
|---|---|
| ES2: imitative outcome gap above frozen's in baseline and grants | Failed. Grants: +7.03 [+6.44, +7.69]. Baseline: undefined; the frozen arm had 9 events after a visible failure, below the registered 10 |
| ES3: frozen arm's visibility DiD covers zero at delay 3 | **Failed.** Grants: −0.785 [−1.156, −0.371]. Hype: +0.199 [−0.295, +0.737]. Baseline: undefined |
| EE1: a visible coauthor entry goes with more entry | Held: +0.520 [+0.367, +0.739] log-odds |
| EE2: homophily before the outcome is visible | Failed: +0.290 [−0.553, +0.583] |

## Study 2 under the estimator

Delay 3, log-odds:

| Scenario | Arm | Visibility DiD | Success-following | Failure-avoidance |
|---|---|---|---|---|
| baseline | imitative | +11.02 [+10.25, +11.88] | +6.73 [+6.20, +7.46] | −4.29 [−4.87, −3.62] |
| grants | frozen | −0.79 [−1.16, −0.37] | −0.52 [−0.91, −0.09] | +0.27 [+0.08, +0.43] |
| grants | imitative | +9.67 [+9.00, +10.43] | +5.15 [+4.66, +5.75] | −4.52 [−4.84, −4.20] |
| hype | frozen | +0.20 [−0.29, +0.74] | +0.27 [+0.00, +0.60] | +0.07 [−0.33, +0.47] |
| hype | imitative | +10.21 [+9.85, +10.58] | +5.52 [+5.23, +5.91] | −4.68 [−5.08, −4.28] |

- **The imitative arm is recovered in direction in every scenario.**
- **The cautionary arm never pivots after a visible failure.** It has 0
  such events in every scenario; after a pending failure it has 3, 113, and
  356. Its failure-avoidance contrasts are therefore undefined under the
  10-event rule.
- **The frozen arm shows a nonzero DiD in the grant scenario.** Its pending
  gap is positive (+0.37) and its visible gap negative (−0.42). Its dynamics
  are identical at both delays, so only the labeling of the states differs.
  The association is positive in the first steps after a neighbor's pivot
  and negative later.
  - A plausible reading, not tested here: agents like a successful neighbor
    pivot early, and those still at risk later are the less inclined ones.

## The data

The 24 discovery episodes hold 10,418 entrants, 6,291 of them precise.

- **Exposure:**
  - 503,879 dyads, 46% of them with an S1-entrant coauthor;
  - 177,254 exposed authors;
  - 634,492 rows with a visible coauthor entry.
- **Events by state.** These are precise, unrecruited entries among exposed
  rows:

  | State | S | F | P_S | P_F | U |
  |---|---:|---:|---:|---:|---:|
  | Events | 17 | 60 | 32 | 110 | 212 |

- **Estimates (log-odds):**

  | Quantity | Estimate |
  |---|---|
  | Exposure, unrecruited entries | +0.520 [+0.367, +0.739] |
  | Exposure, recruited entries counted as events | +1.078 [+0.948, +1.323] |
  | Outcome gap, visible (S − F) | +0.228 [−0.519, +0.789] |
  | Outcome gap, before visible (P_S − P_F) | +0.290 [−0.553, +0.583] |
  | Visibility DiD | −0.062 [−0.492, +0.538] |
  | Success-following | −0.363 [−0.727, +0.247] |
  | Failure-avoidance | −0.302 [−0.749, +0.114] |
  | Entry-order gradient of the label | +0.010 [−0.224, +0.287] |

- **Descriptive shares of precise entrants:**
  - 14.6% have a follow-up topic work within four quarters;
  - 12.2% entered with an earlier entrant as coauthor (recruited);
  - 51.4% have a topic incumbent on their entry work.

## Post hoc, descriptive, not registered

- **The data bound the outcome dependence far below Study 2's imitative
  arm.**
  - Under the same estimator, the imitative arm's visible outcome gap is at
    least 6.95 log-odds, and its visibility DiD at least 9.67.
  - The data's upper bounds are 0.79 and 0.54.
  - The frozen arm's largest bias, 0.79, cannot close that distance.
  - Study 2's learning rate of 1.5 makes agents react to one neighbor's
    outcome far more strongly than coauthors' entries react to it in these
    data. Pivots and topic entries are not the same event, so this compares
    one log-odds scale across two designs.
- **The entry-order gradient is flat in the data**, +0.010
  [−0.224, +0.287]. Study 2's imitative loop needs later entrants to fail
  more, and its imitative arm shows −0.43 to −8.04. The frozen arm shows
  −1.37 to +0.56.
- **Most data entrants do not follow up within a year**, 14.6%. In Study 2,
  33% to 82% of pivots are productive. The two outcome definitions differ.
- **Coauthor entry exposure is modest.** In the data it is +0.52
  (+1.08 with recruitment). Study 2's frozen arm gives +0.39 to +1.27 and
  its imitative arm +0.60 to +4.54.

## What this means

- **This design cannot read learning direction from coauthor data.**
  - The visibility placebo is biased in Study 2's own no-learning arm by
    more than the data's interval half-width, 0.51.
  - The data hold too few events after a visible outcome (17 and 60) for a
    narrower test.
- **Running the generator first caught the flaw before any reading.** The
  design was checked on a world whose truth is known, and the registered
  rule withheld the data reading.
- **Study 2's imitative arm is far from these data.** Its strength
  parameter implies outcome-dependent peer responses that the data bound
  well below; this is post hoc. Its learning rules remain postulated.

## Boundary

- **Partition.** Discovery only; the confirmatory outcomes stay sealed.
- **Associations, not effects.** No result here identifies learning.
- **Two log-odds scales.** The post-hoc comparisons put Study 2's agent
  steps beside OpenAlex quarters on one log-odds scale. They are
  descriptive.
- **Scope.** No LLM call, and no change to Study 2's code or to a generic
  NABM API.
