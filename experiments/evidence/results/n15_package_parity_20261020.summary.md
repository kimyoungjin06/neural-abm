# N15 parity: recorded Pivot runs through the package

Date: 2026-10-20

Status: `N15_PARITY_HOLDS_THE_PACKAGE_REPRODUCES_BOTH_PIVOT_RECORDS`

Specification: [N15 parity](../../../docs/research/n15-package-parity-v1.md),
fixed at `cb0aa44` before the runs.

## What ran

The package modules of Decision 0017 were placed in Pivot's learner slot:

- `neural_abm.policy_ladder`: the prior, keyed randomness, the exploration
  floor, the outcome record, and the trees;
- `neural_abm.policy_learners`: the residual linear and MLP learners.

Two recorded Pivot runs were reproduced on the workstation where they were
recorded.

| Run | Target | Rows identical to the record | Whole file | Wall time |
|---|---|---:|---|---:|
| `v3` | Pivot LLM-prior pilot v3 (`87c90eb`), all 288 tasks | 1,152 of 1,152 | **Identical**, sha256 `1f532d98…` | 639 s |
| `abm2`, release insertion | Nonlinear policy ABM v2 (`c56d741`), field-public linear, tree, and MLP arms, three visibility regimes | 432 of 432 | (a subset) | 185 s |
| `abm2`, event insertion | The same, with outcomes added when realized and tagged with their release step | 432 of 432 | (a subset) | 171 s |

Every group matches in every row: each of the 9 visibility regime and arm
pairs (48 of 48 each), and each v3 arm (288 of 288). The quoted estimate for
the three runs was 15–30 minutes; they took 17.

## The ordering rule (O27)

**The rule.** The package's outcome record returns rows ordered by the step
from which they are visible, then by insertion.

**What the event-insertion run shows.**
- It added each Pivot outcome when it was realized, tagged with its release
  step.
- So Pivot's release order came only from this rule, and it reproduced every
  delayed row.
- Before the freeze, the same run on the frozen core's insertion-order record
  differed from Pivot under both failure delays.

So the rule answers O27. A caller can add outcomes either when they are
realized or when they are released.

## Reading

- **The package reproduces both recorded Pivot runs byte for byte.** This
  holds at immediate visibility and under delayed failures.
- **It changes no global torch state.**
- **Decision 0017's conditions for acceptance hold.** The outcome record is
  extracted with the other primitives.

**Boundary.** Two recorded configurations, on one host. No Pivot result is
reinterpreted, and no frozen script or released tag is changed.
