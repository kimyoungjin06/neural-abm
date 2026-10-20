# Information-spread policy ladder: study v1, stage D4

Date: 2026-10-16

Status: `D4_DESIGN_DEFECT_IN_NULL_WORLD_STUDY_GROUPS_NOT_READ`

Design: [study v1 (D3a)](../../../docs/research/info-spread-policy-ladder-study-v1.md),
frozen at `daf5d80` before the run.

## What ran

The frozen study ran the five non-LLM arms (F0, F1, T0-1, T0-4, and L0) on
the GPU server, with GPU 2 for the MLPs and 24 workers:

- 16 groups (2 regimes × 2 families × 4 geometries);
- 20 cells per group and 16 replicates per cell.

Run times:

| Run | Wall time |
|---|---:|
| Development | 135 min |
| Reproduction | 134 min |

`cells.jsonl` and `summary.json` match byte for byte between the two runs.

The quoted estimate was 70–140 minutes, from a full-load pilot with 2 cells
per group. The full run's per-task times exceeded the pilot's:

| Family | Mean task time, full run | Pilot |
|---|---:|---:|
| A/mlp | 819 s | 489 s |
| B/partial_mlp | 905 s | 540 s |
| A/linear_exact | 205 s | 59 s |
| B/partial_linear_exact | 458 s | 156 s |

## Registered result: the null-world check fails

The contract's design check requires that L0 − F1 not be material and
positive in any null group. It is in one:

| Null group | L0 − F1, mean cell Δ | Interval | Material |
|---|---:|---|---|
| A/linear_exact | −0.004 | [−0.012, +0.003] | No |
| A/mlp | +0.009 | [+0.006, +0.013] | No (below 0.01) |
| B/partial_linear_exact | +0.002 | [−0.002, +0.005] | No |
| **B/partial_mlp** | **+0.027** | **[+0.025, +0.030]** | **Yes** |

The disposition is therefore `DESIGN_DEFECT_IN_NULL_WORLD`. As the contract
states, this blocks any reading of the study groups.

The runner still computed the registered labels. They are kept in the JSON
and are not read:

- closed loop: `MIXED`;
- best table: `BEST_TABLE_BEATS_ADAPTIVE`.

## What the defect means

In the null world the features carry no information about veracity. A
learner that beats the base rate there must be using information that the
design did not intend to give it.

An exploratory diagnosis, recorded separately, locates that information:
- [null-leak diagnosis](info_spread_policy_ladder_null_leak_diagnosis_20261018.summary.md).

## Next

The user chose on 2026-10-18 to change the primary measure: share precision
on items whose fact-check is not yet public at the decision. This needs:

- a check on controls;
- a new freeze;
- a new D4 run.

The LLM arms (D3b) are drafted, but not frozen until then.
