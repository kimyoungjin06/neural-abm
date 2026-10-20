# Information-spread policy ladder: study v2 (D3a v2)

Date: 2026-10-18

Status: `PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION`

Machine-readable contract:
[`info_spread_policy_ladder_study_v2.yaml`](../../experiments/configs/info_spread_policy_ladder_study_v2.yaml)

Study v2 is [study v1](info-spread-policy-ladder-study-v1.md) with a new
primary measure. Its run is stage D4 v2.

## Why v2

v1's run (D4) failed its null-world check.
- An MLP learner recognizes items it has seen fact-checked.
- So share precision over all items rewarded that recognition, even where
  features carry no information.

Records:

- [D4](../../experiments/evidence/results/info_spread_policy_ladder_study_v1_d4_20261016.summary.md);
- [the diagnosis](../../experiments/evidence/results/info_spread_policy_ladder_null_leak_diagnosis_20261018.summary.md).

On 2026-10-18 the user chose a new primary measure. D2d then tested the group
rule on it ([specification](info-spread-policy-ladder-d2d-v1.md)).

## What changes

| | v1 | v2 |
|---|---|---|
| Primary measure | Share precision over all items | **Share precision on unverified items:** items whose fact-check is not public at the decision (never checked, or still pending) |
| Null check | L0 − F1 on share precision | L0 − F1 on the primary measure |
| World seeds | 6000 + cell | **7000 + cell**; learner and rule tags 9 |
| F0 | Threshold rule | The same decisions, in probability form so that it can be measured |

Share precision over all items and false reach are still recorded.

**How the measure is computed.** It does not change the simulator
(`scripts/info_spread_unverified_measure_v1.py`).
- A wrapper logs each decision and reapplies the exploration floor's keyed
  draws.
- It classifies each decision by the item's fact-check state at that step.
- Every run checks the reconstructed shares per item, and the decision count,
  against the simulator's own.

## What stays

Everything else is v1's:

- **World:** 1,000 agents, 60 steps, 20 items per step, a 4-step delay,
  half of the items fact-checked, ε = 0.02, public fact-checks with one row
  per agent and item.
- **Regimes:** A and B, as in v1.
- **Arms:** F0, F1, T0-1, T0-4, and L0.
- **Learners:** exact linear fits; MLPs at the Pivot settings on GPU 2;
  partial pooling in Regime B.
- **Design:** 16 groups (regime × family × geometry, with a null geometry),
  20 cells, and 16 replicates.

## Decision rule

The group rule, on the primary measure:

- the cell-resampled 95% interval of the mean cell Δ excludes zero;
- and that mean is at least 0.01 in magnitude.

The dispositions are v1's, with each regime's 6 study groups counted:

| Disposition | Condition |
|---|---|
| `AS_EXPECTED` | L0 − T0-1 material and positive in at least 4 of 6 groups in Regime B, and in at most 1 of 6 in Regime A |
| `TABLE_SUFFICES_EVEN_IN_B` | At most 1 of 6 in Regime B |
| `CLOSED_LOOP_MATTERS_EVEN_IN_A` | At least 4 of 6 in Regime A |
| `MIXED` | Otherwise |

L0 − T0-4 is labelled per group as in v1.

**Design check.** On the primary measure, L0 − F1 must not be material and
positive in any null group. A violation is `DESIGN_DEFECT_IN_NULL_WORLD` and
blocks any reading of the study groups.

## Runs and record

1. **Pilot.** A full-load pilot with 2 cells per group, for the time
   estimate.
2. **Development and reproduction runs.** On GPU-Server, with `cells.jsonl`
   and `summary.json` byte identical between the two.
3. **Record.** A summary and a result test.

The LLM arms are frozen with this study, as D3b on the same measure.

## Boundary

Synthetic worlds only. No empirical data and no LLM. The study makes no
claim about real information spread or real people.
