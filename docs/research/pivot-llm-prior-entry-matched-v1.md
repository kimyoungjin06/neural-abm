# Pivot LLM-prior entry-matched control v1

Date: 2026-10-29

Status: `PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION`

Machine-readable contract:
[`pivot_llm_prior_entry_matched_v1.yaml`](../../experiments/configs/pivot_llm_prior_entry_matched_v1.yaml)

## Why this study exists

C22 recorded that the calibrated, once-elicited LLM prior with a 20%
exploration floor dominated the hand rule in 7 of 9 world-and-learner cells of
[pilot v3](../../experiments/evidence/results/pivot_llm_prior_pilot_v3_20260929.summary.md).
Its entrant-precision gain was large and consistent. Three records say that the
gain may not come from what the model knows:

- **The LLM prior is far more selective.** Under the fixed rule it admits about
  0.34 of agents, against about 0.80 for the hand rule. A stricter rule raises
  the productive share of entrants even when it ranks no better.
- **The v2 hand-level control did not isolate ranking.** It matched the hand
  rule's mean prior probability, but entry stayed at 0.465 against 0.823.
- **The ranking itself is no better in the interaction worlds.** In v4, the
  design-cohort AUC of the v3 prior was 0.632 against the hand rule's 0.622 in
  the smooth world and 0.644 against 0.643 in the axis world. Only in the
  linear world was it higher, 0.707 against 0.647.

Two further records also apply. C22's learning test compared means only: with
an interval-based non-inferiority test, 2 of the 9 cells pass. And O33 showed
that a calibration matched on a design cohort can miss in the world.

This study separates a prior's ranking from its selectivity, with both
measured in the world.

## Direction and selectivity

A prior has two parts:
- its **direction**, the feature weights; and
- its **selectivity**, the mean fixed-rule entry rate it produces in the world.

A matched condition keeps one direction's weights exactly and shifts only its
intercept. The shift is set so that its fixed-rule entry in the world equals
the other direction's native entry. Comparing two conditions at the same
selectivity then isolates direction, which here means ranking.

| Condition | Direction | Selectivity | Exploratory share |
|---|---|---|---:|
| `hand` | hand rule | native | 0 |
| `llm_selective` | v2 LLM prior | native | 0 |
| `hybrid_20` | v2 LLM prior | native | 0.2 |
| `hand_at_llm_entry` | hand rule | matched to `llm_selective` | 0 |
| `hand_at_llm_entry_20` | hand rule | matched to `llm_selective` | 0.2 |
| `llm_at_hand_entry` | v2 LLM prior | matched to `hand` | 0 |

The exploration floor is the v3 adapter, unchanged. `hand_at_llm_entry_20`
uses the same intercept shift as `hand_at_llm_entry`, so it differs from
`hybrid_20` only in direction.

## Entry calibration in the world

The intercept shifts are fitted before the study replicates are run, on a
separate calibration seed namespace of 16 replicates. The fit is done for each
world and each matched level:

1. Run the source direction at its native level under the fixed rule, and take
   its mean cumulative entry as the target.
2. Bisect the matched direction's intercept shift over [−12, 12] for 40
   iterations. Each evaluation runs the fixed rule on the same calibration
   replicates.
3. Keep the evaluated shift with the smallest entry gap. The in-sample entry
   ratio must lie within 1%.

The calibration stops if the bracket does not contain the target or the
tolerance is missed. In the study replicates, the fixed-rule entry of each
matched condition without exploration, divided by its source's, must lie in
[0.95, 1.05] in every world. Otherwise the run is recorded as
`ENTRY_MATCH_FAILED_STOP`, and no decision is interpreted. The exploration
pair's entry ratio is reported but not checked.

## Registered decisions

Precision uses the C22 rule: the productive share of entrants must improve by
at least 0.02, and in at least 12 of 16 replicates. It is applied in both
directions, giving `FIRST_BETTER`, `SECOND_BETTER`, or `NO_CLEAR_DIFFERENCE`.
Each cell also reports the paired mean and its 95% t interval.

Learning non-inferiority is now interval-based. The lower bound of the 95% t
interval of the reference's probe Brier minus the condition's must be at least
−0.005. The mean-only rule C22 used is reported for continuity, but it does not
decide anything.

| Comparison | First | Second | Arms |
|---|---|---|---|
| Ranking at the LLM's selectivity, no learning (primary) | `llm_selective` | `hand_at_llm_entry` | fixed rule |
| Ranking at the LLM's selectivity, pooled learners | `llm_selective` | `hand_at_llm_entry` | linear, tree, MLP |
| Ranking at the hand rule's selectivity | `llm_at_hand_entry` | `hand` | all four |
| C22 against the entry-matched hand rule | `hybrid_20` | `hand_at_llm_entry_20` | linear, tree, MLP |
| C22 replication on fresh seeds | `hybrid_20` | `hand` | linear, tree, MLP |

The two C22 comparisons classify each cell as `LLM_DOMINATES`,
`SELECTION_ONLY`, `LEARNING_ONLY`, or `NEITHER`. The study status counts how
many of the 9 cells keep a precision gain against the entry-matched hand rule:

| Status | Cells with `FIRST_BETTER` precision |
|---|---|
| `C22_PRECISION_SURVIVES_ENTRY_MATCHING` | 5 to 9 |
| `C22_PRECISION_SURVIVES_IN_MINORITY` | 1 to 4 |
| `C22_PRECISION_IS_SELECTIVITY` | 0 |

## Expectations before the run

These follow from the v4 design-cohort AUC above. Each will be recorded as held
or failed, and a failed expectation is recorded as a rejected claim.

- **E1.** At the LLM's selectivity, the fixed-rule comparison is `FIRST_BETTER`
  in the linear world.
- **E2.** It is `NO_CLEAR_DIFFERENCE` in both interaction worlds.
- **E3.** C22 against the entry-matched hand rule is
  `C22_PRECISION_SURVIVES_IN_MINORITY`, and every cell that keeps the precision
  gain is in the linear world.

## Checks

The run stops as an invalid implementation if any of these fail:
- under the hand condition, the fixed arm equals the persistence predecessor
  in the linear world;
- the hand prior passed through the prior slot at zero shift equals the hand
  rule on every calibration replicate;
- the prior and adapter slots are restored after every task;
- the LLM prior equals the v2 base-rate prior exactly, by hash; and
- every matched prior keeps its direction's weights exactly.

## Design

The arms, worlds, visibility, learners, and replicate design are those of v3:
- the fixed rule and the cross-fitted field linear, tree, and MLP learners;
- three fit worlds under immediate visibility;
- 16 replicates of 256 agents; and
- fresh seed namespaces for the study (20261201), its probe cohort (20261202),
  the calibration (20261203), and the calibration probe cohort (20261204).

The executable must be committed before the development profile runs. A
deterministic rerun on the same host must reproduce every artifact byte for
byte. No post-result change to the shifts, bounds, thresholds, or decisions is
authorized.

## Boundary

The worlds are synthetic. The study makes no LLM or paid API calls, reads no
empirical rows, selects no model, and changes no generic NABM API. It tests
whether this LLM prior ranks better than this hand rule in these worlds. It
says nothing about LLM knowledge of real pivots.
