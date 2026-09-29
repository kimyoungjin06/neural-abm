# Pivot local-LLM prior pilot v2

Date: 2026-09-29

Status: `PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION`

Machine-readable contract:
[`pivot_llm_prior_pilot_v2.yaml`](../../experiments/configs/pivot_llm_prior_pilot_v2.yaml)

## Why v2 exists

The [v1 pilot](../../experiments/evidence/results/pivot_llm_prior_pilot_v1_20260929.summary.md)
found that the Qwen prior ranked researchers about as well as the hand rule, but
at a level far too low. Its mean prior probability was 0.001, against a base
rate near 0.3. As a readiness target, that level let no agent enter. No outcome
evidence was produced, and no learner could correct the prior.

v2 asks whether putting that prior on the decision scale is enough. It is a
result-informed successor with fresh seeds. It makes **no new LLM calls**: it
reuses the v1 elicitation artifacts, bound by their hashes. The LLM is still
consulted only once.

## Calibration

Calibration uses aggregate information only, never an individual label:

- **Base-rate calibration (primary).** Keep the v1 distilled weights. Shift the
  intercept until the mean prior probability over the 1,024 design profiles
  equals a stated population base rate of 0.30. The designer's stylized worlds
  make roughly 30% of hot-field pivots productive, so 0.30 is a stated round
  aggregate, not the measured rate.
- **Hand-level calibration (control).** Keep the same weights and match the
  hand prior's mean over the design profiles. At equal level, any difference
  from the hand rule comes from ranking alone.
- **Verbalized readout.** Distill the v1 0-to-100 answers, which were on a
  moderate scale, with the same ridge fit.

## Priors

| Prior | Role |
|---|---|
| `hand` | Nested predecessor and decision reference |
| `llm_uncalibrated` | The exact v1 prior |
| `llm_base_rate` | Primary calibrated LLM prior |
| `llm_hand_level` | Ranking comparison at equal level |
| `llm_verbalized` | Alternative readout |
| `oracle_linear` | Best linear prior per world, reference |
| `flat` | No knowledge, reference |

The arms, worlds, visibility, learners, and replicate design are exactly those
of v1: fixed rule, neighbor online logistic, and cross-fitted field linear,
tree, and MLP learners, in three fit worlds under immediate visibility, with 16
replicates of 256 agents.

## Checks

The run stops as an invalid implementation if any of these fail:

- under the hand prior, the fixed and online arms must equal the persistence
  predecessor;
- the prior slot must be restored after every task;
- `llm_uncalibrated` must equal the v1 prior exactly; and
- the calibrated variants must keep the v1 weights exactly and hit their target
  means.

## Registered decisions

Each LLM variant is compared with `hand` in each world, using the v1 rules and
thresholds:

- **prior quality:** fixed-rule probe Brier, 0.01;
- **local rescue:** online-logistic probe Brier, 0.01;
- **pooled washout:** field-arm probe Brier in both directions, 0.01; and
- **entry precision:** fixed-rule productive share of entrants, 0.02.

Each decision is `VARIANT_BETTER`, `HAND_BETTER`, or `NO_CLEAR_DIFFERENCE`.
Pooled washout is `PRIOR_WASHES_OUT_UNDER_POOLING` or
`PRIOR_PERSISTS_UNDER_POOLING`, with the persisting arms and their direction.
`llm_base_rate` is the primary variant.

A prior is `SELF_SEALING_PRIOR` in a world if the fixed rule's final entry is
below 0.05 in at least 12 of 16 replicates. For every prior, the scorecard
also reports entry rates and field evidence per learner. v1 showed that a prior
decides how much evidence the population produces.

## Boundary

The executable must be committed before the development profile runs. A
deterministic rerun must reproduce every artifact byte for byte. No
post-result change to the calibration targets, priors, or thresholds is
authorized.

The worlds are synthetic. The stated base rate describes those stylized worlds,
not real pivots. The pilot makes no LLM or paid API calls, reads no empirical
rows, selects no model, and changes no generic NABM API.
