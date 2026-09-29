# Pivot local-LLM prior pilot v3

Date: 2026-09-29

Status: `PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION`

Machine-readable contract:
[`pivot_llm_prior_pilot_v3.yaml`](../../experiments/configs/pivot_llm_prior_pilot_v3.yaml)

## Why v3 exists

The [v2 pilot](../../experiments/evidence/results/pivot_llm_prior_pilot_v2_20260929.summary.md)
found a trade-off. The calibrated Qwen prior picks better entrants than the
hand rule, but it admits only about a third of agents. That leaves pooled
learners about 30% less evidence, and in the interaction worlds they end up
worse. A permissive prior produces the most evidence, but it does not select.

v3 tests whether the two roles can be separated. Knowledge picks who enters,
and a small, prior-independent exploration floor keeps evidence flowing. It is
a result-informed successor with fresh seeds. It makes no LLM calls and reuses
the v2 base-rate prior by hash.

## Exploration floor

At each open step, a not-yet-entered agent whose readiness is below the pivot
threshold still enters if a uniform draw falls below `ε`. The draw is keyed to
(seed, replicate, step, agent), so it perturbs no other randomness. `ε` is set so
that the targeted share of non-ready agents explores at least once over the 16
open steps:

`ε = 1 − (1 − share)^(1/16)`

| Condition | Prior | Exploratory share | ε |
|---|---|---:|---:|
| `hand` | hand rule | 0 | 0 |
| `llm_selective` | v2 base-rate LLM prior | 0 | 0 |
| `hybrid_10` | v2 base-rate LLM prior | 0.1 | 0.0066 |
| `hybrid_20` | v2 base-rate LLM prior | 0.2 | 0.0139 |
| `hybrid_40` | v2 base-rate LLM prior | 0.4 | 0.0314 |
| `flat` | no knowledge | 0 | 0 |

At `ε = 0` the adapter is exactly the persistence predecessor. It is injected
through an adapter slot that is restored after every task, in the same way as
the prior slot.

The arms are the fixed rule and the cross-fitted field linear, tree, and MLP
learners. They run in three fit worlds under immediate visibility, with 16
replicates of 256 agents.

## Registered decisions

Each hybrid is compared with `hand` in each of the nine combinations of world
and field arm:

- **Precision:** the field arm's productive share of entrants must improve by
  at least 0.02, in at least 12 of 16 replicates.
- **Learning non-inferior:** the mean probe Brier may be at most 0.005 worse
  than under the hand rule.
- **Learning restored:** probe Brier must improve over `llm_selective` by at
  least 0.01, in at least 12 of 16 replicates.

Each cell is classified as follows:

| Disposition | Condition |
|---|---|
| `HYBRID_DOMINATES_HAND` | precision passes and learning is non-inferior |
| `SELECTION_ONLY` | only precision passes |
| `LEARNING_ONLY` | only learning is non-inferior |
| `NEITHER` | neither passes |

Across the nine cells, each hybrid is then summarized as dominating in all
cells, in a majority, in a minority, or nowhere. `llm_selective` and `flat` are
scored the same way as references. Every condition also reports entry,
exploratory entry, and evidence per learner.

## Boundary

The executable must be committed before the development profile runs. A
deterministic rerun must reproduce every artifact byte for byte. No post-result
change to the shares, priors, or thresholds is authorized, and no share is
selected as best.

The worlds are synthetic. The pilot makes no LLM or paid API calls, reads no
empirical rows, and changes no generic NABM API.
