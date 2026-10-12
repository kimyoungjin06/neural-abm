# Decision 0016: Out-of-Sample Policy Scoring Contract

## Status

Accepted for the `main` / next-alpha candidate.

## Date

2026-10-12

## Context

The Pivot policy-ladder consolidation (repository-only:
`docs/research/pivot-policy-ladder-consolidation-v1.md`) applied the bridge
rule: a Pivot capability is generalized only when a second model family needs
the same semantics. Of its framework capabilities, only out-of-sample agent
policy scoring (F03) met that rule. It was recorded as an extraction
candidate, to be designed before any API change.

The two consumers score every agent's policy on data the agent never trained
on:

| Consumer | Held-out data | Per-policy score | Aggregate |
| --- | --- | --- | --- |
| `toy_classification` (Toy 1) | A labelled test split and a probe split | Accuracy and cross-entropy of each agent's class distribution | Mean over agents |
| Pivot ABM v2 (repository-only) | A probe cohort of researchers never seen in training | Brier score of each learner's success probability | Mean over learners, and AUC of the mean prediction |

The consolidation also recorded a lesson (F04): scoring a policy in situ leaks
when the learner already holds the scored agent's own outcome. Pivot ABM v2
answered it with a separate probe cohort and a fold-isolation check.

## Decision

The `main` / next-alpha candidate adds a torch-free scoring surface,
`neural_abm.policy_scoring`, re-exported through `neural_abm.api_lite`, as
the scenario surface of Decision 0015 is:

- `score_policies(predictions, labels, *, metrics, split, held_out_ids,
  training_ids)` returns a `PolicyScoreReport`;
- `PolicyScore` holds one policy's scores;
- `PolicyScoringLeakError` is raised when a policy was trained on a held-out
  unit;
- `POLICY_SCORING_METRICS` names the supported metrics.

### Inputs

- `predictions` maps each policy id to its predictions for the same `n`
  held-out examples. It accepts either:
  - a probability vector of shape `(n,)`, read as the probability of class 1
    in a binary task;
  - a row-stochastic matrix of shape `(n, k)`.
- `labels` are integer class indices of shape `(n,)`.
- `held_out_ids` and `training_ids` are optional. When `training_ids` maps a
  policy to the units it trained on, `held_out_ids` must name the `n`
  held-out units. Any overlap raises `PolicyScoringLeakError` before a score
  is computed.

### Metrics

Each metric reproduces its consumer's existing convention exactly.

| Metric | Definition |
| --- | --- |
| `brier` | Binary: mean of (p₁ − y)². With k > 2 classes: mean over examples of Σⱼ (pⱼ − 1[y = j])² |
| `log_loss` | Mean of −log(p_y + 10⁻⁸), Toy 1's convention |
| `accuracy` | Share of examples whose first highest-probability class equals the label |
| `auc` | Binary only. Rank-based area under the ROC curve, with ties given average ranks. `None` when the labels hold a single class |

### Report

- Per-policy scores.
- `mean_scores`: the unweighted mean over policies of each metric. This is
  Toy 1's `domain_mean_*_accuracy` and Pivot's `probe_brier`.
- `ensemble_scores`: each metric of the mean predicted distribution. This is
  Pivot's `probe_auc`.
- `to_payload()` returns a JSON-ready mapping.

## Boundary

The surface computes scores. It does not compare policies against a margin,
decide which policy class wins, or judge a claim. Thresholds, paired
differences, bootstrap rules, and their meaning stay with the caller, as in
Decisions 0013 and 0015.

The leak check enforces only what the caller declares. Isolation schemes
such as Pivot's cross-fitted folds remain domain-owned (F04). So do the choice
of held-out data and of the unit ids.

## Dependency and Release Boundary

The module uses NumPy only and must not import `torch`, `scipy`, or
`scikit-learn`. The released `v0.1.0a5` tag predates it.

## Implementation Consequence

1. Re-export the listed objects through `neural_abm.api_lite` and keep its
   export set explicit in tests. The torch-backed `neural_abm.api` facade is
   unchanged.
2. Parity tests show that both consumers already compute these scores:
   - `score_policies` reproduces Toy 1's `accuracy_from_probs` and
     `cross_entropy_from_probs`;
   - it reproduces Pivot ABM v2's `_brier` and `_auc` on shared fixtures.
3. Neither consumer's source is edited. The consolidation ledger binds
   `toy_classification.py` by hash, and the Pivot scripts are frozen. New
   callers use `score_policies`.
4. The consolidation ledger's F03 entry is met by this decision. F04, F09,
   F10, and F11 stay domain-owned.

## Non-Goals

This decision does not:

- add a model-comparison or statistical-testing framework;
- extract cross-fitting, outcome queues, prior slots, or exploration floors;
- change any Pivot result or frozen script;
- claim that `v0.1.0a5` contains the scoring surface.
