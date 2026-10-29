# Decision 0017: Policy-Ladder Semantics Extraction (N14, N15)

## Status

Accepted, and released in `v0.1.0a6`. Proposed on 2026-10-19 (N14).
The user approved its implementation on 2026-10-20 (N15), and it was
accepted once the parity conditions below held.

## Date

2026-10-19, accepted 2026-10-20

## Context

The Pivot policy ladder compared fixed rules, learned policies, and external
priors in one simulation lifecycle. Its machinery stayed repository-only
under the bridge rule: a capability enters the package only when a second
model family needs the same semantics. Consolidation v5 kept these items
domain-owned until then:

- the shared evidence record;
- the prior slot (F09);
- the exploration floor (F10).

N14 tested the bridge rule in three steps.

1. **A second consumer.**
   - The information-spread policy ladder ran the same comparison in a new
     domain, through a domain-free core (`scripts/policy_ladder_core_v1.py`).
   - Its arms nest exactly: 14 checks in D1.
   - Its registered studies reproduced byte for byte: D4 v2 and D5.
2. **The first consumer through the same core.**
   - The Pivot LLM-prior pilot v3 was re-expressed through the core.
   - It reproduced its recorded `runs.jsonl` byte for byte: 1,152 of 1,152
     rows, sha256 `1f532d98…`. The repository-only record is
     `experiments/evidence/results/n14_pivot_parity_20261019.summary.md`.
3. **Learner parity.** With the Pivot prior, the core's learners reproduce
   Pivot's fitters byte for byte (`tests/test_policy_ladder_core_v1.py`).

The re-expression also showed where the two consumers differ. Those
differences decide what stays domain-owned.

## What the two consumers share

| Semantics | Pivot (through the core) | Information spread | Proposal |
|---|---|---|---|
| A linear prior rule: logit = x · w + b | The learners' anchor | The hand rule F1, the LLM prior P0, and every learner's anchor | **Extract** (torch-free) |
| Keyed randomness: one seed or uniform stream per tuple of integers | Refit seeds and exploration draws | World, stream, learner, and exploration draws | **Extract** (torch-free) |
| An exploration floor: take a declined action with probability ε on a keyed draw, independent of the prior | Exploratory entry | Exploratory sharing | **Extract** (torch-free) |
| An outcome record: rows that become visible from a step, with pooled and per-agent views | Fold learners' evidence, immediate and delayed | Delayed fact-checks, pooled and per agent | **Extract**, with the ordering rule below (torch-free) |
| Prior-anchored residual learners: a residual linear model and a residual MLP, trained by full-batch Adam from the prior logit | The linear and MLP field arms | The study's MLP learners, and linear learners through D2b | **Extract** (`torch` extra) |
| Prior-boosted regression trees from the prior logit | The tree field arm | D1 only: feasibility, and the self-replay nesting check | **Extract** (`research` extra, scikit-learn), noting that its second use is the weakest |

## What stays domain-owned

| Item | Why |
|---|---|
| Evaluating a fixed rule | Not numerically portable. Pivot's per-agent torch rule and the core's numpy prior differ in the last bits (5,452 of 20,000 rows, at most 2.2 × 10⁻¹⁶). The package takes a prior as weights; each domain computes its own rule. |
| An agent's own policy parameters | Pivot also holds the prior as each agent's parameters (reported as `mean_attention_weight`). The core does not, and the information-spread agents have none. |
| The exact Newton linear learner | Information spread only |
| Batched per-agent residuals and partial pooling | Information spread only |
| Replay of a policy trajectory as a table (`PolicySnapshots`) | Information spread only. It is central to that study's result, but has one consumer. |
| Cross-fitted folds | Pivot only (F04) |
| What decline rows mean, and label-selection views | Information spread only. The record keeps a generic integer kind per row, with kind 1 in the default view; what a kind means stays with the caller. |
| The nesting-check harness | Both consumers check nesting, but in domain tests. The pattern is documented, not an API: learning off reproduces the fixed rule. |
| Worlds, measures, archetypes, LLM elicitation, and the unverified-item measure | Domain semantics |

Out-of-sample scoring is already in the package through Decision 0016 and is
unchanged.

## The ordering rule

The core returned outcome rows in insertion order. Pivot buffers delayed
outcomes and adds them in release order. At immediate visibility the two
orders agree. With delays, row order changes how a full-batch learner's rows
are stacked, so it can change the fitted bytes.

**The rule.** The package's record returns rows ordered by the step from
which they are visible, then by insertion. Two cases follow:

- **For a caller that adds outcomes when they are released,** this is
  insertion order. Both consumers did so. The information-spread
  simulator's decision digests are unchanged on the package record.
- **For a caller that adds outcomes when they are realized,** each tagged
  with its release step, the rule gives release order.

**The N15 parity run** checked both cases against the nonlinear policy ABM
v2's field-public arms under failure delays. With either insertion, all 432
rows matched the record. Without the rule, event insertion differed under
both delays. The record is
`experiments/evidence/results/n15_package_parity_20261020.summary.md`
(repository-only).

## Placement

The placement follows Decision 0014.

- **`neural_abm.policy_ladder`** is torch-free and imports NumPy only. It
  holds:
  - `keyed_seed` and `keyed_uniform`;
  - `ExplorationFloor`;
  - `LinearPrior` and `PriorPredictor`;
  - `OutcomeRecord`;
  - `fit_prior_boosted_trees`, with `BoostedTreesPredictor`.

  `fit_prior_boosted_trees` imports scikit-learn on first use (the
  `research` extra). Every public name except `BoostedTreesPredictor` is
  re-exported through `neural_abm.api_lite`.
- **`neural_abm.policy_learners`** needs the `torch` extra. It holds:
  - `ResidualLinear` and `ResidualMLP`;
  - `TorchPredictor`;
  - `train_residual_policy` and `fit_residual_policy`.

  It is imported by module path. Neither `neural_abm.api` nor
  `import neural_abm` imports it.

**What changes from the frozen core.** The package changes no global state.
- The MLP draws its initial weights from a private `torch.Generator`, with
  the same draws as `torch.nn.Linear`'s default initialization under the
  seed.
- Thread counts and deterministic-algorithm settings are left to the caller.
  Bit-for-bit reproduction of a recorded run needs the settings it was
  recorded with.

## Dependency and Release Boundary

`neural_abm.policy_ladder` must not import `torch`, `scipy`, or
`scikit-learn` at import time. The released `v0.1.0a5` tag predates both
modules; `v0.1.0a6` is the first tag that contains them.

## Implementation Consequence

1. **Parity carried into the package.**
   - **Shared fixtures.** The package reproduces the frozen core byte for
     byte for every primitive and learner (`tests/test_policy_ladder_package.py`).
   - **The Pivot v3 record.** Through the package, the Pivot v3
     re-expression reproduced its recorded `runs.jsonl` byte for byte:
     1,152 of 1,152 rows.
   - **The ABM v2 record.** Its delayed field arms also reproduced: 432 of
     432 rows under each insertion mode.
2. **Frozen sources stay unchanged.** The core, the Pivot scripts, and the
   information-spread scripts are all bound by hash. New callers use the
   package; old records stay as they are.
3. **Ledger status.** The consolidation ledger's F09 (prior slot) and F10
   (exploration floor) are extracted, in a narrower form:
   - the prior is a weight vector, not a slot that replaces an agent's
     parameters;
   - the floor is the keyed test, not an adapter.
4. **No release.** A release needs a separate decision.

## Non-Goals

This decision does not:

- add a model-comparison, table-replay, or statistical-testing framework;
- extract fixed-rule evaluation, agent parameters, cross-fitting, partial
  pooling, or any domain measure;
- change any frozen script, recorded result, or released tag;
- claim that a neural policy is necessary, or that the closed loop matters
  in general. The information-spread study found that a table written before
  the run sufficed in its stationary worlds.
